"""Document ingestion pipeline bridging processing to storage layers."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from loguru import logger

from ..core.config import LegalMetadataConfig, get_settings
from ..embeddings import OpenAIEmbeddingModel
from ..knowledge_graph import GraphIngestionManager
from ..processing.cross_reference_extractor import extract_cross_references
from ..processing.pipeline import (
    DocumentProcessingPipeline,
    PipelineConfig as ProcessingConfig,
    ProcessingResult,
)
from ..retrieval.qdrant_manager import QdrantManager, VectorPoint


@dataclass
class DocumentIngestionConfig:
    """Configuration for the document ingestion pipeline."""

    qdrant_collection: str = "doctags_vectors"
    qdrant_distance_metric: str = "cosine"
    recreate_qdrant_collection: bool = False
    create_qdrant_collection: bool = True
    qdrant_batch_size: int = 64
    embedding_batch_size: int = 32
    store_sections: bool = True
    store_subsections: bool = True
    store_chunk_text: bool = True
    chunk_text_truncate: Optional[int] = None
    contextualize_embeddings: bool = True
    embedding_context_max_chars: int = 240
    embedding_include_breadcrumbs: bool = True
    embedding_include_tag_type: bool = True


@dataclass
class IngestionReport:
    """Summary of an ingestion run."""

    processed_documents: int = 0
    failed_documents: List[str] = field(default_factory=list)
    chunks_ingested: int = 0
    qdrant_vectors: int = 0
    neo4j_documents: int = 0
    sections_linked: int = 0
    subsections_linked: int = 0
    cross_references_stored: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "processed_documents": self.processed_documents,
            "failed_documents": self.failed_documents,
            "chunks_ingested": self.chunks_ingested,
            "qdrant_vectors": self.qdrant_vectors,
            "neo4j_documents": self.neo4j_documents,
            "sections_linked": self.sections_linked,
            "subsections_linked": self.subsections_linked,
            "cross_references_stored": self.cross_references_stored,
            "metadata": self.metadata,
        }


class DocumentIngestionPipeline:
    """Coordinate document processing, Neo4j ingestion, and Qdrant indexing."""

    def __init__(
        self,
        embeddings_model: Optional[Any] = None,
        *,
        processing_pipeline: Optional[DocumentProcessingPipeline] = None,
        processing_config: Optional[ProcessingConfig] = None,
        qdrant_manager: Optional[QdrantManager] = None,
        graph_ingestor: Optional[GraphIngestionManager] = None,
        config: Optional[DocumentIngestionConfig] = None,
    ) -> None:
        if embeddings_model is not None and not hasattr(embeddings_model, "encode"):
            raise ValueError("embeddings_model must expose an encode(texts) method")

        self.config = config or DocumentIngestionConfig()

        if processing_pipeline is not None:
            self.processing_pipeline = processing_pipeline
        else:
            proc_config = processing_config or ProcessingConfig()
            self.processing_pipeline = DocumentProcessingPipeline(proc_config)

        self._qdrant_manager = qdrant_manager
        self.graph_ingestor = graph_ingestor or GraphIngestionManager(
            store_sections=self.config.store_sections,
            store_subsections=self.config.store_subsections,
        )

        # If the caller passed their own graph ingestor with a managed Neo4j
        # connection we should reuse it for closing.
        self._owns_graph_ingestor = graph_ingestor is None

        self._qdrant_collection_ready = False
        self._embedding_dim: Optional[int] = None

        if embeddings_model is not None:
            self.embeddings_model = embeddings_model
        else:
            settings = get_settings()
            provider = getattr(settings.embeddings, "provider", "openai")
            if provider != "openai":
                raise ValueError(
                    "DocumentIngestionPipeline requires an embeddings_model when provider != 'openai'"
                )
            self.embeddings_model = OpenAIEmbeddingModel(
                model_name=settings.embeddings.model,
                api_key=settings.embeddings.api_key or os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def process_files(
        self,
        files: Sequence[Path],
        *,
        legal_metadata: Optional[LegalMetadataConfig] = None,
    ) -> IngestionReport:
        """Process individual files and ingest them."""

        results: List[ProcessingResult] = []
        for file_path in files:
            result = self.processing_pipeline.process_file(file_path)
            results.append(result)
        return self.ingest_processing_results(results, legal_metadata=legal_metadata)

    def process_directory(
        self,
        directory: Path,
        *,
        recursive: bool = True,
        legal_metadata: Optional[LegalMetadataConfig] = None,
    ) -> IngestionReport:
        """Process an entire directory before ingestion."""

        results = self.processing_pipeline.process_directory(directory, recursive=recursive)
        return self.ingest_processing_results(results, legal_metadata=legal_metadata)

    def ingest_processing_results(
        self,
        results: Iterable[ProcessingResult],
        *,
        legal_metadata: Optional[LegalMetadataConfig] = None,
    ) -> IngestionReport:
        """Ingest already processed documents."""

        report = IngestionReport()
        for result in results:
            if not result.success or not result.chunks:
                identifier = str(result.file_path)
                logger.warning("Skipping ingestion for %s (stage=%s)", identifier, result.stage)
                report.failed_documents.append(identifier)
                continue

            try:
                chunks = result.chunks
                doc_props = self._build_document_properties(result, legal_metadata=legal_metadata)

                embeddings = self._generate_embeddings(
                    chunks,
                    doc_title=doc_props.get("title"),
                )
                if embeddings.size == 0:
                    raise ValueError("No embeddings generated for chunks")

                embedding_dim = embeddings.shape[1]
                self._ensure_qdrant_collection(embedding_dim)

                embedding_vectors = [
                    [float(x) for x in vector]
                    for vector in embeddings
                ]

                chunk_payloads = self._build_chunk_payloads(result, embedding_vectors)

                qdrant_points = self._build_qdrant_points(
                    result, embedding_vectors, chunk_payloads, legal_metadata=legal_metadata
                )
                inserted = self.qdrant_manager.insert_vectors_batch(
                    qdrant_points,
                    collection_name=self.config.qdrant_collection,
                    batch_size=self.config.qdrant_batch_size,
                )

                graph_stats = self.graph_ingestor.ingest_document(doc_props, chunk_payloads)

                # Extract and store cross-references between provisions
                xref_count = self._store_cross_references(chunk_payloads)

                report.processed_documents += 1
                report.qdrant_vectors += inserted
                report.neo4j_documents += 1
                report.chunks_ingested += graph_stats.chunks_processed
                report.sections_linked += graph_stats.sections_linked
                report.subsections_linked += graph_stats.subsections_linked
                report.cross_references_stored += xref_count

                logger.success(
                    "Ingested %s (%d chunks, %d vectors, %d cross-refs)",
                    doc_props.get("title") or doc_props["doc_id"],
                    graph_stats.chunks_processed,
                    inserted,
                    xref_count,
                )

            except Exception as exc:  # pragma: no cover - defensive logging
                identifier = doc_props.get("doc_id") if 'doc_props' in locals() else str(result.file_path)
                logger.error("Failed to ingest %s: %s", identifier, exc)
                report.failed_documents.append(identifier)

        report.metadata["completed_at"] = datetime.now(UTC).isoformat()
        return report

    def close(self) -> None:
        """Close any resources created by the pipeline."""
        if self._qdrant_manager is not None:
            self._qdrant_manager.close()

        if self._owns_graph_ingestor:
            self.graph_ingestor.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @property
    def qdrant_manager(self) -> QdrantManager:
        if self._qdrant_manager is None:
            self._qdrant_manager = QdrantManager()
        return self._qdrant_manager

    def _generate_embeddings(
        self,
        chunks: Sequence[Any],
        doc_title: Optional[str] = None,
    ) -> np.ndarray:
        texts = [self._build_embedding_text(chunk, doc_title=doc_title) for chunk in chunks]
        embeddings = []
        batch_size = max(1, self.config.embedding_batch_size)

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            vectors = self.embeddings_model.encode(batch, show_progress_bar=False)
            embeddings.append(np.asarray(vectors, dtype=np.float32))

        if embeddings:
            return np.vstack(embeddings)
        return np.zeros((0, 0), dtype=np.float32)

    def _build_embedding_text(
        self,
        chunk: Any,
        doc_title: Optional[str] = None,
    ) -> str:
        """Build context-rich text used for embedding generation."""
        content = (getattr(chunk, "content", "") or "").strip()
        if not self.config.contextualize_embeddings:
            return content

        context = getattr(chunk, "context", {}) or {}
        metadata = getattr(chunk, "metadata", {}) or {}

        context_lines: List[str] = []
        title = context.get("document_title") or doc_title
        if title:
            context_lines.append(f"Document title: {title}")

        section = context.get("section")
        if section:
            context_lines.append(f"Section: {section}")

        subsection = context.get("subsection")
        if subsection:
            context_lines.append(f"Subsection: {subsection}")

        if self.config.embedding_include_breadcrumbs:
            breadcrumbs = context.get("breadcrumbs")
            if breadcrumbs:
                context_lines.append(f"Path: {breadcrumbs}")

        if self.config.embedding_include_tag_type:
            tag_type = metadata.get("tag_type")
            if tag_type:
                context_lines.append(f"Element type: {tag_type}")

        if not context_lines:
            return content

        prefix = "\n".join(context_lines)
        max_chars = max(0, int(self.config.embedding_context_max_chars))
        if max_chars and len(prefix) > max_chars:
            prefix = prefix[: max_chars - 3] + "..."

        return f"{prefix}\n\nContent:\n{content}"

    def _ensure_qdrant_collection(self, vector_dim: int) -> None:
        if self._qdrant_collection_ready:
            return

        if vector_dim <= 0:
            raise ValueError("Embedding dimension must be positive")

        if self.config.create_qdrant_collection:
            logger.info(
                "Ensuring Qdrant collection '%s' (dim=%d)",
                self.config.qdrant_collection,
                vector_dim,
            )
            self.qdrant_manager.create_collection(
                collection_name=self.config.qdrant_collection,
                vector_size=vector_dim,
                distance_metric=self.config.qdrant_distance_metric,
                recreate=self.config.recreate_qdrant_collection,
            )

        if hasattr(self.qdrant_manager, "get_collection_info"):
            info = self.qdrant_manager.get_collection_info(self.config.qdrant_collection)
            existing_dim = info.get("vector_size") if info else None
            if existing_dim is not None:
                try:
                    existing_dim_int = int(existing_dim)
                except (TypeError, ValueError):
                    existing_dim_int = None
                if existing_dim_int and existing_dim_int != int(vector_dim):
                    raise ValueError(
                        "Qdrant collection vector size mismatch "
                        f"(collection={existing_dim_int}, embeddings={vector_dim}). "
                        "Recreate the collection or align embedding dimensions."
                    )

        self._qdrant_collection_ready = True
        self._embedding_dim = vector_dim

    def _build_document_properties(
        self,
        result: ProcessingResult,
        *,
        legal_metadata: Optional[LegalMetadataConfig] = None,
    ) -> Dict[str, Any]:
        doc = result.doctags_doc
        parsed_metadata = result.parsed_doc.metadata if result.parsed_doc else {}

        metadata = {
            "processing": result.metadata,
            "parsed": parsed_metadata,
        }

        props: Dict[str, Any] = {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "source_path": str(result.file_path),
            "num_chunks": len(result.chunks or []),
            "num_tags": len(doc.tags) if doc else 0,
            "num_elements": result.metadata.get("num_elements"),
            "file_type": result.metadata.get("file_type"),
            "metadata": metadata,
            "processed_at": datetime.now(UTC).isoformat(),
        }

        # Attach legal version/amendment metadata when provided (Step 5)
        if legal_metadata is not None:
            if legal_metadata.in_force_from is not None:
                props["in_force_from"] = legal_metadata.in_force_from
            if legal_metadata.in_force_until is not None:
                props["in_force_until"] = legal_metadata.in_force_until
            if legal_metadata.amended_by is not None:
                props["amended_by"] = legal_metadata.amended_by
            if legal_metadata.supersedes is not None:
                props["supersedes"] = legal_metadata.supersedes

        return props

    def _build_chunk_payloads(
        self,
        result: ProcessingResult,
        embedding_vectors: Optional[Sequence[Sequence[float]]] = None,
    ) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        truncate_at = self.config.chunk_text_truncate
        store_text = self.config.store_chunk_text
        source_path = str(result.file_path)

        embeddings_list = list(embedding_vectors or [])

        for index, chunk in enumerate(result.chunks or []):
            content = chunk.content or ""
            if store_text and truncate_at and len(content) > truncate_at:
                content = content[:truncate_at]
            elif not store_text:
                content = ""

            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "content": content,
                "metadata": chunk.metadata or {},
                "context": chunk.context or {},
                "source_path": source_path,
            }

            if embeddings_list and index < len(embeddings_list):
                chunk_dict["embedding"] = list(embeddings_list[index])

            payloads.append(chunk_dict)

        return payloads

    def _build_qdrant_points(
        self,
        result: ProcessingResult,
        embedding_vectors: Sequence[Sequence[float]],
        chunk_payloads: Sequence[Dict[str, Any]],
        *,
        legal_metadata: Optional[LegalMetadataConfig] = None,
    ) -> List[VectorPoint]:
        doc_title = result.doctags_doc.title if result.doctags_doc else None
        points: List[VectorPoint] = []

        # Build legal payload extension once (same for all chunks in document)
        legal_payload: Dict[str, Any] = {}
        if legal_metadata is not None:
            if legal_metadata.in_force_from is not None:
                legal_payload["in_force_from"] = legal_metadata.in_force_from
            if legal_metadata.in_force_until is not None:
                legal_payload["in_force_until"] = legal_metadata.in_force_until
            if legal_metadata.amended_by is not None:
                legal_payload["amended_by"] = legal_metadata.amended_by
            if legal_metadata.supersedes is not None:
                legal_payload["supersedes"] = legal_metadata.supersedes

        for idx, (embedding, chunk) in enumerate(zip(embedding_vectors, chunk_payloads)):
            context = chunk.get("context", {})
            metadata = {
                "doc_id": chunk.get("doc_id"),
                "chunk_id": chunk.get("chunk_id"),
                "chunk_index": chunk.get("chunk_index"),
                "document_title": doc_title,
                "section": context.get("section"),
                "subsection": context.get("subsection"),
                "breadcrumbs": context.get("breadcrumbs"),
                "source_path": chunk.get("source_path"),
                "char_start": chunk.get("char_start"),
                "char_end": chunk.get("char_end"),
                "metadata": chunk.get("metadata", {}),
                "context": context,
                **legal_payload,
            }

            if chunk.get("content"):
                metadata["text"] = chunk["content"]

            points.append(
                VectorPoint(
                    id=chunk.get("chunk_id"),
                    vector=[float(x) for x in embedding],
                    metadata=metadata,
                )
            )

        return points

    def _store_cross_references(self, chunk_payloads: Sequence[Dict[str, Any]]) -> int:
        """Extract and store cross-reference edges in Neo4j for all chunks.

        Iterates each chunk payload, runs the cross-reference extractor, then
        writes REFERENCES edges via the graph ingestor's Neo4j connection.
        Failures are logged as warnings and do not interrupt ingestion.

        Returns:
            Total number of cross-reference edges stored.
        """
        all_refs = []
        for chunk in chunk_payloads:
            chunk_id = chunk.get("chunk_id", "")
            content = chunk.get("content", "")
            doc_id = chunk.get("doc_id", "")
            if not chunk_id or not content:
                continue
            refs = extract_cross_references(chunk_id, content, doc_id)
            all_refs.extend(refs)

        if not all_refs:
            return 0

        try:
            stored = self.graph_ingestor.neo4j.store_cross_references(all_refs)
            return stored
        except Exception as exc:
            logger.warning("Cross-reference storage failed (non-fatal): %s", exc)
            return 0
