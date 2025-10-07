"""
Knowledge Graph Pipeline for End-to-End Graph Construction.

Orchestrates the complete knowledge graph construction process:
1. Document ingestion
2. Entity extraction
3. Relationship extraction
4. Entity resolution
5. Graph construction
6. Cross-document linking
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from tqdm import tqdm

from loguru import logger

from .entity_extractor import EntityExtractor, EntityExtractionResult
from .relationship_extractor import RelationshipExtractor, RelationshipExtractionResult
from .entity_resolver import EntityResolver, ResolutionResult
from .graph_builder import GraphBuilder, DocumentMetadata, ChunkMetadata, GraphBuildResult
from .neo4j_manager import Neo4jManager
from ..core.config import get_settings


@dataclass
class PipelineConfig:
    """Configuration for knowledge graph pipeline."""
    extract_entities: bool = True
    extract_relationships: bool = True
    resolve_entities: bool = True
    use_llm: bool = False
    batch_size: int = 10
    confidence_threshold: float = 0.7
    enable_progress_bar: bool = True


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    documents_processed: int
    total_entities: int
    unique_entities: int
    total_relationships: int
    nodes_created: int
    edges_created: int
    processing_time: float
    statistics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class KnowledgeGraphPipeline:
    """
    Orchestrates complete knowledge graph construction pipeline.

    Features:
    - Multi-stage processing (extraction → resolution → construction)
    - Batch processing for efficiency
    - Progress tracking
    - Error recovery
    - Incremental updates
    - Statistics generation
    """

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        neo4j_manager: Optional[Neo4jManager] = None
    ):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
            neo4j_manager: Neo4j manager instance
        """
        self.config = config or PipelineConfig()
        self.settings = get_settings()

        # Initialize components
        self.neo4j_manager = neo4j_manager
        self.graph_builder: Optional[GraphBuilder] = None
        self._graph_disabled = False
        self._graph_disable_reason: Optional[str] = None

        if self.config.extract_entities:
            self.entity_extractor = EntityExtractor(
                use_llm=self.config.use_llm,
                confidence_threshold=self.config.confidence_threshold,
                batch_size=self.config.batch_size
            )
        else:
            self.entity_extractor = None

        if self.config.extract_relationships:
            self.relationship_extractor = RelationshipExtractor(
                use_llm=self.config.use_llm,
                confidence_threshold=self.config.confidence_threshold
            )
        else:
            self.relationship_extractor = None

        if self.config.resolve_entities:
            self.entity_resolver = EntityResolver(
                similarity_threshold=self.config.confidence_threshold,
                use_embeddings=True,
                algorithm="hybrid"
            )
        else:
            self.entity_resolver = None

        logger.info("Knowledge graph pipeline initialized")

    def process_document(
        self,
        text: str,
        doc_id: str,
        doc_metadata: Optional[Dict[str, Any]] = None,
        chunks: Optional[List[Dict[str, Any]]] = None
    ) -> GraphBuildResult:
        """
        Process a single document and build its graph.

        Args:
            text: Document text
            doc_id: Document identifier
            doc_metadata: Optional document metadata
            chunks: Optional pre-chunked text segments

        Returns:
            GraphBuildResult
        """
        logger.info(f"Processing document: {doc_id}")

        # Prepare document metadata
        metadata = DocumentMetadata(
            doc_id=doc_id,
            title=doc_metadata.get("title") if doc_metadata else None,
            source=doc_metadata.get("source") if doc_metadata else None,
            content_type=doc_metadata.get("content_type") if doc_metadata else None,
            tags=doc_metadata.get("tags", []) if doc_metadata else [],
            processed_at=datetime.now(),
            extra=doc_metadata or {}
        )

        # Prepare chunks
        if chunks is None:
            # Simple chunking if not provided
            chunks = self._simple_chunk(text, doc_id)

        chunk_metas = [
            ChunkMetadata(
                chunk_id=chunk.get("chunk_id", f"{doc_id}_chunk_{i}"),
                doc_id=doc_id,
                text=chunk["text"],
                start_char=chunk.get("start_char", 0),
                end_char=chunk.get("end_char", len(chunk["text"])),
                section_id=chunk.get("section_id"),
                embedding=chunk.get("embedding")
            )
            for i, chunk in enumerate(chunks)
        ]

        # Stage 1: Entity extraction
        entity_result = None
        if self.entity_extractor:
            logger.debug(f"Extracting entities from {doc_id}")
            entity_result = self.entity_extractor.extract_entities(
                text=text,
                document_id=doc_id,
                extract_attributes=True,
                include_context=True
            )
            logger.debug(f"Extracted {len(entity_result.entities)} entities")
        else:
            from .entity_extractor import EntityExtractionResult
            entity_result = EntityExtractionResult(entities=[], document_id=doc_id)

        # Stage 2: Relationship extraction
        relationship_result = None
        if self.relationship_extractor and entity_result.entities:
            logger.debug(f"Extracting relationships from {doc_id}")
            relationship_result = self.relationship_extractor.extract_relationships(
                text=text,
                entities=entity_result.entities,
                document_id=doc_id
            )
            logger.debug(f"Extracted {len(relationship_result.relationships)} relationships")
        else:
            from .relationship_extractor import RelationshipExtractionResult
            relationship_result = RelationshipExtractionResult(relationships=[], document_id=doc_id)

        # Stage 3: Entity resolution
        resolution_result = None
        if self.entity_resolver and entity_result.entities:
            logger.debug(f"Resolving entities for {doc_id}")
            resolution_result = self.entity_resolver.resolve_entities(
                entities=entity_result.entities
            )
            logger.debug(f"Resolved to {resolution_result.unique_entities} unique entities")

        # Stage 4: Graph construction
        logger.debug(f"Building graph for {doc_id}")
        graph_builder = self._get_graph_builder()

        if graph_builder is None:
            logger.warning(
                "Neo4j unavailable, skipping graph build for %s (reason: %s)",
                doc_id,
                self._graph_disable_reason or "unknown",
            )

            return GraphBuildResult(
                nodes_created=0,
                relationships_created=0,
                entities_linked=0,
                documents_processed=1,
                statistics={
                    "graph_disabled": True,
                    "graph_disable_reason": self._graph_disable_reason,
                    "entities": len(entity_result.entities),
                    "relationships": len(relationship_result.relationships),
                },
            )

        build_result = graph_builder.build_document_graph(
            doc_metadata=metadata,
            chunks=chunk_metas,
            entity_result=entity_result,
            relationship_result=relationship_result,
            resolution_result=resolution_result
        )

        logger.info(f"Completed processing document {doc_id}")
        return build_result

    def process_documents_batch(
        self,
        documents: List[Dict[str, Any]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> PipelineResult:
        """
        Process multiple documents in batch.

        Args:
            documents: List of document dicts with 'text', 'doc_id', and optional metadata
            progress_callback: Optional callback for progress updates

        Returns:
            PipelineResult with aggregated statistics
        """
        start_time = datetime.now()
        logger.info(f"Starting batch processing of {len(documents)} documents")

        results = []
        errors = []

        # Process documents
        iterator = tqdm(documents, disable=not self.config.enable_progress_bar)
        for i, doc in enumerate(iterator):
            try:
                result = self.process_document(
                    text=doc["text"],
                    doc_id=doc["doc_id"],
                    doc_metadata=doc.get("metadata"),
                    chunks=doc.get("chunks")
                )
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(documents))

            except Exception as e:
                error_msg = f"Error processing document {doc.get('doc_id', 'unknown')}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)
                continue

        # Create cross-document links
        if len(results) > 1 and self._get_graph_builder():
            logger.info("Creating cross-document links")
            doc_ids = [doc["doc_id"] for doc in documents]
            graph_builder = self._get_graph_builder()
            if graph_builder:
                graph_builder.create_cross_document_links(doc_ids)

        # Aggregate results
        processing_time = (datetime.now() - start_time).total_seconds()

        pipeline_result = PipelineResult(
            documents_processed=len(results),
            total_entities=sum(r.statistics.get("entities", 0) for r in results),
            unique_entities=sum(r.statistics.get("unique_entities", 0) for r in results),
            total_relationships=sum(r.statistics.get("relationships", 0) for r in results),
            nodes_created=sum(r.nodes_created for r in results),
            edges_created=sum(r.relationships_created for r in results),
            processing_time=processing_time,
            statistics=self._aggregate_statistics(results),
            errors=errors
        )

        logger.info(
            f"Batch processing complete: {pipeline_result.documents_processed} documents, "
            f"{pipeline_result.nodes_created} nodes, {pipeline_result.edges_created} edges "
            f"in {processing_time:.2f}s"
        )

        return pipeline_result

    def update_document(
        self,
        text: str,
        doc_id: str,
        doc_metadata: Optional[Dict[str, Any]] = None,
        chunks: Optional[List[Dict[str, Any]]] = None
    ) -> GraphBuildResult:
        """
        Update an existing document in the graph.

        Args:
            text: Updated document text
            doc_id: Document identifier
            doc_metadata: Optional updated metadata
            chunks: Optional updated chunks

        Returns:
            GraphBuildResult
        """
        logger.info(f"Updating document: {doc_id}")

        # Remove old document and its connections
        query = """
        MATCH (d:Document {doc_id: $doc_id})
        OPTIONAL MATCH (d)-[r1]->(c:Chunk)
        OPTIONAL MATCH (c)-[r2]->(e:Entity)
        OPTIONAL MATCH (d)-[r3]->(e2:Entity)
        DETACH DELETE d, c
        """

        graph_builder = self._get_graph_builder()
        if not graph_builder:
            raise RuntimeError(
                "Neo4j is not available; cannot update existing graph documents"
            )

        graph_builder.neo4j_manager.execute_write_query(query, {"doc_id": doc_id})

        # Process document as new
        return self.process_document(text, doc_id, doc_metadata, chunks)

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the graph.

        Args:
            doc_id: Document identifier

        Returns:
            True if deleted successfully
        """
        logger.info(f"Deleting document: {doc_id}")

        query = """
        MATCH (d:Document {doc_id: $doc_id})
        OPTIONAL MATCH (d)-[r1]->(c:Chunk)
        OPTIONAL MATCH (c)-[r2]->(e:Entity)
        OPTIONAL MATCH (d)-[r3]->(e2:Entity)
        DETACH DELETE d, c
        RETURN count(d) as deleted_count
        """

        graph_builder = self._get_graph_builder()
        if not graph_builder:
            raise RuntimeError(
                "Neo4j is not available; cannot delete graph documents"
            )

        result = graph_builder.neo4j_manager.execute_write_query(query, {"doc_id": doc_id})
        deleted = result[0]["deleted_count"] > 0 if result else False

        if deleted:
            logger.info(f"Deleted document {doc_id}")
        else:
            logger.warning(f"Document {doc_id} not found")

        return deleted

    def process_from_file(
        self,
        file_path: Path,
        doc_id: Optional[str] = None
    ) -> GraphBuildResult:
        """
        Process a document from a file.

        Args:
            file_path: Path to document file
            doc_id: Optional document ID (uses filename if not provided)

        Returns:
            GraphBuildResult
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        doc_id = doc_id or file_path.stem

        # Read file
        if file_path.suffix == ".json":
            with open(file_path) as f:
                data = json.load(f)
                text = data.get("text", "")
                metadata = data.get("metadata", {})
                chunks = data.get("chunks")
        else:
            with open(file_path) as f:
                text = f.read()
                metadata = {"source": str(file_path)}
                chunks = None

        return self.process_document(text, doc_id, metadata, chunks)

    def _simple_chunk(
        self,
        text: str,
        doc_id: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """Simple text chunking."""
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                "text": chunk_text,
                "start_char": start,
                "end_char": end
            })

            start = end - overlap

        return chunks

    def _get_graph_builder(self) -> Optional[GraphBuilder]:
        """Lazily initialize the graph builder when Neo4j is available."""
        if self._graph_disabled:
            return None

        if self.graph_builder:
            return self.graph_builder

        manager = self.neo4j_manager

        if manager is None:
            try:
                manager = Neo4jManager()
                self.neo4j_manager = manager
            except Exception as err:  # pragma: no cover - depends on environment
                reason = str(err)
                logger.warning(f"Neo4j unavailable during initialization: {reason}")
                self._graph_disabled = True
                self._graph_disable_reason = reason
                return None

        try:
            self.graph_builder = GraphBuilder(neo4j_manager=manager)
        except Exception as err:  # pragma: no cover - depends on environment
            reason = str(err)
            logger.warning(f"Failed to initialize graph builder: {reason}")
            self._graph_disabled = True
            self._graph_disable_reason = reason
            return None

        return self.graph_builder

    def _aggregate_statistics(
        self,
        results: List[GraphBuildResult]
    ) -> Dict[str, Any]:
        """Aggregate statistics from multiple results."""
        stats = {
            "total_documents": len(results),
            "total_nodes": sum(r.nodes_created for r in results),
            "total_edges": sum(r.relationships_created for r in results),
            "avg_entities_per_doc": sum(
                r.statistics.get("entities", 0) for r in results
            ) / len(results) if results else 0,
            "avg_relationships_per_doc": sum(
                r.statistics.get("relationships", 0) for r in results
            ) / len(results) if results else 0
        }

        return stats

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """Get overall pipeline statistics."""
        graph_builder = self._get_graph_builder()
        if graph_builder is None:
            graph_stats = {
                "graph_disabled": True,
                "reason": self._graph_disable_reason,
            }
        else:
            graph_stats = graph_builder.get_statistics()

        stats = {
            "graph_stats": graph_stats,
            "config": {
                "extract_entities": self.config.extract_entities,
                "extract_relationships": self.config.extract_relationships,
                "resolve_entities": self.config.resolve_entities,
                "use_llm": self.config.use_llm,
                "confidence_threshold": self.config.confidence_threshold
            }
        }

        return stats

    def export_pipeline_config(self, output_path: Path) -> None:
        """Export pipeline configuration to file."""
        config_dict = {
            "extract_entities": self.config.extract_entities,
            "extract_relationships": self.config.extract_relationships,
            "resolve_entities": self.config.resolve_entities,
            "use_llm": self.config.use_llm,
            "batch_size": self.config.batch_size,
            "confidence_threshold": self.config.confidence_threshold
        }

        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Exported pipeline config to {output_path}")
