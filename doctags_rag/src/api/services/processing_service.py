"""Processing service orchestrating document ingestion for the demo API."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
from uuid import uuid4
import os

from loguru import logger

from src.processing import DocumentProcessingPipeline, PipelineConfig
from src.processing.pipeline import ProcessingResult
from src.processing.doctags_processor import DocTagsConverter

from ..models import (
    DocumentInfo,
    DocumentUploadRequest,
    ProcessingStatus,
    UploadResponse,
)
from .document_store import PersistentDocumentStore, StoredDocumentPayload


@dataclass
class StoredDocument:
    """Materialised document data kept in memory for the demo UI."""
    info: DocumentInfo
    chunks: List[Dict[str, Any]]
    markdown: Optional[str]
    text_preview: Optional[str]
    doctags: Optional[Dict[str, Any]]
    message: Optional[str] = None


class ProcessingService:
    """High-level service handling document processing and in-memory storage."""

    def __init__(self) -> None:
        self._pipeline_cache: Dict[Tuple[Any, ...], DocumentProcessingPipeline] = {}
        self._documents: "OrderedDict[str, StoredDocument]" = OrderedDict()
        self._lock = Lock()
        self._semantic_support: Dict[str, Any] = DocumentProcessingPipeline.semantic_support_status()
        self._document_cache_max = max(1, int(os.getenv("DOCTAGS_DOCUMENT_CACHE_MAX", "250")))
        self._document_list_limit = max(1, int(os.getenv("DOCTAGS_DOCUMENT_LIST_LIMIT", "1000")))
        store_path = os.getenv("DOCTAGS_DOCUMENT_STORE_PATH", "data/storage/documents.db")
        self._document_store = PersistentDocumentStore(db_path=store_path)
        self._load_persisted_documents()

    def process_document(
        self,
        file_path: Path,
        original_filename: str,
        request: DocumentUploadRequest,
        file_size_bytes: int
    ) -> UploadResponse:
        """Run the processing pipeline for an uploaded document."""
        pipeline = self._get_pipeline(request)
        logger.info(
            "Processing upload %s with chunk_size=%s overlap=%s method=%s",
            original_filename,
            request.chunk_size,
            request.chunk_overlap,
            request.chunking_method.value,
        )

        start_ts = datetime.utcnow()
        result: Optional[ProcessingResult] = None
        error_message: Optional[str] = None
        error_reference: Optional[str] = None

        try:
            result = pipeline.process_file(file_path)
        except Exception as exc:  # pragma: no cover
            error_reference = f"proc_{uuid4().hex[:10]}"
            logger.exception(
                "Pipeline execution crashed (reference=%s): %s",
                error_reference,
                exc,
            )
            error_message = str(exc)

        elapsed = (datetime.utcnow() - start_ts).total_seconds()

        if not result or not result.success:
            return self._handle_failure(
                original_filename=original_filename,
                file_size_bytes=file_size_bytes,
                elapsed=elapsed,
                result=result,
                error_message=error_message,
                error_reference=error_reference,
            )

        return self._handle_success(
            original_filename=original_filename,
            file_size_bytes=file_size_bytes,
            elapsed=elapsed,
            result=result,
            request=request,
        )

    def list_documents(self) -> List[DocumentInfo]:
        """Return persisted documents sorted by upload time (newest first)."""
        try:
            payloads = self._document_store.list_document_infos(limit=self._document_list_limit)
            documents: List[DocumentInfo] = []
            for payload in payloads:
                if not payload:
                    continue
                try:
                    documents.append(DocumentInfo.model_validate(payload))
                except Exception as exc:
                    logger.warning("Skipping invalid persisted document info: %s", exc)
            return documents
        except Exception as exc:
            logger.warning("Unable to list persisted documents; using in-memory cache: %s", exc)
            with self._lock:
                docs = [stored.info for stored in self._documents.values()]
            return sorted(docs, key=lambda item: item.uploaded_at, reverse=True)

    def get_document(self, document_id: str) -> Optional[StoredDocument]:
        """Retrieve stored document by ID."""
        doc_id = str(document_id).strip()
        if not doc_id:
            return None

        with self._lock:
            cached = self._documents.get(doc_id)
            if cached is not None:
                self._documents.move_to_end(doc_id)
                return cached

        payload = self._document_store.get(doc_id)
        if payload is None:
            return None

        try:
            stored = self._materialize_payload(payload)
        except Exception as exc:
            logger.warning("Unable to decode persisted document %s: %s", doc_id, exc)
            return None

        with self._lock:
            self._documents[doc_id] = stored
            self._documents.move_to_end(doc_id)
            self._enforce_document_cache_limit_locked()
        return stored

    def total_documents(self) -> int:
        """Return total persisted document count."""
        try:
            return self._document_store.count_documents()
        except Exception as exc:
            logger.warning("Unable to count persisted documents; using in-memory cache: %s", exc)
            with self._lock:
                return len(self._documents)

    def _get_pipeline(self, request: DocumentUploadRequest) -> DocumentProcessingPipeline:
        """Return a cached pipeline matching the request configuration."""
        cache_key = (
            request.enable_ocr,
            request.chunk_size,
            request.chunk_overlap,
            request.chunking_method.value,
        )

        with self._lock:
            pipeline = self._pipeline_cache.get(cache_key)
            if pipeline:
                return pipeline

            config = PipelineConfig(
                enable_ocr=request.enable_ocr,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                chunking_method=request.chunking_method.value,
                semantic_model=os.getenv('DOCTAGS_SEMANTIC_MODEL'),
            )
            pipeline = DocumentProcessingPipeline(config)
            self._pipeline_cache[cache_key] = pipeline
            return pipeline

    def _handle_failure(
        self,
        original_filename: str,
        file_size_bytes: int,
        elapsed: float,
        result: Optional[ProcessingResult],
        error_message: Optional[str],
        error_reference: Optional[str] = None,
    ) -> UploadResponse:
        """Create response for failed processing attempts."""
        uploaded_at = datetime.utcnow()
        doc_id = f"doc_{uuid4().hex[:8]}"
        raw_error = (error_message or (result.error if result else "Unknown error")).strip()
        reference = error_reference or f"proc_{uuid4().hex[:10]}"
        public_error = self._build_public_failure_message(raw_error, reference)

        metadata: Dict[str, Any] = {
            "stage": result.stage.value if result else "unknown",
            "error_reference": reference,
        }

        info = DocumentInfo(
            id=doc_id,
            filename=original_filename,
            file_type=(Path(original_filename).suffix or "").lstrip(".") or "unknown",
            file_size_bytes=file_size_bytes,
            num_chunks=0,
            num_entities=None,
            processing_time=elapsed,
            status=ProcessingStatus.FAILED,
            uploaded_at=uploaded_at,
            metadata=metadata,
        )

        stored = StoredDocument(
            info=info,
            chunks=[],
            markdown=None,
            text_preview=None,
            doctags=None,
            message=public_error,
        )

        with self._lock:
            self._documents[doc_id] = stored

        self._persist_document(doc_id, stored)

        return UploadResponse(
            success=False,
            document=info,
            message=public_error,
            processing_time=elapsed,
        )

    def _build_public_failure_message(self, raw_error: str, reference: str) -> str:
        """Map internal failures to safe client-visible messages."""
        normalized = (raw_error or "").strip()
        lowered = normalized.lower()
        if lowered.startswith("file too large"):
            return normalized
        if lowered.startswith("unsupported file type"):
            return normalized
        if lowered.startswith("file validation failed"):
            return "File validation failed"
        return f"Processing failed (reference: {reference})"

    def _handle_success(
        self,
        original_filename: str,
        file_size_bytes: int,
        elapsed: float,
        result: ProcessingResult,
        request: DocumentUploadRequest,
    ) -> UploadResponse:
        """Create response for successful processing attempts."""
        uploaded_at = datetime.utcnow()
        doc_id = result.doctags_doc.doc_id if result.doctags_doc else f"doc_{uuid4().hex[:8]}"

        file_type = result.metadata.get("file_type") if result.metadata else None
        if not file_type:
            file_type = (Path(original_filename).suffix or "").lstrip(".") or "unknown"

        chunks = self._serialise_chunks(result)
        markdown = None
        doctags_payload = None
        text_preview = None

        if result.doctags_doc:
            doctags_payload = result.doctags_doc.to_dict()
            try:
                markdown = DocTagsConverter.to_markdown(result.doctags_doc)
            except Exception as exc:  # pragma: no cover
                logger.warning("Markdown conversion failed: %s", exc)
                markdown = None

        if result.parsed_doc and result.parsed_doc.text:
            text_preview = result.parsed_doc.text[:2000]

        tag_counts: Dict[str, int] = {}
        if result.doctags_doc:
            from collections import Counter

            tag_counts = dict(
                Counter(tag.tag_type.value for tag in result.doctags_doc.tags)
            )

        pipeline_metadata = result.metadata or {}
        actual_chunking_method = pipeline_metadata.get(
            "chunking_method",
            request.chunking_method.value,
        )
        requested_method = request.chunking_method.value
        semantic_error = pipeline_metadata.get("semantic_chunking_error")
        semantic_model = pipeline_metadata.get("semantic_model")

        metadata: Dict[str, Any] = {
            "stage": result.stage.value,
            "chunking_method": actual_chunking_method,
            "chunking_method_requested": requested_method,
            "chunk_size": request.chunk_size,
            "chunk_overlap": request.chunk_overlap,
            "enable_ocr": request.enable_ocr,
            "extract_entities": request.extract_entities,
            "build_raptor": request.build_raptor,
            "semantic_chunking_error": semantic_error,
            "semantic_model": semantic_model,
            "pipeline_metadata": pipeline_metadata,
            "tag_counts": tag_counts,
            "sample_chunks": [
                {
                    "chunk_id": chunk["chunk_id"],
                    "preview": chunk["content"][:200],
                    "length": chunk["length"],
                }
                for chunk in chunks[:5]
            ],
            "text_preview": text_preview,
        }

        info = DocumentInfo(
            id=doc_id,
            filename=original_filename,
            file_type=file_type,
            file_size_bytes=file_size_bytes,
            num_chunks=len(chunks),
            num_entities=None,
            processing_time=elapsed,
            status=ProcessingStatus.COMPLETED,
            uploaded_at=uploaded_at,
            metadata=metadata,
        )

        stored = StoredDocument(
            info=info,
            chunks=chunks,
            markdown=markdown,
            text_preview=text_preview,
            doctags=doctags_payload,
            message="Processed successfully",
        )

        with self._lock:
            self._documents[doc_id] = stored
            self._documents.move_to_end(doc_id)
            self._enforce_document_cache_limit_locked()
            if semantic_error:
                self._semantic_support = {
                    "available": False,
                    "reason": semantic_error,
                }
            elif actual_chunking_method == "semantic":
                self._semantic_support = {
                    "available": True,
                    "model": semantic_model,
                }

        self._persist_document(doc_id, stored)

        message = f"Processed {original_filename}"
        return UploadResponse(
            success=True,
            document=info,
            message=message,
            processing_time=elapsed,
        )

    @staticmethod
    def _serialise_chunks(result: ProcessingResult) -> List[Dict[str, Any]]:
        """Convert chunk objects into serialisable payloads."""
        chunks: List[Dict[str, Any]] = []
        if not result.chunks:
            return chunks

        for chunk in result.chunks:
            chunks.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "length": len(chunk.content),
                    "metadata": chunk.metadata or {},
                    "context": chunk.context or {},
                }
            )
        return chunks

    def semantic_support_status(self) -> Dict[str, Any]:
        """Expose cached semantic chunking support information."""
        with self._lock:
            return dict(self._semantic_support)

    def _persist_document(self, document_id: str, stored: StoredDocument) -> None:
        payload = StoredDocumentPayload(
            info=stored.info.model_dump(mode="json"),
            chunks=stored.chunks,
            markdown=stored.markdown,
            text_preview=stored.text_preview,
            doctags=stored.doctags,
            message=stored.message,
        )
        try:
            self._document_store.upsert(document_id=document_id, payload=payload)
        except Exception as exc:  # pragma: no cover - persistence is best effort
            logger.warning("Unable to persist document %s: %s", document_id, exc)

    def _load_persisted_documents(self) -> None:
        try:
            persisted = self._document_store.load_all(limit=self._document_cache_max)
        except Exception as exc:  # pragma: no cover - startup fallback
            logger.warning("Unable to load persisted documents: %s", exc)
            return

        if not persisted:
            return

        with self._lock:
            for document_id, payload in persisted.items():
                try:
                    self._documents[document_id] = self._materialize_payload(payload)
                    self._documents.move_to_end(document_id)
                except Exception as exc:
                    logger.warning("Unable to load persisted record %s: %s", document_id, exc)
            self._enforce_document_cache_limit_locked()

    def _materialize_payload(self, payload: StoredDocumentPayload) -> StoredDocument:
        info = DocumentInfo.model_validate(payload.info)
        return StoredDocument(
            info=info,
            chunks=list(payload.chunks),
            markdown=payload.markdown,
            text_preview=payload.text_preview,
            doctags=payload.doctags,
            message=payload.message,
        )

    def _enforce_document_cache_limit_locked(self) -> None:
        while len(self._documents) > self._document_cache_max:
            self._documents.popitem(last=False)
