"""Document-related API routes for the demo web interface."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from src.core.config import get_settings

from ..models import (
    ChunkSummary,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentUploadRequest,
    UploadResponse,
)
from ..state import get_app_state

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    settings: str = Form(..., description="JSON encoded DocumentUploadRequest settings"),
) -> UploadResponse:
    """Upload and process a document."""
    try:
        request_model = DocumentUploadRequest.model_validate_json(settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid settings payload: {exc}") from exc

    settings_obj = get_settings()
    max_file_size_mb = max(1, int(settings_obj.document_processing.max_file_size_mb))
    max_file_size_bytes = max_file_size_mb * 1024 * 1024

    reported_size = getattr(file, "size", None)
    if reported_size is not None and int(reported_size) > max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded file exceeds configured limit of {max_file_size_mb}MB",
        )

    file_bytes = await file.read(max_file_size_bytes + 1)
    if len(file_bytes) > max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Uploaded file exceeds configured limit of {max_file_size_mb}MB",
        )
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    suffix = Path(file.filename or "upload").suffix
    tmp_path: Path
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    state = get_app_state()

    try:
        response = await run_in_threadpool(
            state.processing_service.process_document,
            tmp_path,
            file.filename or tmp_path.name,
            request_model,
            len(file_bytes),
        )
        await run_in_threadpool(state.metrics_store.increment_uploads)
        return response
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get("", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """List processed documents."""
    state = get_app_state()
    documents = await run_in_threadpool(state.processing_service.list_documents)

    return DocumentListResponse(
        success=True,
        documents=documents,
        total=len(documents),
        page=1,
        page_size=max(len(documents), 1),
    )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def document_details(document_id: str) -> DocumentDetailResponse:
    """Retrieve full details for a processed document."""
    state = get_app_state()
    stored = await run_in_threadpool(state.processing_service.get_document, document_id)

    if stored is None:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks: List[ChunkSummary] = [ChunkSummary(**chunk) for chunk in stored.chunks]

    return DocumentDetailResponse(
        success=True,
        document=stored.info,
        chunks=chunks,
        markdown=stored.markdown,
        text_preview=stored.text_preview,
        doctags=stored.doctags,
        message=stored.message,
    )


