"""Feedback capture API routes."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse
from loguru import logger

from ..models import (
    RetrievalFeedbackRequest,
    RetrievalFeedbackResponse,
)
from ..state import get_app_state

router = APIRouter(prefix="/feedback", tags=["feedback"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TREND_HISTORY_PATH = PROJECT_ROOT / "reports" / "retrieval_policy_trend_history.jsonl"
TREND_MARKDOWN_PATH = PROJECT_ROOT / "reports" / "retrieval_policy_trends.md"


@router.post("/retrieval", response_model=RetrievalFeedbackResponse)
async def submit_retrieval_feedback(request: RetrievalFeedbackRequest) -> RetrievalFeedbackResponse:
    """Store user feedback for a retrieval query event."""
    state = get_app_state()

    try:
        feedback_id = await run_in_threadpool(
            state.feedback_capture_store.record_feedback_event,
            query_id=request.query_id,
            helpful=request.helpful,
            selected_result_ids=request.selected_result_ids,
            result_labels=[item.model_dump() for item in request.result_labels],
            comment=request.comment,
            user_id=request.user_id,
            metadata=request.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        error_id = uuid4().hex[:12]
        logger.exception("Feedback submission failed (error_id=%s)", error_id)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error (reference: {error_id})",
        ) from exc

    return RetrievalFeedbackResponse(
        success=True,
        query_id=request.query_id,
        feedback_id=feedback_id,
        message="Feedback stored",
    )


@router.get("/trends")
async def retrieval_trend_summary() -> dict:
    """Return retrieval trend summary availability and metadata."""
    history_records = 0
    if TREND_HISTORY_PATH.exists():
        history_records = sum(
            1 for line in TREND_HISTORY_PATH.read_text(encoding="utf-8").splitlines() if line.strip()
        )

    available = TREND_MARKDOWN_PATH.exists()
    last_updated = None
    if available:
        modified = datetime.fromtimestamp(TREND_MARKDOWN_PATH.stat().st_mtime, tz=timezone.utc)
        last_updated = modified.isoformat()

    return {
        "available": available,
        "history_records": history_records,
        "last_updated": last_updated,
        "markdown_endpoint": "/api/feedback/trends/markdown",
    }


@router.get("/trends/markdown")
async def retrieval_trend_markdown() -> PlainTextResponse:
    """Serve the published retrieval trend markdown report."""
    if not TREND_MARKDOWN_PATH.exists():
        raise HTTPException(status_code=404, detail="Trend summary is not available")

    markdown = TREND_MARKDOWN_PATH.read_text(encoding="utf-8")
    return PlainTextResponse(markdown, media_type="text/markdown")
