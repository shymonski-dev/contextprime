"""Search-related API routes."""

from __future__ import annotations

from typing import Any, Dict, List
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from ..models import (
    AdvancedQueryRequest,
    QueryResponse,
    SearchResultItem,
)
from ..state import get_app_state

router = APIRouter(prefix="/search", tags=["search"])


@router.post("/hybrid", response_model=QueryResponse)
async def hybrid_search(request: AdvancedQueryRequest) -> QueryResponse:
    """Run a hybrid (vector + graph) search."""
    state = get_app_state()

    try:
        results, metrics, rerank_applied = await run_in_threadpool(
            state.retrieval_service.hybrid_search,
            request,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        error_id = uuid4().hex[:12]
        logger.exception("Hybrid search failed (error_id=%s)", error_id)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error (reference: {error_id})",
        ) from exc

    items = [
        SearchResultItem(
            id=result.id,
            content=result.content,
            score=result.score,
            confidence=result.confidence,
            source=result.source,
            metadata=result.metadata,
            graph_context=result.graph_context,
        )
        for result in results
    ]
    await run_in_threadpool(state.metrics_store.increment_queries)

    metadata = {
        "query_type": metrics.query_type.value,
        "strategy": metrics.strategy.value,
        "vector_results": metrics.vector_results,
        "graph_results": metrics.graph_results,
        "lexical_results": metrics.lexical_results,
        "combined_results": metrics.combined_results,
        "vector_time_ms": round(metrics.vector_time_ms, 2),
        "graph_time_ms": round(metrics.graph_time_ms, 2),
        "lexical_time_ms": round(metrics.lexical_time_ms, 2),
        "fusion_time_ms": round(metrics.fusion_time_ms, 2),
        "total_time_ms": round(metrics.total_time_ms, 2),
        "cache_hit": metrics.cache_hit,
        "rerank_applied": rerank_applied,
        "rerank_time_ms": round(metrics.rerank_time_ms, 2),
        "services": metrics.services,
    }

    query_event_payload: List[Dict[str, Any]] = [
        {
            "id": item.id,
            "content": item.content,
            "score": item.score,
            "confidence": item.confidence,
            "source": item.source,
            "metadata": item.metadata,
        }
        for item in items
    ]
    try:
        query_id = await run_in_threadpool(
            state.feedback_capture_store.record_query_event,
            query=request.query,
            request_payload=request.model_dump(mode="json"),
            results=query_event_payload,
            metadata=metadata,
        )
        metadata["query_id"] = query_id
        metadata["feedback_endpoint"] = "/api/feedback/retrieval"
    except Exception:
        metadata["feedback_capture"] = "failed"

    return QueryResponse(
        success=True,
        query=request.query,
        results=items,
        total_results=len(items),
        processing_time=metrics.total_time_ms / 1000 if metrics.total_time_ms else 0.0,
        metadata=metadata,
    )
