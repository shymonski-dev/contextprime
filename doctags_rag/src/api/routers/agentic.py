"""Agentic RAG API routes."""

from __future__ import annotations

from typing import List, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from loguru import logger

from ..models import (
    AgenticQueryRequest,
    AgenticQueryResponse,
    SearchResultItem,
)
from ..state import get_app_state

router = APIRouter(prefix="/agentic", tags=["agentic"])


@router.post("", response_model=AgenticQueryResponse)
async def run_agentic_query(request: AgenticQueryRequest) -> AgenticQueryResponse:
    """Execute an agentic reasoning cycle for the supplied query."""
    state = get_app_state()

    try:
        result = await state.retrieval_service.agentic_query(request)
    except Exception as exc:  # pragma: no cover - defensive
        error_id = uuid4().hex[:12]
        logger.exception("Agentic query failed (error_id=%s)", error_id)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error (reference: {error_id})",
        ) from exc

    sources = _serialise_sources(result.results)

    reasoning = None
    if request.return_reasoning:
        reasoning = _serialise_reasoning(result.plan.steps, result.execution_results)

    metadata: Dict[str, Any] = {
        "total_time_ms": round(result.total_time_ms, 2),
        "planning_time_ms": round(result.planning_time_ms, 2),
        "execution_time_ms": round(result.execution_time_ms, 2),
        "evaluation_time_ms": round(result.evaluation_time_ms, 2),
        "learning_time_ms": round(result.learning_time_ms, 2),
        "mode": result.mode.value,
        "improved": result.improved,
        "iteration": result.iteration,
        "assessment": {
            "overall_score": result.assessment.overall_score,
            "faithfulness": result.assessment.faithfulness,
            "groundedness": result.assessment.groundedness,
            "coherence": result.assessment.coherence,
            "completeness": result.assessment.completeness,
        },
    }

    return AgenticQueryResponse(
        success=True,
        query=request.query,
        answer=result.answer,
        confidence=result.assessment.overall_score,
        iterations=result.iteration,
        reasoning_steps=reasoning,
        sources=sources,
        processing_time=result.total_time_ms / 1000,
        metadata=metadata,
    )


def _serialise_sources(raw_sources: List[Dict[str, Any]]) -> List[SearchResultItem]:
    serialised: List[SearchResultItem] = []
    for idx, source in enumerate(raw_sources, start=1):
        serialised.append(
            SearchResultItem(
                id=str(source.get("id") or source.get("doc_id") or f"source-{idx}"),
                content=source.get("content") or source.get("text") or "",
                score=float(source.get("score", 0.0)),
                confidence=float(source.get("confidence", 0.0)),
                source=source.get("source") or source.get("origin") or "hybrid",
                metadata=source.get("metadata") or {},
                graph_context=source.get("graph_context"),
            )
        )
    return serialised


def _serialise_reasoning(steps, execution_results) -> List[Dict[str, Any]]:
    reasoning: List[Dict[str, Any]] = []
    exec_lookup = {res.step_id: res for res in execution_results}
    for step in steps:
        execution = exec_lookup.get(step.step_id)
        reasoning.append({
            "step_id": step.step_id,
            "description": step.description,
            "type": getattr(step.step_type, "value", str(step.step_type)),
            "success": execution.success if execution else None,
            "results": execution.results if execution else [],
            "execution_time_ms": execution.execution_time_ms if execution else None,
            "metadata": execution.metadata if execution else {},
        })
    return reasoning
