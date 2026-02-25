"""FastAPI application exposing the Contextprime processing demo and static user interface."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from .models import (
    HealthResponse,
    ServiceInfo,
    ServiceStatus,
    SystemStatus,
)
from .middleware import AccessControlAndRateLimitMiddleware
from .routers import agentic, documents, search, feedback, admin
from .state import get_app_state
from contextprime.core.config import get_settings

settings = get_settings()
cors_origins = [
    origin.strip()
    for origin in (settings.api.cors_origins or [])
    if origin and origin.strip()
]
allow_credentials = bool(cors_origins) and "*" not in cors_origins

app = FastAPI(title="Contextprime", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
app.add_middleware(AccessControlAndRateLimitMiddleware)

app.include_router(documents.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(agentic.router, prefix="/api")
app.include_router(feedback.router, prefix="/api")
app.include_router(admin.router, prefix="/api")

static_dir = Path(__file__).parent / "static"


class NoCacheStaticFiles(StaticFiles):
    """Static files handler that disables HTTP caching."""

    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        if isinstance(response, FileResponse):
            headers = response.headers
            headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            headers["Pragma"] = "no-cache"
            headers["Expires"] = "0"
        return response


app.mount("/static", NoCacheStaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
async def startup_checks() -> None:
    """Validate security settings and optionally wait for dependency readiness."""
    strict_environment = settings.environment.lower() in {"docker", "production", "staging"}
    skip_strict_startup = os.getenv("DOCTAGS_SKIP_STRICT_STARTUP", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    running_pytest = "PYTEST_CURRENT_TEST" in os.environ
    enforce_strict = strict_environment and not skip_strict_startup and not running_pytest

    issues = settings.validate_runtime_security(strict=False)
    for issue in issues:
        logger.warning("Security check: {}", issue)

    if enforce_strict and issues:
        raise RuntimeError("Runtime security validation failed: " + "; ".join(issues))

    # OCR startup probe — surface engine availability before the first document is processed.
    try:
        from contextprime.processing.ocr_engine import OCREngineFactory
        engine = OCREngineFactory.create_engine(
            engine_type=settings.document_processing.ocr_engine
        )
        if engine is None:
            logger.info("OCR engine: disabled by configuration")
        else:
            logger.info("OCR engine: {} ready", type(engine).__name__)
    except Exception as _ocr_exc:
        logger.warning("OCR engine could not be initialized at startup: {}", _ocr_exc)

    if not (settings.startup_readiness.enabled and enforce_strict):
        return

    state = get_app_state()
    required_services = [
        service.strip().lower()
        for service in settings.startup_readiness.required_services
        if service and service.strip()
    ]
    if not required_services:
        return

    timeout_seconds = max(1, int(settings.startup_readiness.timeout_seconds))
    interval_seconds = max(1, int(settings.startup_readiness.check_interval_seconds))
    deadline = monotonic() + timeout_seconds

    while True:
        dependency_health = await run_in_threadpool(state.retrieval_service.get_dependency_health)
        state.dependency_health = dict(dependency_health)
        missing = [name for name in required_services if not dependency_health.get(name, False)]
        if not missing:
            logger.info("Startup readiness checks passed for services: {}", ", ".join(required_services))
            return

        if monotonic() >= deadline:
            raise RuntimeError(
                "Startup readiness checks failed; unavailable services: " + ", ".join(missing)
            )
        await asyncio.sleep(interval_seconds)


@app.get("/", include_in_schema=False)
async def index(_: Request) -> FileResponse:
    """Serve the front-end application."""
    index_path = static_dir / "index.html"
    return FileResponse(index_path)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> JSONResponse:
    """Dependency-aware health probe for monitoring."""
    state = get_app_state()
    dependency_health = await run_in_threadpool(state.retrieval_service.get_dependency_health)
    state.dependency_health = dict(dependency_health)

    required_services = [
        service.strip().lower()
        for service in settings.startup_readiness.required_services
        if service and service.strip()
    ]
    if not required_services:
        required_services = sorted(dependency_health.keys())

    missing = [name for name in required_services if not dependency_health.get(name, False)]
    ready = not missing
    payload = HealthResponse(
        status="ok" if ready else "degraded",
        timestamp=datetime.utcnow(),
        version=app.version,
        ready=ready,
        services=dependency_health,
        missing_services=missing,
    )
    return JSONResponse(
        status_code=200 if ready else 503,
        content=payload.model_dump(mode="json"),
    )


@app.get("/api/readiness")
async def readiness() -> JSONResponse:
    """Readiness probe for upstream orchestrators."""
    state = get_app_state()
    dependency_health = await run_in_threadpool(state.retrieval_service.get_dependency_health)
    state.dependency_health = dict(dependency_health)

    required_services = [
        service.strip().lower()
        for service in settings.startup_readiness.required_services
        if service and service.strip()
    ]
    if not required_services:
        required_services = sorted(dependency_health.keys())

    missing = [name for name in required_services if not dependency_health.get(name, False)]
    ready = not missing
    payload = {
        "status": "ready" if ready else "not_ready",
        "timestamp": datetime.utcnow().isoformat(),
        "required_services": required_services,
        "services": dependency_health,
        "missing_services": missing,
    }
    return JSONResponse(status_code=200 if ready else 503, content=payload)


@app.get("/api/status", response_model=SystemStatus)
async def status() -> SystemStatus:
    """Return basic runtime statistics for the demo."""
    state = get_app_state()
    now = datetime.utcnow()
    uptime = (now - state.started_at).total_seconds()

    documents = state.processing_service.list_documents()
    total_documents = await run_in_threadpool(state.processing_service.total_documents)
    semantic_support = state.processing_service.semantic_support_status()
    feedback_stats = state.feedback_capture_store.get_statistics()
    counters = await run_in_threadpool(state.metrics_store.get_snapshot)
    total_uploads = int(counters.get("total_uploads", 0))
    total_queries = int(counters.get("total_queries", 0))
    dependency_health = await run_in_threadpool(state.retrieval_service.get_dependency_health)
    state.dependency_health = dict(dependency_health)

    qdrant_ready = bool(dependency_health.get("qdrant"))
    neo4j_ready = bool(dependency_health.get("neo4j"))
    if qdrant_ready and neo4j_ready:
        retrieval_status = ServiceStatus.HEALTHY
        retrieval_message = "Vector and graph retrieval services are healthy"
    elif qdrant_ready or neo4j_ready:
        retrieval_status = ServiceStatus.DEGRADED
        retrieval_message = "One retrieval dependency is unavailable"
    else:
        retrieval_status = ServiceStatus.UNHEALTHY
        retrieval_message = "Retrieval dependencies are unavailable"

    services: Dict[str, ServiceInfo] = {
        "processing": ServiceInfo(
            name="processing",
            status=ServiceStatus.HEALTHY,
            message="Processing pipeline ready",
            response_time_ms=None,
            details={
                "documents_processed": total_documents,
                "documents_listed": len(documents),
                "documents_total": total_documents,
                "total_uploads": total_uploads,
                "semantic_chunking": semantic_support,
                "feedback_capture": feedback_stats,
            },
        ),
        "retrieval": ServiceInfo(
            name="retrieval",
            status=retrieval_status,
            message=retrieval_message,
            response_time_ms=None,
            details={
                "dependencies": dependency_health,
                "query_count": total_queries,
            },
        ),
    }

    overall_status = ServiceStatus.HEALTHY
    if any(service.status == ServiceStatus.UNHEALTHY for service in services.values()):
        overall_status = ServiceStatus.UNHEALTHY
    elif any(service.status == ServiceStatus.DEGRADED for service in services.values()):
        overall_status = ServiceStatus.DEGRADED

    return SystemStatus(
        status=overall_status,
        timestamp=now,
        uptime_seconds=uptime,
        services=services,
    )
