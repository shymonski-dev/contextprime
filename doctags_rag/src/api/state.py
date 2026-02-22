"""Application state and dependency helpers for the demo API."""

from __future__ import annotations

from datetime import datetime
from threading import Lock
from typing import Dict, Optional
import os

from .services.processing_service import ProcessingService
from .services.retrieval_service import RetrievalService
from .services.metrics_store import OperationalMetricsStore
from src.retrieval.feedback_capture_store import FeedbackCaptureStore


class AppState:
    """Container for long-lived objects shared across the API."""

    def __init__(self) -> None:
        self.processing_service = ProcessingService()
        self.retrieval_service = RetrievalService()
        self.feedback_capture_store = FeedbackCaptureStore()
        metrics_path = os.getenv("DOCTAGS_METRICS_STORE_PATH", "data/storage/metrics.db")
        self.metrics_store = OperationalMetricsStore(db_path=metrics_path)
        self.started_at = datetime.utcnow()
        self.dependency_health: Dict[str, bool] = {
            "qdrant": False,
            "neo4j": False,
        }


_app_state: Optional[AppState] = None
_state_lock = Lock()


def get_app_state() -> AppState:
    """Return the singleton application state, creating it if required (thread-safe)."""
    global _app_state
    if _app_state is None:
        with _state_lock:
            if _app_state is None:
                _app_state = AppState()
    return _app_state
