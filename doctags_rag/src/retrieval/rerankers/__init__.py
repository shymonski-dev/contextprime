"""Reranker utilities for the retrieval pipeline."""

# MonoT5 requires transformers; guard so environments without it still import cleanly.
try:
    from .mono_t5 import MonoT5Reranker
except Exception:  # pragma: no cover
    from loguru import logger as _logger
    _logger.warning("rerankers: MonoT5Reranker unavailable (transformers not installed)")
    MonoT5Reranker = None  # type: ignore[assignment,misc]

__all__ = ["MonoT5Reranker"]
