"""Retrieval module for Contextprime system."""

from __future__ import annotations

from typing import Dict

_OPTIONAL_IMPORT_ERRORS: Dict[str, str] = {}


def _record_optional_import_error(module_name: str, err: Exception) -> None:
    _OPTIONAL_IMPORT_ERRORS[module_name] = f"{type(err).__name__}: {err}"


try:
    from .qdrant_manager import (
        QdrantManager,
        VectorPoint,
        SearchResult as QdrantSearchResult,
    )
except Exception as err:  # pragma: no cover - environment dependent
    QdrantManager = None  # type: ignore[assignment]
    VectorPoint = None  # type: ignore[assignment]
    QdrantSearchResult = None  # type: ignore[assignment]
    _record_optional_import_error("qdrant_manager", err)

try:
    from .hybrid_retriever import (
        HybridRetriever,
        QueryType,
        SearchStrategy,
        HybridSearchResult,
        SearchMetrics,
    )
except Exception as err:  # pragma: no cover - environment dependent
    HybridRetriever = None  # type: ignore[assignment]
    QueryType = None  # type: ignore[assignment]
    SearchStrategy = None  # type: ignore[assignment]
    HybridSearchResult = None  # type: ignore[assignment]
    SearchMetrics = None  # type: ignore[assignment]
    _record_optional_import_error("hybrid_retriever", err)

try:
    from .confidence_scorer import (
        ConfidenceScorer,
        ConfidenceLevel,
        CorrectiveAction,
        ConfidenceScore,
    )
except Exception as err:  # pragma: no cover - environment dependent
    ConfidenceScorer = None  # type: ignore[assignment]
    ConfidenceLevel = None  # type: ignore[assignment]
    CorrectiveAction = None  # type: ignore[assignment]
    ConfidenceScore = None  # type: ignore[assignment]
    _record_optional_import_error("confidence_scorer", err)

try:
    from .query_router import (
        QueryRouter,
        QueryAnalysis,
        RetrievalStrategy,
        QueryComplexity,
    )
except Exception as err:  # pragma: no cover - environment dependent
    QueryRouter = None  # type: ignore[assignment]
    QueryAnalysis = None  # type: ignore[assignment]
    RetrievalStrategy = None  # type: ignore[assignment]
    QueryComplexity = None  # type: ignore[assignment]
    _record_optional_import_error("query_router", err)

try:
    from .query_expansion import (
        QueryExpander,
        ExpandedQuery,
    )
except Exception as err:  # pragma: no cover - environment dependent
    QueryExpander = None  # type: ignore[assignment]
    ExpandedQuery = None  # type: ignore[assignment]
    _record_optional_import_error("query_expansion", err)

try:
    from .reranker import (
        Reranker,
        RerankedResult,
    )
except Exception as err:  # pragma: no cover - environment dependent
    Reranker = None  # type: ignore[assignment]
    RerankedResult = None  # type: ignore[assignment]
    _record_optional_import_error("reranker", err)

try:
    from .cache_manager import (
        CacheManager,
        LRUCache,
        SemanticQueryCache,
    )
except Exception as err:  # pragma: no cover - environment dependent
    CacheManager = None  # type: ignore[assignment]
    LRUCache = None  # type: ignore[assignment]
    SemanticQueryCache = None  # type: ignore[assignment]
    _record_optional_import_error("cache_manager", err)

try:
    from .iterative_refiner import (
        IterativeRefiner,
        RefinedResult,
        RefinementStep,
    )
except Exception as err:  # pragma: no cover - environment dependent
    IterativeRefiner = None  # type: ignore[assignment]
    RefinedResult = None  # type: ignore[assignment]
    RefinementStep = None  # type: ignore[assignment]
    _record_optional_import_error("iterative_refiner", err)
from .context_selector import (
    TrainableContextSelector,
    SelectorExample,
    SelectorMetrics,
)
from .policy_benchmark import (
    BenchmarkSample,
    PolicySampleMetrics,
    PolicyAggregateMetrics,
    load_benchmark_samples,
    evaluate_policy_sample,
    aggregate_policy_metrics,
    metrics_to_dict,
    sample_metrics_to_dict,
)
from .benchmark_trends import (
    BenchmarkTrendRecord,
    load_benchmark_report,
    extract_trend_records,
    append_trend_records,
    load_trend_history,
    write_trend_markdown,
)
from .feedback_capture_store import FeedbackCaptureStore
from .feedback_dataset import (
    FeedbackDatasetSummary,
    load_jsonl_records,
    build_selector_examples_from_events,
    save_selector_examples,
)

try:
    from .advanced_pipeline import (
        AdvancedRetrievalPipeline,
        PipelineConfig,
        PipelineResult,
    )
except Exception as err:  # pragma: no cover - environment dependent
    AdvancedRetrievalPipeline = None  # type: ignore[assignment]
    PipelineConfig = None  # type: ignore[assignment]
    PipelineResult = None  # type: ignore[assignment]
    _record_optional_import_error("advanced_pipeline", err)


def get_optional_import_errors() -> Dict[str, str]:
    """Expose optional import failures for diagnostics."""
    return dict(_OPTIONAL_IMPORT_ERRORS)


__all__ = [
    # Core retrieval
    "QdrantManager",
    "VectorPoint",
    "QdrantSearchResult",
    "HybridRetriever",
    "QueryType",
    "SearchStrategy",
    "HybridSearchResult",
    "SearchMetrics",
    # Advanced features
    "ConfidenceScorer",
    "ConfidenceLevel",
    "CorrectiveAction",
    "ConfidenceScore",
    "QueryRouter",
    "QueryAnalysis",
    "RetrievalStrategy",
    "QueryComplexity",
    "QueryExpander",
    "ExpandedQuery",
    "Reranker",
    "RerankedResult",
    "CacheManager",
    "LRUCache",
    "SemanticQueryCache",
    "IterativeRefiner",
    "RefinedResult",
    "RefinementStep",
    "TrainableContextSelector",
    "SelectorExample",
    "SelectorMetrics",
    "BenchmarkSample",
    "PolicySampleMetrics",
    "PolicyAggregateMetrics",
    "load_benchmark_samples",
    "evaluate_policy_sample",
    "aggregate_policy_metrics",
    "metrics_to_dict",
    "sample_metrics_to_dict",
    "BenchmarkTrendRecord",
    "load_benchmark_report",
    "extract_trend_records",
    "append_trend_records",
    "load_trend_history",
    "write_trend_markdown",
    "FeedbackCaptureStore",
    "FeedbackDatasetSummary",
    "load_jsonl_records",
    "build_selector_examples_from_events",
    "save_selector_examples",
    "AdvancedRetrievalPipeline",
    "PipelineConfig",
    "PipelineResult",
    "get_optional_import_errors",
]
