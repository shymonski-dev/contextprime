"""Retrieval module for DocTags RAG system."""

from .qdrant_manager import (
    QdrantManager,
    VectorPoint,
    SearchResult as QdrantSearchResult,
)
from .hybrid_retriever import (
    HybridRetriever,
    QueryType,
    SearchStrategy,
    HybridSearchResult,
    SearchMetrics,
)
from .confidence_scorer import (
    ConfidenceScorer,
    ConfidenceLevel,
    CorrectiveAction,
    ConfidenceScore,
)
from .query_router import (
    QueryRouter,
    QueryAnalysis,
    RetrievalStrategy,
    QueryComplexity,
)
from .query_expansion import (
    QueryExpander,
    ExpandedQuery,
)
from .reranker import (
    Reranker,
    RerankedResult,
)
from .cache_manager import (
    CacheManager,
    LRUCache,
    SemanticQueryCache,
)
from .iterative_refiner import (
    IterativeRefiner,
    RefinedResult,
    RefinementStep,
)
from .advanced_pipeline import (
    AdvancedRetrievalPipeline,
    PipelineConfig,
    PipelineResult,
)

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
    "AdvancedRetrievalPipeline",
    "PipelineConfig",
    "PipelineResult",
]
