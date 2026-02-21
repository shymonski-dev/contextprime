"""
Hybrid Retrieval Manager for Contextprime.

Combines Neo4j graph database and Qdrant vector database for hybrid search:
- Fusion scoring to combine results from both sources
- Configurable weights for vector vs graph results
- Query routing based on query type
- Result ranking and deduplication
- Confidence scoring
"""

from typing import Dict, List, Any, Optional, Tuple, Set, Hashable
from dataclasses import dataclass, field, replace
from enum import Enum
import re
import time
import copy
from collections import defaultdict, OrderedDict
from pathlib import Path

from loguru import logger

from ..knowledge_graph.neo4j_manager import Neo4jManager, SearchResult as Neo4jResult
from ..retrieval.rerankers import MonoT5Reranker
from .qdrant_manager import QdrantManager, SearchResult as QdrantResult
from ..core.config import get_settings


class QueryType(Enum):
    """Types of queries for routing."""
    FACTUAL = "factual"  # What, when, where, who
    RELATIONSHIP = "relationship"  # How related, connections
    COMPLEX = "complex"  # Multi-hop reasoning
    HYBRID = "hybrid"  # Combination


class SearchStrategy(Enum):
    """Search strategies."""
    VECTOR_ONLY = "vector"
    GRAPH_ONLY = "graph"
    HYBRID = "hybrid"


class GraphRetrievalPolicy(Enum):
    """Graph retrieval policy modes."""
    STANDARD = "standard"
    LOCAL = "local"
    GLOBAL = "global"
    DRIFT = "drift"
    COMMUNITY = "community"
    ADAPTIVE = "adaptive"


@dataclass
class HybridSearchResult:
    """Combined search result from both databases."""
    id: str
    content: str
    score: float
    confidence: float
    source: str  # "vector", "graph", or "hybrid"
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    lexical_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    graph_context: Optional[Dict[str, Any]] = None


@dataclass
class SearchMetrics:
    """Metrics for search performance."""
    query_type: QueryType
    strategy: SearchStrategy
    vector_results: int
    graph_results: int
    lexical_results: int
    combined_results: int
    vector_time_ms: float
    graph_time_ms: float
    lexical_time_ms: float
    fusion_time_ms: float
    total_time_ms: float
    cache_hit: bool = False
    rerank_time_ms: float = 0.0
    rerank_applied: bool = False
    services: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _CacheEntry:
    results: List[HybridSearchResult]
    timestamp: float
    strategy: SearchStrategy


class HybridRetriever:
    """
    Hybrid retrieval manager combining Neo4j and Qdrant.

    Features:
    - Dual database search
    - Reciprocal rank fusion for result combination
    - Query type detection and routing
    - Configurable search strategies
    - Confidence scoring
    - Result deduplication
    """

    def __init__(
        self,
        neo4j_manager: Optional[Neo4jManager] = None,
        qdrant_manager: Optional[QdrantManager] = None,
        vector_weight: float = 0.7,
        graph_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever.

        Args:
            neo4j_manager: Neo4j manager instance
            qdrant_manager: Qdrant manager instance
            vector_weight: Weight for vector search results
            graph_weight: Weight for graph search results
        """
        _settings = get_settings()

        self.neo4j = neo4j_manager
        self.qdrant = qdrant_manager
        self._owns_neo4j = neo4j_manager is None
        self._owns_qdrant = qdrant_manager is None
        self._neo4j_init_failed = False
        self._qdrant_init_failed = False

        if vector_weight < 0 or graph_weight < 0:
            raise ValueError("vector_weight and graph_weight must be non-negative")

        # Normalise weights while keeping sensible defaults when misconfigured
        total_weight = vector_weight + graph_weight
        if total_weight == 0:
            logger.warning(
                "Hybrid retriever received zero total weight; defaulting to vector=0.7, graph=0.3"
            )
            vector_weight, graph_weight = 0.7, 0.3
            total_weight = vector_weight + graph_weight

        self.vector_weight = vector_weight / total_weight
        self.graph_weight = graph_weight / total_weight

        manager_collection_name = getattr(
            getattr(qdrant_manager, "config", None),
            "collection_name",
            None,
        )
        self.default_collection_name = manager_collection_name or getattr(
            _settings.qdrant,
            "collection_name",
            None,
        )
        hybrid_cfg = getattr(_settings.retrieval, "hybrid_search", {}) or {}
        self.default_graph_vector_index = hybrid_cfg.get("graph_vector_index")

        graph_policy_cfg = hybrid_cfg.get("graph_policy", {}) if isinstance(hybrid_cfg, dict) else {}
        graph_policy_mode = str(graph_policy_cfg.get("mode", GraphRetrievalPolicy.STANDARD.value)).lower()
        if graph_policy_mode not in {policy.value for policy in GraphRetrievalPolicy}:
            graph_policy_mode = GraphRetrievalPolicy.STANDARD.value
        self.graph_policy_mode = graph_policy_mode
        self.graph_local_seed_k = max(2, int(graph_policy_cfg.get("local_seed_k", 8)))
        self.graph_local_max_depth = max(1, int(graph_policy_cfg.get("local_max_depth", 2)))
        self.graph_local_neighbor_limit = max(10, int(graph_policy_cfg.get("local_neighbor_limit", 80)))
        self.graph_global_scan_nodes = max(100, int(graph_policy_cfg.get("global_scan_nodes", 1500)))
        self.graph_global_max_terms = max(2, int(graph_policy_cfg.get("global_max_terms", 8)))
        self.graph_drift_local_weight = max(0.0, float(graph_policy_cfg.get("drift_local_weight", 0.65)))
        self.graph_drift_global_weight = max(0.0, float(graph_policy_cfg.get("drift_global_weight", 0.35)))
        self.graph_community_scan_nodes = max(100, int(graph_policy_cfg.get("community_scan_nodes", 500)))
        self.graph_community_max_terms = max(2, int(graph_policy_cfg.get("community_max_terms", 8)))
        self.graph_community_top_communities = max(1, int(graph_policy_cfg.get("community_top_communities", 5)))
        self.graph_community_members_per_community = max(
            1, int(graph_policy_cfg.get("community_members_per_community", 6))
        )
        self.graph_community_vector_weight = max(0.0, float(graph_policy_cfg.get("community_vector_weight", 0.45)))
        self.graph_community_summary_weight = max(0.0, float(graph_policy_cfg.get("community_summary_weight", 0.35)))
        self.graph_community_member_weight = max(0.0, float(graph_policy_cfg.get("community_member_weight", 0.20)))
        community_version = graph_policy_cfg.get("community_version")
        self.graph_community_version = str(community_version).strip() if community_version else None
        lexical_cfg = hybrid_cfg.get("lexical", {}) if isinstance(hybrid_cfg, dict) else {}
        self.lexical_enabled = bool(lexical_cfg.get("enable", False))
        self.lexical_weight = max(0.0, float(lexical_cfg.get("weight", 0.2)))
        self.lexical_max_scan_points = max(100, int(lexical_cfg.get("max_scan_points", 1500)))
        self.lexical_scan_ratio = min(1.0, max(0.0, float(lexical_cfg.get("scan_ratio", 0.02))))
        self.lexical_max_scan_cap = max(
            self.lexical_max_scan_points,
            int(lexical_cfg.get("max_scan_cap", 20000)),
        )
        self.lexical_page_size = max(20, int(lexical_cfg.get("page_size", 200)))
        self.lexical_bm25_k1 = float(lexical_cfg.get("bm25_k1", 1.2))
        self.lexical_bm25_b = float(lexical_cfg.get("bm25_b", 0.75))

        cache_cfg = hybrid_cfg.get("cache", {}) if isinstance(hybrid_cfg, dict) else {}
        self.cache_enabled = cache_cfg.get("enable", True)
        self.cache_max_size = int(cache_cfg.get("max_size", 128))
        self.cache_ttl = float(cache_cfg.get("ttl_seconds", 600))
        self._cache: "OrderedDict[Hashable, _CacheEntry]" = OrderedDict()

        self.models_dir: Optional[Path] = None
        paths_cfg = getattr(_settings, "paths", None)
        models_dir_value = getattr(paths_cfg, "models_dir", None) if paths_cfg else None
        if models_dir_value:
            try:
                self.models_dir = Path(models_dir_value)
                self.models_dir.mkdir(parents=True, exist_ok=True)
            except Exception as err:  # pragma: no cover - defensive
                logger.warning("Unable to prepare models directory %s (%s)", models_dir_value, err)
                self.models_dir = None

        # Query routing patterns
        self.routing_patterns = {
            QueryType.FACTUAL: [
                r'\b(what|when|where|who|which)\b',
                r'\b(define|definition|meaning|is|are)\b',
            ],
            QueryType.RELATIONSHIP: [
                r'\b(how.*related|relationship|connection|link)\b',
                r'\b(between|among|connects|influences)\b',
                r'\b(causes?|effects?|impacts?)\b',
            ],
            QueryType.COMPLEX: [
                r'\b(explain|analyze|compare|why)\b',
                r'\b(multiple|several|various)\b',
            ],
        }

        confidence_cfg = getattr(_settings.retrieval, "confidence_scoring", {}) or {}
        self.min_confidence_threshold = confidence_cfg.get("min_confidence", 0.1)

        rerank_cfg = getattr(_settings.retrieval, "rerank_settings", {}) or {}
        self.reranker_top_n = int(rerank_cfg.get("top_n", 50))
        self.reranker: Optional[MonoT5Reranker] = None
        if rerank_cfg.get("enable", False):
            model_name = rerank_cfg.get("model_name", "castorini/monot5-base-msmarco-10k")
            device = rerank_cfg.get("device")
            try:
                cache_dir = self.models_dir if self.models_dir is not None else None
                self.reranker = MonoT5Reranker(
                    model_name=model_name,
                    device=device,
                    cache_dir=cache_dir,
                )
                logger.info("MonoT5 reranker initialised: %s", model_name)
            except Exception as err:  # pragma: no cover - optional dependency failures
                logger.warning(
                    "Failed to initialise reranker (%s); continuing without reranking",
                    err,
                    exc_info=True,
                )

        logger.info(
            f"Hybrid retriever initialized (vector: {self.vector_weight:.2f}, "
            f"graph: {self.graph_weight:.2f}, lexical: {self.lexical_weight:.2f}, "
            f"lexical_enabled={self.lexical_enabled}, graph_policy={self.graph_policy_mode}, "
            f"min_conf={self.min_confidence_threshold:.2f}, "
            f"cache={'on' if self.cache_enabled else 'off'})"
        )

    def _ensure_neo4j(self) -> Optional[Neo4jManager]:
        """Lazily initialize Neo4j manager if needed."""
        if self.neo4j is not None:
            return self.neo4j

        if self._neo4j_init_failed:
            return None

        try:
            self.neo4j = Neo4jManager()
            return self.neo4j
        except Exception as err:
            logger.warning(f"Failed to initialize Neo4j manager: {err}")
            self._neo4j_init_failed = True
            return None

    def _ensure_qdrant(self) -> Optional[QdrantManager]:
        """Lazily initialize Qdrant manager if needed."""
        if self.qdrant is not None:
            return self.qdrant

        if self._qdrant_init_failed:
            return None

        try:
            self.qdrant = QdrantManager()
            return self.qdrant
        except Exception as err:
            logger.warning(f"Failed to initialize Qdrant manager: {err}")
            self._qdrant_init_failed = True
            return None

    def _resolve_graph_policy(
        self,
        graph_policy: Optional[str],
    ) -> GraphRetrievalPolicy:
        """Resolve graph retrieval policy from request or configuration."""
        candidate = (graph_policy or self.graph_policy_mode or "standard").lower()
        mapping = {
            GraphRetrievalPolicy.STANDARD.value: GraphRetrievalPolicy.STANDARD,
            GraphRetrievalPolicy.LOCAL.value: GraphRetrievalPolicy.LOCAL,
            GraphRetrievalPolicy.GLOBAL.value: GraphRetrievalPolicy.GLOBAL,
            GraphRetrievalPolicy.DRIFT.value: GraphRetrievalPolicy.DRIFT,
            GraphRetrievalPolicy.COMMUNITY.value: GraphRetrievalPolicy.COMMUNITY,
            GraphRetrievalPolicy.ADAPTIVE.value: GraphRetrievalPolicy.ADAPTIVE,
        }
        return mapping.get(candidate, GraphRetrievalPolicy.STANDARD)

    def _select_adaptive_graph_policy(
        self,
        query_text: str,
        query_type: QueryType,
    ) -> GraphRetrievalPolicy:
        """Select graph retrieval policy from query intent and breadth."""
        normalized = (query_text or "").lower()
        token_count = len(re.findall(r"\w+", normalized))

        global_markers = {
            "overall",
            "across",
            "trend",
            "compare",
            "summary",
            "landscape",
            "broad",
        }
        community_markers = {
            "community",
            "cluster",
            "group",
            "segment",
        }

        if any(marker in normalized for marker in community_markers):
            return GraphRetrievalPolicy.COMMUNITY

        if any(marker in normalized for marker in global_markers):
            return GraphRetrievalPolicy.GLOBAL

        if token_count >= 16:
            return GraphRetrievalPolicy.DRIFT

        if query_type == QueryType.COMPLEX:
            return GraphRetrievalPolicy.DRIFT

        if query_type == QueryType.RELATIONSHIP:
            return GraphRetrievalPolicy.LOCAL

        if token_count <= 5:
            return GraphRetrievalPolicy.STANDARD

        return GraphRetrievalPolicy.LOCAL

    def _resolve_collection_name(self, requested: Optional[str]) -> Optional[str]:
        """Resolve effective Qdrant collection name."""
        if requested:
            return requested

        manager_collection_name = getattr(
            getattr(self.qdrant, "config", None),
            "collection_name",
            None,
        )
        if manager_collection_name:
            return manager_collection_name

        return self.default_collection_name

    def detect_query_type(self, query: str) -> QueryType:
        """
        Detect query type from query text.

        Args:
            query: Query string

        Returns:
            Detected query type
        """
        query_lower = query.lower()

        # Check each pattern
        scores = defaultdict(int)

        for qtype, patterns in self.routing_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores[qtype] += 1

        if not scores:
            return QueryType.HYBRID

        relationship_score = scores.get(QueryType.RELATIONSHIP, 0)
        factual_score = scores.get(QueryType.FACTUAL, 0)
        if relationship_score > 0 and relationship_score >= factual_score:
            return QueryType.RELATIONSHIP

        # Return type with highest score
        max_type = max(scores.items(), key=lambda x: x[1])

        # When multiple query families match, treat as complex.
        if len(scores) > 1:
            return QueryType.COMPLEX

        return max_type[0]

    def route_query(self, query_type: QueryType) -> SearchStrategy:
        """
        Route query to appropriate search strategy.

        Args:
            query_type: Detected query type

        Returns:
            Search strategy to use
        """
        routing = {
            QueryType.FACTUAL: SearchStrategy.VECTOR_ONLY,
            QueryType.RELATIONSHIP: SearchStrategy.GRAPH_ONLY,
            QueryType.COMPLEX: SearchStrategy.HYBRID,
            QueryType.HYBRID: SearchStrategy.HYBRID,
        }

        return routing.get(query_type, SearchStrategy.HYBRID)

    def search(
        self,
        query_vector: Optional[List[float]],
        query_text: str,
        top_k: int = 10,
        strategy: Optional[SearchStrategy] = None,
        graph_policy: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        min_confidence: Optional[float] = None,
        vector_index_name: Optional[str] = None,
        collection_name: Optional[str] = None
    ) -> Tuple[List[HybridSearchResult], SearchMetrics]:
        """
        Perform hybrid search combining vector and graph databases.

        Args:
            query_vector: Query embedding vector
            query_text: Original query text
            top_k: Number of results to return
            strategy: Search strategy (auto-detected if None)
            graph_policy: Graph retrieval policy
            filters: Optional filters for both databases
            min_confidence: Minimum confidence threshold (defaults to config value)
            vector_index_name: Neo4j vector index name
            collection_name: Qdrant collection name

        Returns:
            Tuple of (results, metrics)
        """
        import time

        if query_vector is not None and not isinstance(query_vector, list):
            try:
                query_vector = list(query_vector)
            except TypeError as err:
                raise ValueError("query_vector must be an iterable of floats") from err

        start_time = time.time()

        # Detect query type and strategy
        query_type = self.detect_query_type(query_text)
        if strategy is None:
            strategy = self.route_query(query_type)
        requested_graph_policy = self._resolve_graph_policy(graph_policy)
        effective_graph_policy = requested_graph_policy
        if requested_graph_policy == GraphRetrievalPolicy.ADAPTIVE:
            effective_graph_policy = self._select_adaptive_graph_policy(
                query_text=query_text,
                query_type=query_type,
            )
        effective_collection_name = self._resolve_collection_name(collection_name)

        logger.info(
            f"Query type: {query_type.value}, Strategy: {strategy.value}, "
            f"Graph policy: {requested_graph_policy.value}->{effective_graph_policy.value}"
        )

        # Initialize metrics
        metrics = SearchMetrics(
            query_type=query_type,
            strategy=strategy,
            vector_results=0,
            graph_results=0,
            lexical_results=0,
            combined_results=0,
            vector_time_ms=0,
            graph_time_ms=0,
            lexical_time_ms=0,
            fusion_time_ms=0,
            total_time_ms=0,
        )

        cache_key: Optional[Hashable] = None
        if self.cache_enabled:
            cache_key = self._build_cache_key(
                query_text=query_text,
                query_vector=query_vector,
                strategy=strategy,
                top_k=top_k,
                filters=filters,
                collection_name=effective_collection_name,
                vector_index_name=vector_index_name,
                graph_policy=effective_graph_policy.value,
            )
            cached_entry = self._cache_get(cache_key)
            if cached_entry:
                metrics.cache_hit = True
                metrics.combined_results = len(cached_entry.results)
                metrics.total_time_ms = (time.time() - start_time) * 1000
                metrics.rerank_applied = bool(self.reranker)
                metrics.services = {
                    "qdrant": bool(self.qdrant and not self._qdrant_init_failed),
                    "neo4j": bool(self.neo4j and not self._neo4j_init_failed),
                    "lexical": self.lexical_enabled,
                    "graph_policy": effective_graph_policy.value,
                    "graph_policy_requested": requested_graph_policy.value,
                }
                logger.info(
                    "Cache hit for query '%s' (strategy=%s)",
                    query_text,
                    strategy.value,
                )
                return self._clone_results(cached_entry.results)[:top_k], metrics

        # Execute searches based on strategy
        vector_results = []
        graph_results = []
        lexical_results = []

        can_use_vector = query_vector is not None and len(query_vector) > 0
        requires_embedding = strategy in {
            SearchStrategy.VECTOR_ONLY,
            SearchStrategy.GRAPH_ONLY,
            SearchStrategy.HYBRID
        }

        if requires_embedding and not can_use_vector:
            raise ValueError(
                f"Retrieval strategy '{strategy.value}' requires a query embedding. "
                "Provide query_vector or configure embedding support."
            )

        if strategy in [SearchStrategy.VECTOR_ONLY, SearchStrategy.HYBRID]:
            if can_use_vector:
                vector_start = time.time()
                vector_results = self._search_vector(
                    query_vector, top_k, filters, effective_collection_name
                )
                metrics.vector_time_ms = (time.time() - vector_start) * 1000
                metrics.vector_results = len(vector_results)
            else:
                logger.warning("Vector search requested but no query embedding provided; skipping vector lookup")

            if self.lexical_enabled and query_text.strip():
                lexical_start = time.time()
                lexical_results = self._search_lexical(
                    query_text=query_text,
                    top_k=top_k,
                    filters=filters,
                    collection_name=effective_collection_name,
                )
                metrics.lexical_time_ms = (time.time() - lexical_start) * 1000
                metrics.lexical_results = len(lexical_results)

        if strategy in [SearchStrategy.GRAPH_ONLY, SearchStrategy.HYBRID]:
            if can_use_vector:
                graph_start = time.time()
                effective_index = vector_index_name or self.default_graph_vector_index
                graph_results = self._search_graph(
                    query_vector,
                    query_text,
                    top_k,
                    filters,
                    effective_index,
                    effective_graph_policy,
                )
                metrics.graph_time_ms = (time.time() - graph_start) * 1000
                metrics.graph_results = len(graph_results)
            else:
                logger.warning("Graph vector search requested but no query embedding provided; skipping graph lookup")

        # Combine results
        fusion_start = time.time()
        if strategy == SearchStrategy.HYBRID:
            combined_results = self._fusion_combine(
                vector_results, graph_results, lexical_results, top_k
            )
        elif strategy == SearchStrategy.VECTOR_ONLY:
            if vector_results and lexical_results:
                combined_results = self._fusion_combine(
                    vector_results=vector_results,
                    graph_results=[],
                    lexical_results=lexical_results,
                    top_k=top_k,
                )
            elif vector_results:
                combined_results = self._convert_vector_results(vector_results)
            else:
                combined_results = self._convert_lexical_results(lexical_results)
        else:  # GRAPH_ONLY
            combined_results = self._convert_graph_results(graph_results)

        metrics.fusion_time_ms = (time.time() - fusion_start) * 1000

        # Apply confidence filtering
        threshold = (
            min_confidence if min_confidence is not None else self.min_confidence_threshold
        )

        filtered_results = [
            r for r in combined_results if r.confidence >= threshold
        ]

        if self.reranker and filtered_results:
            rerank_start = time.time()
            try:
                rerank_top_n = min(len(filtered_results), self.reranker_top_n)
                filtered_results = self.reranker.rerank(
                    query_text,
                    filtered_results,
                    top_k=rerank_top_n,
                )
                metrics.rerank_applied = True
                metrics.rerank_time_ms = (time.time() - rerank_start) * 1000
            except Exception as err:  # pragma: no cover - reranker optional
                logger.warning("Reranker failed: %s", err)
                metrics.rerank_applied = False
                metrics.rerank_time_ms = (time.time() - rerank_start) * 1000

        metrics.combined_results = len(filtered_results)
        metrics.total_time_ms = (time.time() - start_time) * 1000

        metrics.services = {
            "qdrant": bool(self.qdrant and not self._qdrant_init_failed),
            "neo4j": bool(self.neo4j and not self._neo4j_init_failed),
            "lexical": self.lexical_enabled,
            "graph_policy": effective_graph_policy.value,
            "graph_policy_requested": requested_graph_policy.value,
        }

        if self.cache_enabled and cache_key is not None:
            self._cache_set(cache_key, filtered_results, strategy)

        logger.info(
            f"Search completed: {metrics.combined_results} results "
            f"in {metrics.total_time_ms:.2f}ms (threshold={threshold:.2f})"
        )

        return filtered_results[:top_k], metrics

    def _search_vector(
        self,
        query_vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]],
        collection_name: Optional[str]
    ) -> List[QdrantResult]:
        """Search Qdrant vector database."""
        qdrant = self._ensure_qdrant()
        if qdrant is None:
            logger.warning("Qdrant client unavailable; skipping vector search")
            return []

        try:
            results = qdrant.search(
                query_vector=query_vector,
                top_k=top_k * 2,  # Get more for fusion
                filters=filters,
                collection_name=collection_name,
            )
            logger.debug(f"Vector search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _search_lexical(
        self,
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        collection_name: Optional[str],
    ) -> List[QdrantResult]:
        """Search lexical sparse signal from Qdrant payload text."""
        qdrant = self._ensure_qdrant()
        if qdrant is None:
            logger.warning("Qdrant client unavailable; skipping lexical search")
            return []

        try:
            results = qdrant.search_lexical(
                query_text=query_text,
                top_k=top_k * 2,
                filters=filters,
                collection_name=collection_name,
                max_scan_points=self.lexical_max_scan_points,
                scan_ratio=self.lexical_scan_ratio,
                max_scan_cap=self.lexical_max_scan_cap,
                page_size=self.lexical_page_size,
                bm25_k1=self.lexical_bm25_k1,
                bm25_b=self.lexical_bm25_b,
            )
            logger.debug(f"Lexical search returned {len(results)} results")
            return results
        except Exception as err:
            logger.error(f"Lexical search failed: {err}")
            return []

    def _search_graph(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        vector_index_name: Optional[str],
        graph_policy: GraphRetrievalPolicy,
    ) -> List[Neo4jResult]:
        """Search Neo4j graph database."""
        neo4j = self._ensure_neo4j()
        if neo4j is None:
            logger.warning("Neo4j client unavailable; skipping graph search")
            return []

        try:
            if graph_policy == GraphRetrievalPolicy.GLOBAL:
                results = neo4j.keyword_search_nodes(
                    query_text=query_text,
                    top_k=top_k * 2,
                    scan_limit=self.graph_global_scan_nodes,
                    max_terms=self.graph_global_max_terms,
                )
                logger.debug(f"Graph global keyword search returned {len(results)} results")
                return results

            if graph_policy == GraphRetrievalPolicy.COMMUNITY:
                summary_results = neo4j.community_summary_search(
                    query_text=query_text,
                    top_k=max(top_k * 2, self.graph_community_top_communities),
                    version=self.graph_community_version,
                    scan_limit=self.graph_community_scan_nodes,
                    max_terms=self.graph_community_max_terms,
                )

                community_scores: Dict[str, float] = {}
                for result in summary_results[: self.graph_community_top_communities]:
                    community_id = str(result.properties.get("community_id", "")).strip()
                    if community_id:
                        community_scores[community_id] = float(result.score)

                member_results: List[Neo4jResult] = []
                if community_scores:
                    member_results = neo4j.community_member_search(
                        community_scores=community_scores,
                        top_k=top_k * 2,
                        members_per_community=self.graph_community_members_per_community,
                    )

                base_results: List[Neo4jResult] = []
                if vector_index_name:
                    base_results = neo4j.vector_similarity_search(
                        index_name=vector_index_name,
                        query_vector=query_vector,
                        top_k=max(top_k * 2, self.graph_local_seed_k),
                        filters=filters,
                    )

                merged = self._merge_weighted_graph_results(
                    weighted_groups=[
                        (base_results, self.graph_community_vector_weight),
                        (summary_results, self.graph_community_summary_weight),
                        (member_results, self.graph_community_member_weight),
                    ],
                    top_k=top_k * 2,
                )
                if merged:
                    return merged
                if member_results:
                    return member_results
                if summary_results:
                    return summary_results
                if base_results:
                    return base_results

                # Community summaries might not be available yet; fall back to global scan.
                fallback_results = neo4j.keyword_search_nodes(
                    query_text=query_text,
                    top_k=top_k * 2,
                    scan_limit=self.graph_global_scan_nodes,
                    max_terms=self.graph_global_max_terms,
                )
                return fallback_results

            if not vector_index_name:
                logger.warning("No vector index provided for graph search")
                return []

            # Standard vector graph search anchors all graph policies.
            base_results = neo4j.vector_similarity_search(
                index_name=vector_index_name,
                query_vector=query_vector,
                top_k=max(top_k * 2, self.graph_local_seed_k),
                filters=filters,
            )

            if graph_policy == GraphRetrievalPolicy.STANDARD:
                logger.debug(f"Graph vector search returned {len(base_results)} results")
                return base_results

            local_results = self._build_local_graph_results(
                base_results=base_results,
                top_k=top_k,
            )

            if graph_policy == GraphRetrievalPolicy.LOCAL:
                return local_results

            global_results = neo4j.keyword_search_nodes(
                query_text=query_text,
                top_k=top_k * 2,
                scan_limit=self.graph_global_scan_nodes,
                max_terms=self.graph_global_max_terms,
            )

            # Drift policy combines local neighborhood and global thematic anchors.
            merged = self._merge_weighted_graph_results(
                weighted_groups=[
                    (local_results, self.graph_drift_local_weight),
                    (global_results, self.graph_drift_global_weight),
                ],
                top_k=top_k * 2,
            )
            return merged
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    def _build_local_graph_results(
        self,
        base_results: List[Neo4jResult],
        top_k: int,
    ) -> List[Neo4jResult]:
        """Build local graph retrieval results from vector seed nodes."""
        neo4j = self._ensure_neo4j()
        if neo4j is None:
            return base_results

        if not base_results:
            return []

        seed_scores: Dict[str, float] = {}
        for result in base_results[: self.graph_local_seed_k]:
            seed_scores[result.node_id] = float(result.score)

        expanded = neo4j.expand_from_seed_nodes(
            seed_scores=seed_scores,
            max_depth=self.graph_local_max_depth,
            limit=self.graph_local_neighbor_limit,
        )

        merged = self._merge_weighted_graph_results(
            weighted_groups=[(base_results, 0.6), (expanded, 0.4)],
            top_k=top_k * 2,
        )
        return merged

    def _merge_weighted_graph_results(
        self,
        weighted_groups: List[Tuple[List[Neo4jResult], float]],
        top_k: int,
        rrf_k: int = 60,
    ) -> List[Neo4jResult]:
        """Merge graph result groups using weighted reciprocal rank fusion."""
        merged: Dict[str, Dict[str, Any]] = {}

        for results, weight in weighted_groups:
            if not results or weight <= 0:
                continue
            for rank, result in enumerate(results, start=1):
                node_id = result.node_id
                if node_id not in merged:
                    merged[node_id] = {
                        "score": 0.0,
                        "labels": list(result.labels),
                        "properties": dict(result.properties),
                        "metadata": dict(result.metadata or {}),
                    }
                merged[node_id]["score"] += (weight / (rrf_k + rank)) + (0.05 * weight * result.score)
                merged[node_id]["metadata"].setdefault("graph_signals", []).append(
                    {
                        "weight": weight,
                        "rank": rank,
                        "source_score": result.score,
                    }
                )

        fused_results: List[Neo4jResult] = []
        for node_id, payload in merged.items():
            fused_results.append(
                Neo4jResult(
                    node_id=node_id,
                    score=float(payload["score"]),
                    labels=payload["labels"],
                    properties=payload["properties"],
                    metadata=payload["metadata"],
                )
            )
        fused_results.sort(key=lambda item: item.score, reverse=True)
        return fused_results[:top_k]

    def _fusion_combine(
        self,
        vector_results: List[QdrantResult],
        graph_results: List[Neo4jResult],
        lexical_results: List[QdrantResult],
        top_k: int,
        k: int = 60
    ) -> List[HybridSearchResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        Args:
            vector_results: Results from vector search
            graph_results: Results from graph search
            lexical_results: Results from lexical search
            top_k: Number of results to return
            k: RRF constant (default: 60)

        Returns:
            Combined and ranked results
        """
        # Build score maps
        rrf_scores: Dict[str, Dict[str, Any]] = {}

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            result_id = str(result.id)
            rrf_score = self.vector_weight / (k + rank)

            if result_id not in rrf_scores:
                rrf_scores[result_id] = {
                    "score": 0,
                    "vector_score": result.score,
                    "graph_score": None,
                    "lexical_score": None,
                    "metadata": result.metadata,
                    "content": result.metadata.get("text", ""),
                    "sources": set(["vector"]),
                }

            rrf_scores[result_id]["score"] += rrf_score

        # Process graph results
        for rank, result in enumerate(graph_results, start=1):
            result_id = result.node_id
            rrf_score = self.graph_weight / (k + rank)

            if result_id not in rrf_scores:
                rrf_scores[result_id] = {
                    "score": 0,
                    "vector_score": None,
                    "graph_score": result.score,
                    "lexical_score": None,
                    "metadata": result.properties,
                    "content": result.properties.get("text", ""),
                    "sources": set(["graph"]),
                    "graph_context": {
                        "labels": result.labels,
                        "node_id": result.node_id,
                    },
                }
            else:
                rrf_scores[result_id]["graph_score"] = result.score
                rrf_scores[result_id]["sources"].add("graph")
                rrf_scores[result_id]["graph_context"] = {
                    "labels": result.labels,
                    "node_id": result.node_id,
                }

            rrf_scores[result_id]["score"] += rrf_score

        # Process lexical results
        if lexical_results and self.lexical_weight > 0:
            lexical_scores = [result.score for result in lexical_results]
            max_lex = max(lexical_scores)
            min_lex = min(lexical_scores)
            denom = max(max_lex - min_lex, 1e-9)

            for rank, result in enumerate(lexical_results, start=1):
                result_id = str(result.id)
                rrf_score = self.lexical_weight / (k + rank)
                normalized_lex = (result.score - min_lex) / denom if max_lex > min_lex else 1.0

                if result_id not in rrf_scores:
                    rrf_scores[result_id] = {
                        "score": 0,
                        "vector_score": None,
                        "graph_score": None,
                        "lexical_score": normalized_lex,
                        "metadata": result.metadata,
                        "content": result.metadata.get("text", ""),
                        "sources": set(["lexical"]),
                    }
                else:
                    rrf_scores[result_id]["lexical_score"] = normalized_lex
                    rrf_scores[result_id]["sources"].add("lexical")

                rrf_scores[result_id]["score"] += rrf_score

        # Convert to HybridSearchResult
        combined_results = []

        for result_id, data in rrf_scores.items():
            # Calculate confidence based on source diversity
            confidence = self._calculate_confidence(
                data["vector_score"],
                data["graph_score"],
                data.get("lexical_score"),
                len(data["sources"])
            )

            # Determine source
            if len(data["sources"]) > 1:
                source = "hybrid"
            else:
                source = list(data["sources"])[0]

            result = HybridSearchResult(
                id=result_id,
                content=data["content"],
                score=data["score"],
                confidence=confidence,
                source=source,
                vector_score=data["vector_score"],
                graph_score=data["graph_score"],
                lexical_score=data.get("lexical_score"),
                metadata=data["metadata"],
                graph_context=data.get("graph_context"),
            )

            combined_results.append(result)

        # Sort by score
        combined_results.sort(key=lambda x: x.score, reverse=True)

        return combined_results[:top_k]

    def _convert_vector_results(
        self,
        results: List[QdrantResult]
    ) -> List[HybridSearchResult]:
        """Convert Qdrant results to HybridSearchResult."""
        return [
            HybridSearchResult(
                id=str(r.id),
                content=r.metadata.get("text", ""),
                score=r.score,
                confidence=r.score,  # Direct score as confidence
                source="vector",
                vector_score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ]

    def _convert_graph_results(
        self,
        results: List[Neo4jResult]
    ) -> List[HybridSearchResult]:
        """Convert Neo4j results to HybridSearchResult."""
        return [
            HybridSearchResult(
                id=r.node_id,
                content=r.properties.get("text", ""),
                score=r.score,
                confidence=r.score,  # Direct score as confidence
                source="graph",
                graph_score=r.score,
                metadata=r.properties,
                graph_context={
                    "labels": r.labels,
                    "node_id": r.node_id,
                },
            )
            for r in results
        ]

    def _convert_lexical_results(
        self,
        results: List[QdrantResult],
    ) -> List[HybridSearchResult]:
        """Convert lexical Qdrant results to HybridSearchResult."""
        if not results:
            return []

        max_score = max(r.score for r in results)
        min_score = min(r.score for r in results)
        denom = max(max_score - min_score, 1e-9)

        converted: List[HybridSearchResult] = []
        for result in results:
            normalized = (result.score - min_score) / denom if max_score > min_score else 1.0
            converted.append(
                HybridSearchResult(
                    id=str(result.id),
                    content=result.metadata.get("text", ""),
                    score=normalized,
                    confidence=normalized,
                    source="lexical",
                    lexical_score=normalized,
                    metadata=result.metadata,
                )
            )
        return converted

    def _build_cache_key(
        self,
        *,
        query_text: str,
        query_vector: Optional[List[float]],
        strategy: SearchStrategy,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        collection_name: Optional[str],
        vector_index_name: Optional[str],
        graph_policy: str,
    ) -> Hashable:
        return (
            strategy.value,
            top_k,
            query_text,
            self._hash_vector(query_vector),
            self._to_hashable(filters),
            collection_name,
            vector_index_name,
            graph_policy,
        )

    def _hash_vector(self, vector: Optional[List[float]]) -> Optional[Tuple[float, ...]]:
        if vector is None:
            return None
        return tuple(round(float(x), 4) for x in vector)

    def _to_hashable(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return tuple(sorted((k, self._to_hashable(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple, set)):
            return tuple(self._to_hashable(v) for v in value)
        return value

    def _cache_get(self, key: Hashable) -> Optional[_CacheEntry]:
        if not self.cache_enabled:
            return None
        entry = self._cache.get(key)
        if entry is None:
            return None
        if self.cache_ttl > 0 and (time.time() - entry.timestamp) > self.cache_ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return entry

    def _cache_set(
        self,
        key: Hashable,
        results: List[HybridSearchResult],
        strategy: SearchStrategy,
    ) -> None:
        if not self.cache_enabled:
            return
        cloned = self._clone_results(results)
        self._cache[key] = _CacheEntry(
            results=cloned,
            timestamp=time.time(),
            strategy=strategy,
        )
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_max_size:
            self._cache.popitem(last=False)

    def _clone_results(self, results: List[HybridSearchResult]) -> List[HybridSearchResult]:
        return [copy.deepcopy(r) for r in results]

    def _calculate_confidence(
        self,
        vector_score: Optional[float],
        graph_score: Optional[float],
        lexical_score: Optional[float] = None,
        source_count: int = 1
    ) -> float:
        """
        Calculate confidence score based on multiple factors.

        Args:
            vector_score: Score from vector search
            graph_score: Score from graph search
            lexical_score: Score from lexical search
            source_count: Number of sources that found this result

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from scores
        scores = [s for s in [vector_score, graph_score, lexical_score] if s is not None]

        if not scores:
            return 0.5

        avg_score = sum(scores) / len(scores)

        # Boost confidence if found in multiple sources
        diversity_boost = 0.1 * (source_count - 1)

        confidence = min(1.0, avg_score + diversity_boost)

        return confidence

    def enrich_with_graph_context(
        self,
        result: HybridSearchResult,
        max_depth: int = 2,
        relationship_types: Optional[List[str]] = None
    ) -> HybridSearchResult:
        """
        Enrich a result with additional graph context.

        Args:
            result: Search result to enrich
            max_depth: Maximum traversal depth
            relationship_types: Relationship types to follow

        Returns:
            Enriched result
        """
        if result.source == "vector" or not result.graph_context:
            return result

        neo4j = self._ensure_neo4j()
        if neo4j is None:
            logger.warning("Neo4j client unavailable; cannot enrich graph context")
            return result

        try:
            # Get graph neighborhood
            neighbors = neo4j.traverse_graph(
                start_node_id=result.graph_context["node_id"],
                relationship_types=relationship_types,
                max_depth=max_depth,
                limit=20,
            )

            # Add to graph context
            if result.graph_context is None:
                result.graph_context = {}

            result.graph_context["neighbors"] = neighbors

            logger.debug(f"Enriched result {result.id} with {len(neighbors)} neighbors")

        except Exception as e:
            logger.warning(f"Failed to enrich result: {e}")

        return result

    def batch_enrich_results(
        self,
        results: List[HybridSearchResult],
        max_depth: int = 2
    ) -> List[HybridSearchResult]:
        """
        Enrich multiple results with graph context in batch.

        Args:
            results: List of results to enrich
            max_depth: Maximum traversal depth

        Returns:
            List of enriched results
        """
        enriched = []

        for result in results:
            enriched_result = self.enrich_with_graph_context(
                result, max_depth=max_depth
            )
            enriched.append(enriched_result)

        return enriched

    def rerank_results(
        self,
        results: List[HybridSearchResult],
        query_text: str,
        use_diversity: bool = True
    ) -> List[HybridSearchResult]:
        """
        Rerank results using additional criteria.

        Args:
            results: Results to rerank
            query_text: Original query text
            use_diversity: Apply diversity penalty to similar results

        Returns:
            Reranked results
        """
        if not results:
            return results

        # Apply diversity if requested
        if use_diversity:
            results = self._apply_diversity_penalty(results)

        # Sort by confidence * score
        results.sort(
            key=lambda x: x.confidence * x.score,
            reverse=True
        )

        return results

    def _apply_diversity_penalty(
        self,
        results: List[HybridSearchResult],
        similarity_threshold: float = 0.9
    ) -> List[HybridSearchResult]:
        """
        Apply diversity penalty to reduce redundant results.

        Args:
            results: Results to process
            similarity_threshold: Threshold for considering results similar

        Returns:
            Results with diversity penalty applied
        """
        if len(results) <= 1:
            return results

        # Simple content-based deduplication
        seen_contents: Set[str] = set()
        diverse_results = []

        for result in results:
            content_hash = hash(result.content[:200])  # Use first 200 chars

            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                diverse_results.append(result)
            else:
                # Apply penalty to score
                result.score *= 0.5
                result.confidence *= 0.8
                diverse_results.append(result)

        return diverse_results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from both databases.

        Returns:
            Combined statistics
        """
        stats: Dict[str, Any] = {
            "weights": {
                "vector": self.vector_weight,
                "graph": self.graph_weight,
                "lexical": self.lexical_weight,
            }
        }
        stats["lexical"] = {
            "enabled": self.lexical_enabled,
            "max_scan_points": self.lexical_max_scan_points,
            "scan_ratio": self.lexical_scan_ratio,
            "max_scan_cap": self.lexical_max_scan_cap,
            "page_size": self.lexical_page_size,
            "bm25_k1": self.lexical_bm25_k1,
            "bm25_b": self.lexical_bm25_b,
        }
        stats["graph_policy"] = {
            "mode": self.graph_policy_mode,
            "local_seed_k": self.graph_local_seed_k,
            "local_max_depth": self.graph_local_max_depth,
            "local_neighbor_limit": self.graph_local_neighbor_limit,
            "global_scan_nodes": self.graph_global_scan_nodes,
            "global_max_terms": self.graph_global_max_terms,
            "drift_local_weight": self.graph_drift_local_weight,
            "drift_global_weight": self.graph_drift_global_weight,
            "community_scan_nodes": self.graph_community_scan_nodes,
            "community_max_terms": self.graph_community_max_terms,
            "community_top_communities": self.graph_community_top_communities,
            "community_members_per_community": self.graph_community_members_per_community,
            "community_vector_weight": self.graph_community_vector_weight,
            "community_summary_weight": self.graph_community_summary_weight,
            "community_member_weight": self.graph_community_member_weight,
            "community_version": self.graph_community_version,
        }

        neo4j = self._ensure_neo4j()
        qdrant = self._ensure_qdrant()

        if neo4j:
            try:
                stats["neo4j"] = neo4j.get_statistics()
            except Exception as err:
                logger.warning(f"Failed to fetch Neo4j statistics: {err}")
        else:
            stats["neo4j"] = None

        if qdrant:
            try:
                stats["qdrant"] = qdrant.get_statistics()
            except Exception as err:
                logger.warning(f"Failed to fetch Qdrant statistics: {err}")
        else:
            stats["qdrant"] = None

        return stats

    def health_check(self) -> Dict[str, bool]:
        """
        Check health of both databases.

        Returns:
            Health status for each database
        """
        neo4j = self._ensure_neo4j()
        qdrant = self._ensure_qdrant()

        return {
            "neo4j": neo4j.health_check() if neo4j else False,
            "qdrant": qdrant.health_check() if qdrant else False,
        }

    def close(self) -> None:
        """Close connections to both databases."""
        if self.neo4j and self._owns_neo4j:
            try:
                self.neo4j.close()
            except Exception as err:
                logger.warning(f"Failed to close Neo4j manager: {err}")

        if self.qdrant and self._owns_qdrant:
            try:
                self.qdrant.close()
            except Exception as err:
                logger.warning(f"Failed to close Qdrant manager: {err}")

        logger.info("Hybrid retriever closed")
