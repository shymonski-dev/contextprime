"""
Hybrid Retrieval Manager for DocTags RAG System.

Combines Neo4j graph database and Qdrant vector database for hybrid search:
- Fusion scoring to combine results from both sources
- Configurable weights for vector vs graph results
- Query routing based on query type
- Result ranking and deduplication
- Confidence scoring
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict

from loguru import logger

from ..knowledge_graph.neo4j_manager import Neo4jManager, SearchResult as Neo4jResult
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
    metadata: Dict[str, Any] = field(default_factory=dict)
    graph_context: Optional[Dict[str, Any]] = None


@dataclass
class SearchMetrics:
    """Metrics for search performance."""
    query_type: QueryType
    strategy: SearchStrategy
    vector_results: int
    graph_results: int
    combined_results: int
    vector_time_ms: float
    graph_time_ms: float
    fusion_time_ms: float
    total_time_ms: float


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

        self.default_collection_name = getattr(_settings.qdrant, "collection_name", None)
        hybrid_cfg = getattr(_settings.retrieval, "hybrid_search", {}) or {}
        self.default_graph_vector_index = hybrid_cfg.get("graph_vector_index")

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

        logger.info(
            f"Hybrid retriever initialized (vector: {self.vector_weight:.2f}, "
            f"graph: {self.graph_weight:.2f}, min_conf={self.min_confidence_threshold:.2f})"
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

        # Return type with highest score
        max_type = max(scores.items(), key=lambda x: x[1])

        # If complex or multiple matches, use hybrid
        if max_type[1] >= 2 or len(scores) > 1:
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

        logger.info(f"Query type: {query_type.value}, Strategy: {strategy.value}")

        # Initialize metrics
        metrics = SearchMetrics(
            query_type=query_type,
            strategy=strategy,
            vector_results=0,
            graph_results=0,
            combined_results=0,
            vector_time_ms=0,
            graph_time_ms=0,
            fusion_time_ms=0,
            total_time_ms=0,
        )

        # Execute searches based on strategy
        vector_results = []
        graph_results = []

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
                effective_collection = collection_name or self.default_collection_name
                vector_results = self._search_vector(
                    query_vector, top_k, filters, effective_collection
                )
                metrics.vector_time_ms = (time.time() - vector_start) * 1000
                metrics.vector_results = len(vector_results)
            else:
                logger.warning("Vector search requested but no query embedding provided; skipping vector lookup")

        if strategy in [SearchStrategy.GRAPH_ONLY, SearchStrategy.HYBRID]:
            if can_use_vector:
                graph_start = time.time()
                effective_index = vector_index_name or self.default_graph_vector_index
                graph_results = self._search_graph(
                    query_vector, query_text, top_k, filters, effective_index
                )
                metrics.graph_time_ms = (time.time() - graph_start) * 1000
                metrics.graph_results = len(graph_results)
            else:
                logger.warning("Graph vector search requested but no query embedding provided; skipping graph lookup")

        # Combine results
        fusion_start = time.time()
        if strategy == SearchStrategy.HYBRID:
            combined_results = self._fusion_combine(
                vector_results, graph_results, top_k
            )
        elif strategy == SearchStrategy.VECTOR_ONLY:
            combined_results = self._convert_vector_results(vector_results)
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

        metrics.combined_results = len(filtered_results)
        metrics.total_time_ms = (time.time() - start_time) * 1000

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

    def _search_graph(
        self,
        query_vector: List[float],
        query_text: str,
        top_k: int,
        filters: Optional[Dict[str, Any]],
        vector_index_name: Optional[str]
    ) -> List[Neo4jResult]:
        """Search Neo4j graph database."""
        neo4j = self._ensure_neo4j()
        if neo4j is None:
            logger.warning("Neo4j client unavailable; skipping graph search")
            return []

        try:
            # Try vector similarity search if index exists
            if vector_index_name:
                results = neo4j.vector_similarity_search(
                    index_name=vector_index_name,
                    query_vector=query_vector,
                    top_k=top_k * 2,  # Get more for fusion
                    filters=filters,
                )
                logger.debug(f"Graph vector search returned {len(results)} results")
                return results
            else:
                # Fallback to text-based search
                logger.warning("No vector index provided for graph search")
                return []
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []

    def _fusion_combine(
        self,
        vector_results: List[QdrantResult],
        graph_results: List[Neo4jResult],
        top_k: int,
        k: int = 60
    ) -> List[HybridSearchResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        Args:
            vector_results: Results from vector search
            graph_results: Results from graph search
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

        # Convert to HybridSearchResult
        combined_results = []

        for result_id, data in rrf_scores.items():
            # Calculate confidence based on source diversity
            confidence = self._calculate_confidence(
                data["vector_score"],
                data["graph_score"],
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

    def _calculate_confidence(
        self,
        vector_score: Optional[float],
        graph_score: Optional[float],
        source_count: int
    ) -> float:
        """
        Calculate confidence score based on multiple factors.

        Args:
            vector_score: Score from vector search
            graph_score: Score from graph search
            source_count: Number of sources that found this result

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from scores
        scores = [s for s in [vector_score, graph_score] if s is not None]

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
            }
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
        if self.neo4j:
            try:
                self.neo4j.close()
            except Exception as err:
                logger.warning(f"Failed to close Neo4j manager: {err}")

        if self.qdrant:
            try:
                self.qdrant.close()
            except Exception as err:
                logger.warning(f"Failed to close Qdrant manager: {err}")

        logger.info("Hybrid retriever closed")
