"""
Advanced Retrieval Pipeline for DocTags RAG.

Orchestrates all advanced retrieval features:
- Query analysis and routing
- Query expansion
- Multi-stage retrieval
- Confidence scoring
- Iterative refinement
- Result reranking
- Caching
"""

from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time

from loguru import logger

from .hybrid_retriever import HybridRetriever, HybridSearchResult
from .query_router import QueryRouter, QueryAnalysis, RetrievalStrategy
from .query_expansion import QueryExpander, ExpandedQuery
from .confidence_scorer import ConfidenceScorer, ConfidenceScore, CorrectiveAction
from .iterative_refiner import IterativeRefiner, RefinedResult, RefinementStep
from .reranker import Reranker, RerankedResult
from .cache_manager import CacheManager


@dataclass
class PipelineConfig:
    """Configuration for advanced pipeline."""
    enable_query_expansion: bool = True
    enable_iterative_refinement: bool = True
    enable_reranking: bool = True
    enable_caching: bool = True
    enable_confidence_scoring: bool = True

    max_refinement_iterations: int = 3
    min_confidence_threshold: float = 0.7
    use_cross_encoder: bool = True
    cache_ttl: int = 3600

    top_k: int = 10
    min_results: int = 3


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval pipeline."""
    total_time_ms: float
    query_analysis_time_ms: float
    expansion_time_ms: float
    retrieval_time_ms: float
    confidence_scoring_time_ms: float
    refinement_time_ms: float
    reranking_time_ms: float

    cache_hit: bool
    query_type: str
    retrieval_strategy: str
    results_count: int
    refinement_iterations: int
    avg_confidence: float


@dataclass
class PipelineResult:
    """Complete pipeline result with metadata."""
    query: str
    results: List[Dict[str, Any]]
    metrics: RetrievalMetrics
    query_analysis: QueryAnalysis
    expanded_query: Optional[ExpandedQuery] = None
    confidence_scores: Optional[List[ConfidenceScore]] = None
    refinement_steps: Optional[List[RefinementStep]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedRetrievalPipeline:
    """
    Advanced retrieval pipeline orchestrating all components.

    Pipeline stages:
    1. Query analysis and routing
    2. Query expansion (optional)
    3. Initial retrieval
    4. Confidence scoring
    5. Iterative refinement (if needed)
    6. Result reranking
    7. Final aggregation
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        config: Optional[PipelineConfig] = None,
        cache_dir: Optional[Path] = None,
        performance_file: Optional[Path] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None
    ):
        """
        Initialize advanced pipeline.

        Args:
            hybrid_retriever: Hybrid retriever instance
            config: Pipeline configuration
            cache_dir: Directory for caching
            performance_file: File for routing performance tracking
        """
        self.hybrid_retriever = hybrid_retriever
        self.config = config or PipelineConfig()

        # Initialize components
        self.query_router = QueryRouter(
            enable_learning=True,
            performance_file=performance_file
        )

        self.query_expander = QueryExpander(
            enable_wordnet=True,
            enable_semantic=True,
            enable_contextual=True
        )

        self.confidence_scorer = ConfidenceScorer()

        self.iterative_refiner = IterativeRefiner(
            confidence_scorer=self.confidence_scorer,
            max_iterations=self.config.max_refinement_iterations,
            min_confidence_threshold=self.config.min_confidence_threshold
        )

        self.reranker = Reranker(
            enable_cross_encoder=self.config.use_cross_encoder
        )

        self.cache_manager = None
        if self.config.enable_caching:
            self.cache_manager = CacheManager(
                cache_dir=cache_dir,
                query_ttl=self.config.cache_ttl
            )

        logger.info("Advanced retrieval pipeline initialized")

        # Optional embedding provider (e.g., OpenAI, local encoder)
        self.embedding_function = embedding_function

    def retrieve(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        context: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
        override_config: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Execute complete retrieval pipeline.

        Args:
            query: User query
            query_vector: Pre-computed query vector (optional)
            context: Additional context for retrieval
            top_k: Number of results to return
            override_config: Override configuration for this query

        Returns:
            Complete pipeline result with metrics
        """
        start_time = time.time()
        top_k = top_k or self.config.top_k

        # Apply config overrides
        config = self._apply_config_overrides(override_config)

        # Initialize metrics
        metrics = {
            "cache_hit": False,
            "query_analysis_time_ms": 0,
            "expansion_time_ms": 0,
            "retrieval_time_ms": 0,
            "confidence_scoring_time_ms": 0,
            "refinement_time_ms": 0,
            "reranking_time_ms": 0
        }

        logger.info(f"Pipeline started for query: {query}")

        # Check cache
        if config.enable_caching and self.cache_manager:
            cached_results = self.cache_manager.get_query_results(
                query, query_vector
            )
            if cached_results:
                logger.info("Cache hit!")
                metrics["cache_hit"] = True
                return self._build_cached_result(
                    query, cached_results, metrics, start_time
                )

        # Stage 1: Query analysis and routing
        stage_start = time.time()
        strategy, query_analysis = self.query_router.route_query(query, context)
        metrics["query_analysis_time_ms"] = (time.time() - stage_start) * 1000

        # Stage 2: Query expansion
        expanded_query = None
        if config.enable_query_expansion:
            stage_start = time.time()
            expanded_query = self.query_expander.expand_query(
                query,
                strategy="comprehensive"
            )
            metrics["expansion_time_ms"] = (time.time() - stage_start) * 1000
            logger.info(f"Query expanded: {expanded_query.expanded_query}")

        # Stage 3: Initial retrieval
        stage_start = time.time()
        query_for_retrieval = (
            expanded_query.expanded_query if expanded_query else query
        )

        embedding_target = query_for_retrieval

        if query_vector is None and self.embedding_function:
            try:
                query_vector = self.embedding_function(embedding_target)
            except Exception as embed_err:
                logger.error(f"Failed to generate query embedding: {embed_err}")
                query_vector = None

        if query_vector is not None and not isinstance(query_vector, list):
            try:
                # Handle numpy arrays or other sequences
                query_vector = list(query_vector)
            except TypeError:
                query_vector = None

        has_embedding = query_vector is not None and len(query_vector) > 0
        requires_embedding = strategy in {
            RetrievalStrategy.VECTOR_ONLY,
            RetrievalStrategy.GRAPH_ONLY,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.MULTI_STAGE
        }

        if requires_embedding and not has_embedding:
            message = (
                "No query embedding available for retrieval. Supply a pre-computed "
                "query_vector, configure an embedding_function, or choose a retrieval "
                "strategy that does not require embeddings."
            )
            logger.error(message)
            raise ValueError(message)

        initial_results, search_metrics = self.hybrid_retriever.search(
            query_vector=query_vector,
            query_text=query_for_retrieval,
            top_k=top_k * 2,  # Get more for refinement
            strategy=strategy,
            filters=context.get("filters") if context else None
        )
        metrics["retrieval_time_ms"] = (time.time() - stage_start) * 1000

        # Convert to dict format
        results_dicts = [
            {
                "content": r.content,
                "score": r.score,
                "confidence": r.confidence,
                "id": r.id,
                "metadata": r.metadata,
                "vector_score": r.vector_score,
                "graph_score": r.graph_score,
                "graph_context": r.graph_context
            }
            for r in initial_results
        ]

        # Stage 4: Confidence scoring
        confidence_scores = None
        if config.enable_confidence_scoring:
            stage_start = time.time()
            confidence_scores = self.confidence_scorer.score_results_batch(
                query, results_dicts
            )
            metrics["confidence_scoring_time_ms"] = (time.time() - stage_start) * 1000

            # Update result confidences
            for result, conf_score in zip(results_dicts, confidence_scores):
                result["confidence"] = conf_score.overall_score

        # Stage 5: Iterative refinement (if needed)
        refinement_steps = None
        if config.enable_iterative_refinement and confidence_scores:
            stage_start = time.time()

            # Check if refinement is needed
            avg_confidence = sum(cs.overall_score for cs in confidence_scores) / len(confidence_scores)

            if avg_confidence < config.min_confidence_threshold or len(results_dicts) < config.min_results:
                logger.info(f"Initiating refinement (avg confidence: {avg_confidence:.2f})")

                # Define retrieval function for refiner
                def retrieval_func(refined_query: str, ctx: Optional[Dict] = None):
                    refined_results, _ = self.hybrid_retriever.search(
                        query_vector=query_vector,
                        query_text=refined_query,
                        top_k=top_k,
                        strategy=strategy,
                        filters=ctx.get("filters") if ctx else None
                    )
                    return [
                        {
                            "content": r.content,
                            "score": r.score,
                            "confidence": r.confidence,
                            "id": r.id,
                            "metadata": r.metadata
                        }
                        for r in refined_results
                    ]

                refined_results, refinement_steps = self.iterative_refiner.refine_retrieval(
                    original_query=query,
                    initial_results=results_dicts,
                    retrieval_func=retrieval_func,
                    context=context
                )

                # Update results with refined ones
                results_dicts = [
                    {
                        "content": r.content,
                        "score": r.score,
                        "confidence": r.confidence,
                        "id": r.result_id,
                        "metadata": r.metadata
                    }
                    for r in refined_results
                ]

            metrics["refinement_time_ms"] = (time.time() - stage_start) * 1000

        # Stage 6: Reranking
        if config.enable_reranking:
            stage_start = time.time()

            reranked_results = self.reranker.rerank(
                query=query,
                results=results_dicts,
                top_k=top_k,
                enable_diversity=True
            )

            # Convert back to dict format
            results_dicts = [
                {
                    "content": r.content,
                    "score": r.reranked_score,
                    "original_score": r.original_score,
                    "rank": r.rank,
                    "original_rank": r.original_rank,
                    "id": r.result_id,
                    "metadata": r.metadata
                }
                for r in reranked_results
            ]

            metrics["reranking_time_ms"] = (time.time() - stage_start) * 1000

        # Cache results
        if config.enable_caching and self.cache_manager:
            cache_vector = None
            if query_vector is not None:
                try:
                    import numpy as np

                    cache_vector = np.asarray(query_vector, dtype=float)
                except Exception:
                    cache_vector = None

            self.cache_manager.cache_query_results(
                query,
                cache_vector,
                results_dicts
            )

        # Build final result
        total_time = time.time() - start_time

        final_metrics = RetrievalMetrics(
            total_time_ms=total_time * 1000,
            query_analysis_time_ms=metrics["query_analysis_time_ms"],
            expansion_time_ms=metrics["expansion_time_ms"],
            retrieval_time_ms=metrics["retrieval_time_ms"],
            confidence_scoring_time_ms=metrics["confidence_scoring_time_ms"],
            refinement_time_ms=metrics["refinement_time_ms"],
            reranking_time_ms=metrics["reranking_time_ms"],
            cache_hit=metrics["cache_hit"],
            query_type=query_analysis.query_type.value,
            retrieval_strategy=strategy.value,
            results_count=len(results_dicts),
            refinement_iterations=len(refinement_steps) if refinement_steps else 0,
            avg_confidence=sum(r.get("confidence", 0.0) for r in results_dicts) / len(results_dicts) if results_dicts else 0.0
        )

        result = PipelineResult(
            query=query,
            results=results_dicts,
            metrics=final_metrics,
            query_analysis=query_analysis,
            expanded_query=expanded_query,
            confidence_scores=confidence_scores,
            refinement_steps=refinement_steps,
            metadata={
                "search_metrics": search_metrics.__dict__ if hasattr(search_metrics, '__dict__') else {}
            }
        )

        logger.info(
            f"Pipeline complete: {len(results_dicts)} results in {total_time*1000:.2f}ms "
            f"(avg confidence: {final_metrics.avg_confidence:.2f})"
        )

        # Record performance for learning
        if config.enable_confidence_scoring:
            success = final_metrics.avg_confidence >= config.min_confidence_threshold
            self.query_router.record_performance(
                query=query,
                strategy=strategy,
                query_type=query_analysis.query_type,
                success=success,
                confidence=final_metrics.avg_confidence,
                num_results=len(results_dicts)
            )

        return result

    def _apply_config_overrides(
        self,
        overrides: Optional[Dict[str, Any]]
    ) -> PipelineConfig:
        """Apply configuration overrides."""
        if not overrides:
            return self.config

        config_dict = self.config.__dict__.copy()
        config_dict.update(overrides)
        return PipelineConfig(**config_dict)

    def _build_cached_result(
        self,
        query: str,
        cached_results: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        start_time: float
    ) -> PipelineResult:
        """Build result from cached data."""
        total_time = time.time() - start_time

        # Create dummy query analysis
        query_analysis = QueryAnalysis(
            query_text=query,
            query_type=self.query_router.analyze_query(query).query_type,
            complexity=self.query_router.analyze_query(query).complexity,
            recommended_strategy=self.query_router.analyze_query(query).recommended_strategy,
            confidence=1.0
        )

        final_metrics = RetrievalMetrics(
            total_time_ms=total_time * 1000,
            query_analysis_time_ms=0,
            expansion_time_ms=0,
            retrieval_time_ms=0,
            confidence_scoring_time_ms=0,
            refinement_time_ms=0,
            reranking_time_ms=0,
            cache_hit=True,
            query_type=query_analysis.query_type.value,
            retrieval_strategy="cached",
            results_count=len(cached_results),
            refinement_iterations=0,
            avg_confidence=sum(r.get("confidence", 0.0) for r in cached_results) / len(cached_results) if cached_results else 0.0
        )

        return PipelineResult(
            query=query,
            results=cached_results,
            metrics=final_metrics,
            query_analysis=query_analysis
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = {
            "query_router": self.query_router.get_statistics(),
            "query_expander": self.query_expander.get_statistics(),
            "reranker": self.reranker.get_statistics(),
            "hybrid_retriever": self.hybrid_retriever.get_statistics()
        }

        if self.cache_manager:
            stats["cache"] = self.cache_manager.get_statistics()

        return stats

    def explain_result(
        self,
        result: PipelineResult,
        result_index: int = 0
    ) -> Dict[str, Any]:
        """
        Explain a specific result from the pipeline.

        Args:
            result: Pipeline result
            result_index: Index of result to explain

        Returns:
            Explanation dictionary
        """
        if result_index >= len(result.results):
            return {}

        explanation = {
            "query": result.query,
            "query_type": result.query_analysis.query_type.value,
            "query_complexity": result.query_analysis.complexity.value,
            "strategy_used": result.query_analysis.recommended_strategy.value,
            "result_rank": result_index + 1,
            "result_score": result.results[result_index].get("score", 0.0),
            "result_confidence": result.results[result_index].get("confidence", 0.0),
        }

        # Add expansion info
        if result.expanded_query:
            explanation["query_expanded"] = True
            explanation["expansion_terms"] = (
                result.expanded_query.synonyms +
                result.expanded_query.related_entities +
                result.expanded_query.semantic_terms
            )[:5]

        # Add refinement info
        if result.refinement_steps:
            explanation["refinement_applied"] = True
            explanation["refinement_iterations"] = len(result.refinement_steps)
            explanation["refinement_queries"] = [
                step.refined_query for step in result.refinement_steps
            ]

        # Add confidence info
        if result.confidence_scores and result_index < len(result.confidence_scores):
            conf_score = result.confidence_scores[result_index]
            explanation["confidence_level"] = conf_score.level.value
            explanation["confidence_reasoning"] = conf_score.reasoning
            explanation["corrective_action"] = conf_score.corrective_action.value

        return explanation

    def batch_retrieve(
        self,
        queries: List[str],
        query_vectors: Optional[List[List[float]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[PipelineResult]:
        """
        Process multiple queries in batch.

        Args:
            queries: List of queries
            query_vectors: Optional pre-computed vectors
            context: Shared context for all queries

        Returns:
            List of pipeline results
        """
        results = []

        for i, query in enumerate(queries):
            query_vector = query_vectors[i] if query_vectors else None

            try:
                result = self.retrieve(
                    query=query,
                    query_vector=query_vector,
                    context=context
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Batch retrieval failed for query {i}: {e}")
                continue

        logger.info(f"Batch retrieval complete: {len(results)}/{len(queries)} successful")

        return results

    def health_check(self) -> Dict[str, bool]:
        """Check health of all components."""
        health = {
            "hybrid_retriever": self.hybrid_retriever.health_check()
        }

        return health

    def close(self) -> None:
        """Cleanup resources."""
        self.hybrid_retriever.close()
        logger.info("Advanced pipeline closed")
