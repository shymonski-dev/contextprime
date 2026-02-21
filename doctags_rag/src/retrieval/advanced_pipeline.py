"""
Advanced Retrieval Pipeline for Contextprime.

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
import hashlib
import re
import time

from loguru import logger

from .hybrid_retriever import (
    HybridRetriever,
    HybridSearchResult,
    SearchStrategy as HybridSearchStrategy,
    SearchMetrics as HybridSearchMetrics,
)
from .query_router import QueryRouter, QueryAnalysis, RetrievalStrategy
from .query_expansion import QueryExpander, ExpandedQuery
from .confidence_scorer import ConfidenceScorer, ConfidenceScore, CorrectiveAction
from .iterative_refiner import IterativeRefiner, RefinedResult, RefinementStep
from .reranker import Reranker, RerankedResult
from .cache_manager import CacheManager
from .context_selector import TrainableContextSelector


@dataclass
class PipelineConfig:
    """Configuration for advanced pipeline."""
    enable_query_expansion: bool = True
    enable_multi_query_retrieval: bool = True
    enable_corrective_retrieval: bool = False
    enable_context_pruning: bool = False
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
    max_query_variants: int = 3
    multi_query_rrf_k: int = 60
    multi_query_min_variants: int = 2
    corrective_min_results: int = 3
    corrective_min_average_confidence: float = 0.55
    corrective_top_k_multiplier: float = 2.0
    corrective_force_hybrid_strategy: bool = True
    corrective_max_variants: int = 2
    corrective_max_initial_variants: int = 3
    context_pruning_max_sentences_per_result: int = 4
    context_pruning_max_chars_per_result: int = 900
    context_pruning_min_sentence_tokens: int = 3
    enable_trainable_context_selector: bool = False
    context_selector_model_path: Optional[str] = None
    context_selector_min_score: float = 0.2
    context_selector_min_results: int = 1


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
    query_variants: int = 1


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
        self._context_selector: Optional[TrainableContextSelector] = None
        self._context_selector_path: Optional[str] = None

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
        hybrid_strategy = self._to_hybrid_strategy(strategy)
        metrics["query_analysis_time_ms"] = (time.time() - stage_start) * 1000

        # Stage 2: Query expansion
        expanded_query = None
        query_variants: List[str] = [query]
        if config.enable_query_expansion:
            stage_start = time.time()
            expanded_query = self.query_expander.expand_query(
                query,
                strategy="comprehensive"
            )
            multi_expansions = self.query_expander.expand_multi_strategy(
                query,
            )
            query_variants = self._build_query_variants(
                query=query,
                primary_query=expanded_query.expanded_query,
                expanded_variants=[item.expanded_query for item in multi_expansions],
                max_variants=config.max_query_variants,
            )
            metrics["expansion_time_ms"] = (time.time() - stage_start) * 1000
            logger.info(f"Query expanded: {expanded_query.expanded_query}")
        else:
            query_variants = [query]

        # Stage 3: Initial retrieval
        stage_start = time.time()
        query_for_retrieval = query_variants[0]

        # Resolve embeddings per query variant.
        variant_vectors: Dict[str, Optional[List[float]]] = {}
        query_vector = self._coerce_query_vector(query_vector)
        variant_vectors[query_for_retrieval] = query_vector

        if variant_vectors[query_for_retrieval] is None and self.embedding_function:
            variant_vectors[query_for_retrieval] = self._embed_query(query_for_retrieval)

        has_embedding = (
            variant_vectors[query_for_retrieval] is not None
            and len(variant_vectors[query_for_retrieval]) > 0
        )
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

        variant_payloads: List[Dict[str, Any]] = []
        variant_metrics: List[HybridSearchMetrics] = []
        corrective_variants: List[str] = []
        corrective_applied = False
        context_pruning_stats: Dict[str, Any] = {
            "enabled": False,
            "pruned_results": 0,
            "removed_characters": 0,
            "selector": {
                "applied": False,
                "threshold": 0.0,
                "kept_results": 0,
                "dropped_results": 0,
            },
        }

        can_multi_query = (
            config.enable_multi_query_retrieval
            and len(query_variants) >= config.multi_query_min_variants
        )

        queries_to_run = query_variants if can_multi_query else [query_for_retrieval]
        for idx, variant_query in enumerate(queries_to_run):
            if variant_query not in variant_vectors:
                if self.embedding_function:
                    variant_vectors[variant_query] = self._embed_query(variant_query)
                else:
                    variant_vectors[variant_query] = variant_vectors[query_for_retrieval]

            results, search_metrics = self.hybrid_retriever.search(
                query_vector=variant_vectors[variant_query],
                query_text=variant_query,
                top_k=top_k * 2,  # Get more for fusion and refinement
                strategy=hybrid_strategy,
                filters=context.get("filters") if context else None
            )
            variant_metrics.append(search_metrics)
            variant_payloads.append(
                {
                    "query": variant_query,
                    "rank": idx,
                    "results": self._results_to_dicts(results),
                }
            )

        if config.enable_corrective_retrieval and self._should_run_corrective_pass(
            results=[item for payload in variant_payloads for item in payload["results"]],
            query_variants=queries_to_run,
            min_results=max(1, config.corrective_min_results),
            min_average_confidence=config.corrective_min_average_confidence,
            max_initial_variants=max(1, config.corrective_max_initial_variants),
        ):
            corrective_variants = self._build_corrective_query_variants(
                query=query,
                existing_variants=queries_to_run,
                max_variants=max(1, config.corrective_max_variants),
            )

            if corrective_variants:
                corrective_applied = True
                corrective_strategy = (
                    HybridSearchStrategy.HYBRID
                    if config.corrective_force_hybrid_strategy
                    else hybrid_strategy
                )
                corrective_top_k = max(
                    top_k + 2,
                    int(top_k * max(1.0, config.corrective_top_k_multiplier)),
                )

                for variant_query in corrective_variants:
                    if variant_query not in variant_vectors:
                        if self.embedding_function:
                            variant_vectors[variant_query] = self._embed_query(variant_query)
                        else:
                            variant_vectors[variant_query] = variant_vectors[query_for_retrieval]

                    results, search_metrics = self.hybrid_retriever.search(
                        query_vector=variant_vectors[variant_query],
                        query_text=variant_query,
                        top_k=corrective_top_k,
                        strategy=corrective_strategy,
                        filters=context.get("filters") if context else None,
                    )
                    variant_metrics.append(search_metrics)
                    variant_payloads.append(
                        {
                            "query": variant_query,
                            "rank": len(variant_payloads),
                            "results": self._results_to_dicts(results),
                        }
                    )

        if len(variant_payloads) > 1:
            results_dicts = self._fuse_query_variant_results(
                variant_payloads=variant_payloads,
                top_k=top_k * 2,
                rrf_k=config.multi_query_rrf_k,
            )
        else:
            results_dicts = variant_payloads[0]["results"] if variant_payloads else []

        metrics["query_variants"] = len(variant_payloads)
        metrics["retrieval_time_ms"] = (time.time() - stage_start) * 1000

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
                    refined_vector = query_vector
                    if self.embedding_function:
                        candidate_vector = self._embed_query(refined_query)
                        if candidate_vector:
                            refined_vector = candidate_vector

                    refined_results, _ = self.hybrid_retriever.search(
                        query_vector=refined_vector,
                        query_text=refined_query,
                        top_k=top_k,
                        strategy=hybrid_strategy,
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

        if config.enable_context_pruning:
            selector = self._get_context_selector(config)
            results_dicts, context_pruning_stats = self._apply_context_pruning(
                query=query,
                results=results_dicts,
                max_sentences=max(1, config.context_pruning_max_sentences_per_result),
                max_chars=max(120, config.context_pruning_max_chars_per_result),
                min_sentence_tokens=max(2, config.context_pruning_min_sentence_tokens),
                context_selector=selector,
                selector_min_score=max(0.0, config.context_selector_min_score),
                selector_min_results=max(1, config.context_selector_min_results),
            )

        search_metrics_payload = {
            "mode": "multi_query" if len(variant_payloads) > 1 else "single_query",
            "query_variants": [payload["query"] for payload in variant_payloads],
            "hybrid_metrics": [self._search_metrics_to_dict(m) for m in variant_metrics],
            "hybrid_metrics_aggregate": self._aggregate_search_metrics(variant_metrics),
            "corrective_pass": {
                "applied": corrective_applied,
                "variants": corrective_variants,
            },
            "context_pruning": context_pruning_stats,
        }

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
            avg_confidence=sum(r.get("confidence", 0.0) for r in results_dicts) / len(results_dicts) if results_dicts else 0.0,
            query_variants=metrics.get("query_variants", 1),
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
                "search_metrics": search_metrics_payload
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

    def _to_hybrid_strategy(
        self,
        strategy: RetrievalStrategy,
    ) -> HybridSearchStrategy:
        """Map router strategy enum to hybrid retriever strategy enum."""
        mapping = {
            RetrievalStrategy.VECTOR_ONLY: HybridSearchStrategy.VECTOR_ONLY,
            RetrievalStrategy.GRAPH_ONLY: HybridSearchStrategy.GRAPH_ONLY,
            RetrievalStrategy.HYBRID: HybridSearchStrategy.HYBRID,
            RetrievalStrategy.HIERARCHICAL: HybridSearchStrategy.HYBRID,
            RetrievalStrategy.MULTI_STAGE: HybridSearchStrategy.HYBRID,
        }
        return mapping.get(strategy, HybridSearchStrategy.HYBRID)

    def _coerce_query_vector(
        self,
        query_vector: Optional[Any],
    ) -> Optional[List[float]]:
        """Convert query vector to a list when provided."""
        if query_vector is None:
            return None
        if isinstance(query_vector, list):
            return query_vector
        try:
            return list(query_vector)
        except TypeError:
            return None

    def _embed_query(self, query: str) -> Optional[List[float]]:
        """Generate a query embedding using the configured embedding function."""
        if not self.embedding_function:
            return None
        try:
            embedding = self.embedding_function(query)
        except Exception as embed_err:
            logger.warning(f"Failed to generate embedding for query variant: {embed_err}")
            return None
        return self._coerce_query_vector(embedding)

    def _build_query_variants(
        self,
        query: str,
        primary_query: str,
        expanded_variants: List[str],
        max_variants: int,
    ) -> List[str]:
        """Build de-duplicated query variants for multi query retrieval."""
        candidates = [query, primary_query]
        candidates.extend(expanded_variants)

        variants: List[str] = []
        seen = set()
        for candidate in candidates:
            normalized = (candidate or "").strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            variants.append(normalized)
            if len(variants) >= max_variants:
                break
        return variants or [query]

    def _results_to_dicts(
        self,
        results: List[HybridSearchResult],
    ) -> List[Dict[str, Any]]:
        """Convert search results to dictionary payloads."""
        return [
            {
                "content": r.content,
                "score": r.score,
                "confidence": r.confidence,
                "id": r.id,
                "metadata": r.metadata,
                "vector_score": r.vector_score,
                "graph_score": r.graph_score,
                "graph_context": r.graph_context,
                "source": r.source,
            }
            for r in results
        ]

    def _stable_result_key(self, result: Dict[str, Any]) -> str:
        """Generate a stable key for de-duplicating fused retrieval results."""
        result_id = result.get("id")
        if result_id:
            return str(result_id)
        content = (result.get("content") or "").strip().lower()
        digest = hashlib.md5(content[:500].encode("utf-8")).hexdigest()
        return digest

    def _fuse_query_variant_results(
        self,
        variant_payloads: List[Dict[str, Any]],
        top_k: int,
        rrf_k: int,
    ) -> List[Dict[str, Any]]:
        """Fuse multi query result lists with reciprocal rank fusion."""
        fused: Dict[str, Dict[str, Any]] = {}
        variant_count = max(1, len(variant_payloads))

        for payload in variant_payloads:
            query_text = payload["query"]
            results = payload["results"]
            weight = 1.0 / variant_count

            for rank, result in enumerate(results, start=1):
                key = self._stable_result_key(result)
                rrf_score = weight / (rrf_k + rank)

                if key not in fused:
                    merged = dict(result)
                    merged["score"] = 0.0
                    merged["confidence"] = float(result.get("confidence", 0.0))
                    merged["variant_hits"] = []
                    fused[key] = merged

                fused[key]["score"] += rrf_score
                fused[key]["confidence"] = max(
                    float(fused[key].get("confidence", 0.0)),
                    float(result.get("confidence", 0.0)),
                )
                fused[key]["variant_hits"].append(
                    {
                        "query": query_text,
                        "rank": rank,
                        "source_score": float(result.get("score", 0.0)),
                    }
                )

        fused_results = list(fused.values())
        fused_results.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return fused_results[:top_k]

    def _should_run_corrective_pass(
        self,
        results: List[Dict[str, Any]],
        query_variants: List[str],
        min_results: int,
        min_average_confidence: float,
        max_initial_variants: int,
    ) -> bool:
        """Decide if a corrective retrieval pass should run."""
        if len(query_variants) > max_initial_variants:
            return False

        if len(results) < min_results:
            return True

        if not results:
            return True

        confidence_values = []
        for item in results:
            if item.get("confidence") is not None:
                confidence_values.append(float(item.get("confidence", 0.0)))
            else:
                confidence_values.append(float(item.get("score", 0.0)))

        if not confidence_values:
            return True

        average_confidence = sum(confidence_values) / len(confidence_values)
        return average_confidence < min_average_confidence

    def _build_corrective_query_variants(
        self,
        query: str,
        existing_variants: List[str],
        max_variants: int,
    ) -> List[str]:
        """Build additional variants for corrective retrieval."""
        variants: List[str] = []

        try:
            aggressive = self.query_expander.expand_query(query, strategy="aggressive")
            variants.append(aggressive.expanded_query)
        except Exception as err:
            logger.warning(f"Corrective aggressive expansion failed: {err}")

        try:
            suggestions = self.query_expander.suggest_related_queries(query, num_suggestions=2)
            variants.extend(suggestions)
        except Exception as err:
            logger.warning(f"Corrective query suggestion failed: {err}")

        variants.append(f"{query} evidence source details")

        existing = {item.strip().lower() for item in existing_variants}
        deduped: List[str] = []
        seen = set(existing)
        for item in variants:
            normalized = (item or "").strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
            if len(deduped) >= max_variants:
                break
        return deduped

    def _get_context_selector(
        self,
        config: PipelineConfig,
    ) -> Optional[TrainableContextSelector]:
        """Load and cache trainable context selector model when enabled."""
        if not config.enable_trainable_context_selector:
            return None
        if not config.context_selector_model_path:
            return None

        model_path = str(config.context_selector_model_path)
        if self._context_selector is not None and self._context_selector_path == model_path:
            return self._context_selector

        try:
            selector = TrainableContextSelector.load(model_path)
            self._context_selector = selector
            self._context_selector_path = model_path
            return selector
        except Exception as err:
            logger.warning(f"Unable to load context selector model from {model_path}: {err}")
            return None

    def _apply_context_pruning(
        self,
        query: str,
        results: List[Dict[str, Any]],
        max_sentences: int,
        max_chars: int,
        min_sentence_tokens: int,
        context_selector: Optional[TrainableContextSelector] = None,
        selector_min_score: float = 0.2,
        selector_min_results: int = 1,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Apply sentence-level context pruning to retrieval results."""
        if not results:
            return results, {"enabled": True, "pruned_results": 0, "removed_characters": 0}

        selector_stats = {
            "applied": False,
            "threshold": selector_min_score,
            "kept_results": len(results),
            "dropped_results": 0,
        }
        results_for_pruning = results
        if context_selector is not None:
            selector_stats["applied"] = True
            selected_results = context_selector.select(
                query=query,
                results=results,
                top_k=len(results),
                min_score=selector_min_score,
                min_selected=selector_min_results,
            )
            selector_stats["kept_results"] = len(selected_results)
            selector_stats["dropped_results"] = max(0, len(results) - len(selected_results))
            results_for_pruning = selected_results

        query_terms = self._tokenize_for_pruning(query)
        updated_results: List[Dict[str, Any]] = []
        pruned_count = 0
        removed_characters = 0

        for result in results_for_pruning:
            content = (result.get("content") or "").strip()
            if not content:
                updated_results.append(result)
                continue

            pruned_content = self._prune_content(
                content=content,
                query_terms=query_terms,
                max_sentences=max_sentences,
                max_chars=max_chars,
                min_sentence_tokens=min_sentence_tokens,
            )

            if pruned_content != content:
                pruned_count += 1
                removed_characters += max(0, len(content) - len(pruned_content))
                updated = dict(result)
                metadata = dict(updated.get("metadata") or {})
                metadata["context_pruned"] = True
                metadata["original_char_count"] = len(content)
                metadata["pruned_char_count"] = len(pruned_content)
                updated["content"] = pruned_content
                updated["metadata"] = metadata
                updated_results.append(updated)
            else:
                updated_results.append(result)

        return updated_results, {
            "enabled": True,
            "pruned_results": pruned_count,
            "removed_characters": removed_characters,
            "selector": selector_stats,
        }

    def _prune_content(
        self,
        content: str,
        query_terms: List[str],
        max_sentences: int,
        max_chars: int,
        min_sentence_tokens: int,
    ) -> str:
        """Keep high-overlap sentences under configured limits."""
        if len(content) <= max_chars:
            return content

        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", content) if item.strip()]
        if not sentences:
            return content[:max_chars]

        query_term_set = set(query_terms)
        scored: List[Tuple[float, int, str]] = []
        for idx, sentence in enumerate(sentences):
            sentence_terms = self._tokenize_for_pruning(sentence)
            if len(sentence_terms) < min_sentence_tokens and idx > 0:
                continue

            sentence_term_set = set(sentence_terms)
            overlap = len(query_term_set & sentence_term_set)
            overlap_ratio = overlap / max(1, len(query_term_set))
            density = overlap / max(1, len(sentence_term_set))
            position_bias = 1.0 / (idx + 1)
            score = (0.6 * overlap_ratio) + (0.3 * density) + (0.1 * position_bias)
            scored.append((score, idx, sentence))

        if not scored:
            pruned = " ".join(sentences[:max_sentences]).strip()
        else:
            scored.sort(key=lambda item: item[0], reverse=True)
            selected = sorted(scored[:max_sentences], key=lambda item: item[1])
            pruned = " ".join(sentence for _, _, sentence in selected).strip()

        if len(pruned) <= max_chars:
            return pruned

        shortened = pruned[:max_chars].rstrip()
        if " " in shortened:
            shortened = shortened.rsplit(" ", 1)[0]
        return f"{shortened}..."

    def _tokenize_for_pruning(
        self,
        text: str,
    ) -> List[str]:
        """Tokenize text for sentence overlap scoring."""
        stopwords = {
            "the", "a", "an", "and", "or", "but", "if", "then", "else", "in",
            "on", "at", "to", "for", "of", "with", "by", "from", "as", "is",
            "are", "was", "were", "be", "been", "being", "do", "does", "did",
            "what", "who", "where", "when", "why", "how", "which", "that",
            "this", "these", "those", "it", "its", "their", "there", "can",
            "could", "would", "should", "may", "might", "will", "about",
        }
        tokens = re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())
        return [token for token in tokens if token not in stopwords]

    def _search_metrics_to_dict(self, metrics: HybridSearchMetrics) -> Dict[str, Any]:
        """Convert search metrics dataclass to a serializable dictionary."""
        return {
            "query_type": metrics.query_type.value,
            "strategy": metrics.strategy.value,
            "vector_results": metrics.vector_results,
            "graph_results": metrics.graph_results,
            "lexical_results": metrics.lexical_results,
            "combined_results": metrics.combined_results,
            "vector_time_ms": metrics.vector_time_ms,
            "graph_time_ms": metrics.graph_time_ms,
            "lexical_time_ms": metrics.lexical_time_ms,
            "fusion_time_ms": metrics.fusion_time_ms,
            "total_time_ms": metrics.total_time_ms,
            "cache_hit": metrics.cache_hit,
            "rerank_applied": metrics.rerank_applied,
            "rerank_time_ms": metrics.rerank_time_ms,
            "services": metrics.services,
        }

    def _aggregate_search_metrics(
        self,
        metrics_list: List[HybridSearchMetrics],
    ) -> Dict[str, Any]:
        """Aggregate hybrid search metrics across query variants."""
        if not metrics_list:
            return {}

        return {
            "variant_count": len(metrics_list),
            "vector_results": sum(m.vector_results for m in metrics_list),
            "graph_results": sum(m.graph_results for m in metrics_list),
            "lexical_results": sum(m.lexical_results for m in metrics_list),
            "combined_results": max(m.combined_results for m in metrics_list),
            "vector_time_ms": sum(m.vector_time_ms for m in metrics_list),
            "graph_time_ms": sum(m.graph_time_ms for m in metrics_list),
            "lexical_time_ms": sum(m.lexical_time_ms for m in metrics_list),
            "fusion_time_ms": sum(m.fusion_time_ms for m in metrics_list),
            "total_time_ms": sum(m.total_time_ms for m in metrics_list),
            "cache_hit_count": sum(1 for m in metrics_list if m.cache_hit),
            "rerank_applied_count": sum(1 for m in metrics_list if m.rerank_applied),
            "rerank_time_ms": sum(m.rerank_time_ms for m in metrics_list),
        }

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
            avg_confidence=sum(r.get("confidence", 0.0) for r in cached_results) / len(cached_results) if cached_results else 0.0,
            query_variants=1,
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
