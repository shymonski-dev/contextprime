"""Retrieval and agentic service layer for the demo API."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, replace
from threading import Lock
from time import perf_counter
from typing import Dict, Optional, Tuple, Any, List

from loguru import logger

from contextprime.core.config import get_settings
from contextprime.core.safety_guard import (
    PromptInjectionGuard,
    RuntimeComplianceGate,
)
from contextprime.embeddings import OpenAIEmbeddingModel
from contextprime.retrieval.hybrid_retriever import (
    HybridRetriever,
    SearchStrategy as HybridStrategy,
    HybridSearchResult,
    SearchMetrics as HybridMetrics,
)
from contextprime.retrieval.query_expansion import QueryExpander
from contextprime.retrieval.rerankers import MonoT5Reranker
from contextprime.retrieval.context_selector import TrainableContextSelector
from contextprime.agents.agentic_pipeline import AgenticPipeline, AgenticMode, AgenticResult

from ..models import AdvancedQueryRequest, AgenticQueryRequest


@dataclass(frozen=True)
class RequestSearchBudget:
    """Hard limits applied to each retrieval request."""

    max_top_k: int
    max_query_variants: int
    max_corrective_variants: int
    max_total_variant_searches: int
    max_search_time_ms: int


class RetrievalService:
    """High level retrieval facade shared by the API endpoints."""

    def __init__(self) -> None:
        self._retriever_cache: Dict[Tuple[float, float, bool], HybridRetriever] = {}
        self._embedder: Optional[OpenAIEmbeddingModel] = None
        self._query_expander: Optional[QueryExpander] = None
        self._agentic_pipeline: Optional[AgenticPipeline] = None
        self._context_selector: Optional[TrainableContextSelector] = None
        self._context_selector_path: Optional[str] = None
        self._input_guard = PromptInjectionGuard()
        self._runtime_compliance_gate = RuntimeComplianceGate()
        self._lock = Lock()
        self._agentic_lock = Lock()

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        request: AdvancedQueryRequest,
    ) -> Tuple[List[HybridSearchResult], HybridMetrics, bool]:
        """Execute a hybrid search according to the request parameters."""
        guard_decision = self._input_guard.enforce_query(request.query, strict=False)

        hybrid_cfg = self._get_hybrid_feature_config()
        corrective_cfg = hybrid_cfg.get("corrective", {})
        pruning_cfg = hybrid_cfg.get("context_pruning", {})
        request_budget = self._resolve_request_budget(hybrid_cfg)

        retriever = self._get_retriever(
            vector_weight=request.vector_weight,
            graph_weight=request.graph_weight,
            use_reranking=request.use_reranking,
        )
        embedder = self._get_embedder()
        strategy = HybridStrategy(request.strategy.value)
        effective_top_k = max(1, min(int(request.top_k), request_budget.max_top_k))
        expansion_variant_limit = min(
            self._query_expansion_max_variants(),
            request_budget.max_query_variants,
        )
        enable_query_expansion = bool(
            request.use_query_expansion
            and expansion_variant_limit > 1
            and self._should_expand_query(request.query)
        )
        query_variants = (
            self._build_query_variants(request.query, max_variants=expansion_variant_limit)
            if enable_query_expansion
            else [request.query]
        )
        if len(query_variants) > request_budget.max_query_variants:
            query_variants = query_variants[: request_budget.max_query_variants]

        search_deadline = None
        if request_budget.max_search_time_ms > 0:
            search_deadline = perf_counter() + (request_budget.max_search_time_ms / 1000.0)
        query_vectors = embedder.encode(query_variants)

        start = perf_counter()
        variant_outputs = self._run_variant_searches(
            retriever=retriever,
            strategy=strategy,
            query_variants=query_variants,
            query_vectors=query_vectors,
            top_k=effective_top_k,
            filters=request.filters,
            graph_policy=request.graph_policy,
            max_variant_searches=request_budget.max_total_variant_searches,
            deadline=search_deadline,
        )
        if not variant_outputs:
            raise RuntimeError("No query variants were executed for retrieval")

        all_outputs = list(variant_outputs)
        if len(variant_outputs) > 1:
            results = self._fuse_variant_results(variant_outputs, top_k=effective_top_k)
            metrics = self._aggregate_variant_metrics(variant_outputs, final_count=len(results))
        else:
            _, results, metrics = variant_outputs[0]

        corrective_applied = False
        remaining_searches = max(0, request_budget.max_total_variant_searches - len(variant_outputs))
        if remaining_searches > 0 and self._should_run_corrective_pass(
            results=results,
            query_variants=query_variants,
            request=request,
            corrective_cfg=corrective_cfg,
        ):
            configured_corrective_variants = max(1, int(corrective_cfg.get("max_variants", 2)))
            corrective_variant_limit = min(
                configured_corrective_variants,
                request_budget.max_corrective_variants,
                remaining_searches,
            )
            corrective_variants = self._build_corrective_query_variants(
                query=request.query,
                existing_variants=query_variants,
                max_variants=corrective_variant_limit,
            )

            if corrective_variants:
                corrective_applied = True
                corrective_top_k = max(
                    effective_top_k + 2,
                    int(effective_top_k * float(corrective_cfg.get("top_k_multiplier", 2.0))),
                )
                corrective_top_k = max(
                    effective_top_k,
                    min(corrective_top_k, request_budget.max_top_k),
                )
                corrective_strategy = strategy
                if bool(corrective_cfg.get("force_hybrid", True)):
                    corrective_strategy = HybridStrategy.HYBRID

                corrective_vectors = embedder.encode(corrective_variants)
                corrective_outputs = self._run_variant_searches(
                    retriever=retriever,
                    strategy=corrective_strategy,
                    query_variants=corrective_variants,
                    query_vectors=corrective_vectors,
                    top_k=corrective_top_k,
                    filters=request.filters,
                    graph_policy=request.graph_policy,
                    max_variant_searches=remaining_searches,
                    deadline=search_deadline,
                )

                if corrective_outputs:
                    all_outputs.extend(corrective_outputs)
                    results = self._fuse_variant_results(all_outputs, top_k=effective_top_k)
                    metrics = self._aggregate_variant_metrics(all_outputs, final_count=len(results))

        results, pruning_stats = self._apply_context_pruning(
            query=request.query,
            results=results,
            pruning_cfg=pruning_cfg,
        )

        metrics.services = dict(metrics.services or {})
        metrics.services["query_expansion_requested"] = bool(request.use_query_expansion)
        metrics.services["query_expansion"] = enable_query_expansion
        if request.use_query_expansion and not enable_query_expansion:
            if expansion_variant_limit <= 1:
                metrics.services["query_expansion_skipped"] = "variant_budget_limit"
            else:
                metrics.services["query_expansion_skipped"] = "query_too_short"
        metrics.services["query_variants"] = len(query_variants)
        metrics.services["variant_searches_executed"] = len(all_outputs)
        metrics.services["corrective_pass"] = corrective_applied
        metrics.services["context_pruning"] = pruning_stats["enabled"]
        metrics.services["context_pruned_results"] = pruning_stats["pruned_results"]
        metrics.services["context_pruned_chars"] = pruning_stats["removed_characters"]
        metrics.services["request_top_k"] = int(request.top_k)
        metrics.services["effective_top_k"] = int(effective_top_k)
        metrics.services["top_k_clamped"] = bool(request.top_k > effective_top_k)
        metrics.services["request_budget"] = {
            "max_top_k": request_budget.max_top_k,
            "max_query_variants": request_budget.max_query_variants,
            "max_corrective_variants": request_budget.max_corrective_variants,
            "max_total_variant_searches": request_budget.max_total_variant_searches,
            "max_search_time_ms": request_budget.max_search_time_ms,
        }
        metrics.services["budget_exhausted"] = bool(
            len(all_outputs) >= request_budget.max_total_variant_searches
            or (search_deadline is not None and perf_counter() >= search_deadline)
        )
        metrics.services["input_guard_risk_score"] = guard_decision.risk_score
        if guard_decision.flags:
            metrics.services["input_guard_flags"] = guard_decision.flags

        metrics.total_time_ms = (perf_counter() - start) * 1000
        rerank_applied = retriever.reranker is not None

        if not request.include_graph_context:
            for item in results:
                item.graph_context = None

        return results, metrics, rerank_applied

    # ------------------------------------------------------------------
    # Agentic query
    # ------------------------------------------------------------------

    async def agentic_query(self, request: AgenticQueryRequest) -> AgenticResult:
        """Process a query through the agentic pipeline."""
        guard_decision = self._input_guard.enforce_query(request.query, strict=True)

        pipeline = self._get_agentic_pipeline()
        result = await pipeline.process_query(
            query=guard_decision.normalized_text,
            context=None,
            max_iterations=request.max_iterations,
            min_quality_threshold=0.7 if request.use_evaluation else 0.0,
        )
        compliance_decision = self._runtime_compliance_gate.enforce_before_response(
            query=guard_decision.normalized_text,
            answer=result.answer,
        )
        metadata = dict(result.metadata or {})
        metadata["input_guard"] = {
            "risk_score": guard_decision.risk_score,
            "flags": guard_decision.flags,
        }
        metadata["runtime_compliance"] = {
            "enforced": compliance_decision.enforced,
            "recorded": compliance_decision.recorded,
            "high_risk_topics": compliance_decision.high_risk_topics,
            "oversight_file": compliance_decision.oversight_file,
        }
        result.metadata = metadata
        return result

    def get_dependency_health(self) -> Dict[str, bool]:
        """Return health status for retrieval dependencies."""
        health = {"qdrant": False, "neo4j": False}

        try:
            retriever = self._get_retriever(
                vector_weight=0.7,
                graph_weight=0.3,
                use_reranking=False,
            )
        except Exception as err:
            logger.warning("Unable to initialize retriever for health check: %s", err)
            return health

        try:
            qdrant = retriever._ensure_qdrant()
            if qdrant is not None:
                health["qdrant"] = bool(qdrant.health_check())
        except Exception as err:
            logger.warning("Qdrant health probe failed: %s", err)

        try:
            neo4j = retriever._ensure_neo4j()
            if neo4j is not None:
                health["neo4j"] = bool(neo4j.health_check())
        except Exception as err:
            logger.warning("Neo4j health probe failed: %s", err)

        return health

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _normalise_weights(self, vector_weight: float, graph_weight: float) -> Tuple[float, float]:
        total = vector_weight + graph_weight
        if total <= 0:
            return 0.7, 0.3
        return vector_weight / total, graph_weight / total

    def _get_retriever(
        self,
        vector_weight: float,
        graph_weight: float,
        use_reranking: bool,
    ) -> HybridRetriever:
        vector_weight, graph_weight = self._normalise_weights(vector_weight, graph_weight)
        key = (round(vector_weight, 3), round(graph_weight, 3), use_reranking)

        with self._lock:
            retriever = self._retriever_cache.get(key)
            if retriever is None:
                retriever = HybridRetriever(
                    vector_weight=vector_weight,
                    graph_weight=graph_weight,
                )
                if use_reranking:
                    self._ensure_reranker(retriever)
                else:
                    retriever.reranker = None
                self._retriever_cache[key] = retriever
        return retriever

    def _ensure_reranker(self, retriever: HybridRetriever) -> None:
        if retriever.reranker is not None:
            return

        settings = get_settings()
        rerank_cfg = getattr(settings.retrieval, "rerank_settings", {}) or {}
        model_name = rerank_cfg.get("model_name", "castorini/monot5-base-msmarco-10k")
        device = rerank_cfg.get("device")
        try:
            retriever.reranker = MonoT5Reranker(
                model_name=model_name,
                device=device,
                cache_dir=settings.paths.models_dir,
            )
            logger.info("MonoT5 reranker initialised for API calls: %s", model_name)
        except Exception as err:  # pragma: no cover - optional dependency failure
            logger.warning("Unable to initialise MonoT5 reranker: %s", err)
            retriever.reranker = None

    def _get_embedder(self) -> OpenAIEmbeddingModel:
        with self._lock:
            if self._embedder is None:
                self._embedder = OpenAIEmbeddingModel()
        return self._embedder

    def _get_query_expander(self) -> QueryExpander:
        with self._lock:
            if self._query_expander is None:
                self._query_expander = QueryExpander(
                    enable_wordnet=True,
                    enable_semantic=False,
                    enable_contextual=True,
                )
        return self._query_expander

    def _build_query_variants(self, query: str, max_variants: Optional[int] = None) -> List[str]:
        """Build de-duplicated query variants from expansion strategies."""
        if max_variants is None:
            max_variants = self._query_expansion_max_variants()
        max_variants = max(1, int(max_variants))

        expander = self._get_query_expander()
        variants = [query]

        try:
            expanded = expander.expand_multi_strategy(query)
            variants.extend(item.expanded_query for item in expanded)
        except Exception as err:
            logger.warning("Query expansion failed; continuing with original query: %s", err)

        deduped: List[str] = []
        seen = set()
        for item in variants:
            normalized = (item or "").strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(normalized)
            if len(deduped) >= max_variants:
                break

        return deduped or [query]

    def _query_expansion_max_variants(self) -> int:
        raw = os.getenv("DOCTAGS_QUERY_EXPANSION_MAX_VARIANTS", "2")
        try:
            return max(1, min(5, int(raw)))
        except ValueError:
            return 2

    def _should_expand_query(self, query: str) -> bool:
        min_tokens_raw = os.getenv("DOCTAGS_QUERY_EXPANSION_MIN_TOKENS", "4")
        min_chars_raw = os.getenv("DOCTAGS_QUERY_EXPANSION_MIN_CHARS", "24")
        try:
            min_tokens = max(1, int(min_tokens_raw))
        except ValueError:
            min_tokens = 4
        try:
            min_chars = max(8, int(min_chars_raw))
        except ValueError:
            min_chars = 24

        normalized = (query or "").strip()
        if len(normalized) >= min_chars:
            return True
        token_count = len(re.findall(r"\w+", normalized))
        return token_count >= min_tokens

    def _run_variant_searches(
        self,
        retriever: HybridRetriever,
        strategy: HybridStrategy,
        query_variants: List[str],
        query_vectors: List[List[float]],
        top_k: int,
        filters: Optional[Dict[str, Any]],
        graph_policy: Optional[str],
        max_variant_searches: Optional[int] = None,
        deadline: Optional[float] = None,
    ) -> List[Tuple[str, List[HybridSearchResult], HybridMetrics]]:
        """Run retrieval for each query variant."""
        variant_outputs: List[Tuple[str, List[HybridSearchResult], HybridMetrics]] = []
        for variant_query, query_vector in zip(query_variants, query_vectors):
            if (
                max_variant_searches is not None
                and len(variant_outputs) >= max(1, int(max_variant_searches))
            ):
                break
            if deadline is not None and perf_counter() >= deadline and variant_outputs:
                break

            try:
                results, metrics = retriever.search(
                    query_vector=query_vector,
                    query_text=variant_query,
                    top_k=top_k,
                    strategy=strategy,
                    graph_policy=graph_policy,
                    filters=filters,
                )
            except TypeError:
                # Backward compatibility with test doubles that use a shorter signature.
                results, metrics = retriever.search(
                    query_vector=query_vector,
                    query_text=variant_query,
                    top_k=top_k,
                    strategy=strategy,
                    filters=filters,
                )
            variant_outputs.append((variant_query, results, metrics))
        return variant_outputs

    def _get_hybrid_feature_config(self) -> Dict[str, Any]:
        """Load hybrid retrieval feature configuration."""
        settings = get_settings()
        hybrid_cfg = getattr(settings.retrieval, "hybrid_search", {}) or {}
        return hybrid_cfg if isinstance(hybrid_cfg, dict) else {}

    def _resolve_request_budget(self, hybrid_cfg: Dict[str, Any]) -> RequestSearchBudget:
        """Resolve strict per-request search budget from configuration."""
        budget_cfg = hybrid_cfg.get("request_budget", {}) if isinstance(hybrid_cfg, dict) else {}
        if not isinstance(budget_cfg, dict):
            budget_cfg = {}

        max_top_k = max(1, int(budget_cfg.get("max_top_k", 12)))
        max_query_variants = max(1, int(budget_cfg.get("max_query_variants", 3)))
        max_corrective_variants = max(1, int(budget_cfg.get("max_corrective_variants", 2)))
        max_total_variant_searches = max(
            1,
            int(budget_cfg.get("max_total_variant_searches", 5)),
        )
        max_search_time_ms = max(250, int(budget_cfg.get("max_search_time_ms", 4500)))

        return RequestSearchBudget(
            max_top_k=max_top_k,
            max_query_variants=max_query_variants,
            max_corrective_variants=max_corrective_variants,
            max_total_variant_searches=max_total_variant_searches,
            max_search_time_ms=max_search_time_ms,
        )

    def _should_run_corrective_pass(
        self,
        results: List[HybridSearchResult],
        query_variants: List[str],
        request: AdvancedQueryRequest,
        corrective_cfg: Dict[str, Any],
    ) -> bool:
        """Decide if corrective retrieval should run."""
        if not bool(corrective_cfg.get("enable", False)):
            return False

        max_initial_variants = max(1, int(corrective_cfg.get("max_initial_variants", 3)))
        if len(query_variants) > max_initial_variants:
            return False

        min_results = max(1, int(corrective_cfg.get("min_results", request.top_k)))
        min_confidence = float(corrective_cfg.get("min_average_confidence", 0.55))

        if len(results) < min_results:
            return True

        if not results:
            return True

        avg_confidence = sum(float(item.confidence) for item in results) / len(results)
        return avg_confidence < min_confidence

    def _build_corrective_query_variants(
        self,
        query: str,
        existing_variants: List[str],
        max_variants: int = 2,
    ) -> List[str]:
        """Build additional variants for corrective retrieval."""
        variants: List[str] = []
        expander = self._get_query_expander()

        try:
            aggressive = expander.expand_query(query, strategy="aggressive")
            variants.append(aggressive.expanded_query)
        except Exception as err:
            logger.warning("Corrective aggressive expansion failed: %s", err)

        try:
            suggestions = expander.suggest_related_queries(query, num_suggestions=2)
            variants.extend(suggestions)
        except Exception as err:
            logger.warning("Corrective query suggestion failed: %s", err)

        # A fixed evidence-oriented variant improves recovery for vague queries.
        variants.append(f"{query} evidence source details")

        existing = {item.strip().lower() for item in existing_variants}
        deduped: List[str] = []
        seen = set(existing)
        for item in variants:
            normalized = (item or "").strip()
            if not normalized:
                continue
            lowered = normalized.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(normalized)
            if len(deduped) >= max_variants:
                break
        return deduped

    def _get_context_selector(
        self,
        pruning_cfg: Dict[str, Any],
    ) -> Optional[TrainableContextSelector]:
        """Load and cache the trainable context selector when configured."""
        selector_cfg = pruning_cfg.get("context_selector", {}) if isinstance(pruning_cfg, dict) else {}
        if not bool(selector_cfg.get("enable", False)):
            return None

        model_path = str(selector_cfg.get("model_path", "")).strip()
        if not model_path:
            return None

        if self._context_selector is not None and self._context_selector_path == model_path:
            return self._context_selector

        try:
            selector = TrainableContextSelector.load(model_path)
            self._context_selector = selector
            self._context_selector_path = model_path
            return selector
        except Exception as err:
            logger.warning("Unable to load context selector model from %s: %s", model_path, err)
            return None

    def _fuse_variant_results(
        self,
        variant_outputs: List[Tuple[str, List[HybridSearchResult], HybridMetrics]],
        top_k: int,
        rrf_k: int = 60,
    ) -> List[HybridSearchResult]:
        """Fuse multiple query result sets using reciprocal rank fusion."""
        fused: Dict[str, HybridSearchResult] = {}
        scores: Dict[str, float] = {}
        hits: Dict[str, List[Dict[str, Any]]] = {}
        variant_weight = 1.0 / max(1, len(variant_outputs))

        for query_text, results, _ in variant_outputs:
            for rank, result in enumerate(results, start=1):
                key = str(result.id) if result.id else hashlib.md5(
                    result.content[:500].lower().encode("utf-8")
                ).hexdigest()
                if key not in fused:
                    fused[key] = replace(result)
                    scores[key] = 0.0
                    hits[key] = []

                scores[key] += variant_weight / (rrf_k + rank)
                fused[key].confidence = max(fused[key].confidence, result.confidence)
                hits[key].append(
                    {
                        "query": query_text,
                        "rank": rank,
                        "score": result.score,
                    }
                )

        for key, result in fused.items():
            result.score = scores[key]
            result.metadata = dict(result.metadata or {})
            result.metadata["query_variant_hits"] = hits[key]

        ordered = sorted(fused.values(), key=lambda item: item.score, reverse=True)
        return ordered[:top_k]

    def _apply_context_pruning(
        self,
        query: str,
        results: List[HybridSearchResult],
        pruning_cfg: Dict[str, Any],
    ) -> Tuple[List[HybridSearchResult], Dict[str, Any]]:
        """Apply sentence-level context pruning to reduce noise."""
        if not bool(pruning_cfg.get("enable", False)):
            return results, {"enabled": False, "pruned_results": 0, "removed_characters": 0}

        max_sentences = max(1, int(pruning_cfg.get("max_sentences_per_result", 4)))
        max_chars = max(120, int(pruning_cfg.get("max_chars_per_result", 900)))
        min_sentence_tokens = max(2, int(pruning_cfg.get("min_sentence_tokens", 3)))
        context_selector = self._get_context_selector(pruning_cfg)
        selector_cfg = pruning_cfg.get("context_selector", {}) if isinstance(pruning_cfg, dict) else {}
        selector_min_score = max(0.0, float(selector_cfg.get("min_score", 0.2)))
        selector_min_results = max(1, int(selector_cfg.get("min_results", 1)))

        selector_stats = {
            "applied": False,
            "threshold": selector_min_score,
            "kept_results": len(results),
            "dropped_results": 0,
        }
        results_for_pruning = results
        if context_selector is not None and results:
            selector_stats["applied"] = True
            candidate_dicts = []
            for result in results:
                candidate_dicts.append(
                    {
                        "id": result.id,
                        "content": result.content,
                        "score": result.score,
                        "confidence": result.confidence,
                        "source": result.source,
                        "vector_score": result.vector_score,
                        "graph_score": result.graph_score,
                        "lexical_score": result.lexical_score,
                        "metadata": dict(result.metadata or {}),
                        "graph_context": result.graph_context,
                    }
                )
            selected_dicts = context_selector.select(
                query=query,
                results=candidate_dicts,
                top_k=len(candidate_dicts),
                min_score=selector_min_score,
                min_selected=selector_min_results,
            )
            selector_stats["kept_results"] = len(selected_dicts)
            selector_stats["dropped_results"] = max(0, len(results) - len(selected_dicts))
            selected_map = {str(item.get("id")): item for item in selected_dicts}
            filtered_results: List[HybridSearchResult] = []
            for result in results:
                key = str(result.id)
                if key not in selected_map:
                    continue
                selected_payload = selected_map[key]
                merged_metadata = dict(result.metadata or {})
                merged_metadata.update(dict(selected_payload.get("metadata") or {}))
                filtered_results.append(replace(result, metadata=merged_metadata))
            if filtered_results:
                results_for_pruning = filtered_results

        query_terms = self._tokenize_for_pruning(query)

        updated: List[HybridSearchResult] = []
        pruned_count = 0
        removed_characters = 0

        for result in results_for_pruning:
            content = (result.content or "").strip()
            if not content:
                updated.append(result)
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
                metadata = dict(result.metadata or {})
                metadata["context_pruned"] = True
                metadata["original_char_count"] = len(content)
                metadata["pruned_char_count"] = len(pruned_content)
                updated.append(replace(result, content=pruned_content, metadata=metadata))
            else:
                updated.append(result)

        stats = {
            "enabled": True,
            "pruned_results": pruned_count,
            "removed_characters": removed_characters,
            "selector": selector_stats,
        }
        return updated, stats

    def _prune_content(
        self,
        content: str,
        query_terms: List[str],
        max_sentences: int,
        max_chars: int,
        min_sentence_tokens: int,
    ) -> str:
        """Keep most relevant sentences under configured limits."""
        if len(content) <= max_chars:
            return content

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()]
        if len(sentences) <= max_sentences and len(content) <= max_chars:
            return content

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
            top_sentences = sorted(scored[:max_sentences], key=lambda item: item[1])
            pruned = " ".join(sentence for _, _, sentence in top_sentences).strip()

        if len(pruned) <= max_chars:
            return pruned

        shortened = pruned[:max_chars].rstrip()
        if " " in shortened:
            shortened = shortened.rsplit(" ", 1)[0]
        return f"{shortened}..."

    def _tokenize_for_pruning(self, text: str) -> List[str]:
        """Tokenize text for query to sentence overlap scoring."""
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

    def _aggregate_variant_metrics(
        self,
        variant_outputs: List[Tuple[str, List[HybridSearchResult], HybridMetrics]],
        final_count: int,
    ) -> HybridMetrics:
        """Aggregate metrics across query variants."""
        metrics = [item[2] for item in variant_outputs]
        base = replace(metrics[0])
        base.vector_results = sum(m.vector_results for m in metrics)
        base.graph_results = sum(m.graph_results for m in metrics)
        base.lexical_results = sum(m.lexical_results for m in metrics)
        base.combined_results = final_count
        base.vector_time_ms = sum(m.vector_time_ms for m in metrics)
        base.graph_time_ms = sum(m.graph_time_ms for m in metrics)
        base.lexical_time_ms = sum(m.lexical_time_ms for m in metrics)
        base.fusion_time_ms = sum(m.fusion_time_ms for m in metrics)
        base.rerank_time_ms = sum(m.rerank_time_ms for m in metrics)
        base.cache_hit = any(m.cache_hit for m in metrics)
        base.rerank_applied = any(m.rerank_applied for m in metrics)
        return base

    def _get_agentic_pipeline(self) -> AgenticPipeline:
        with self._agentic_lock:
            if self._agentic_pipeline is None:
                self._agentic_pipeline = AgenticPipeline(mode=AgenticMode.FAST)
        return self._agentic_pipeline
