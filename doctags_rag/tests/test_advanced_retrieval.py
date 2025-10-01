"""
Comprehensive tests for advanced retrieval features.

Tests all components:
- Confidence scoring
- Query routing
- Iterative refinement
- Reranking
- Query expansion
- Caching
- Advanced pipeline
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.retrieval.confidence_scorer import (
    ConfidenceScorer, ConfidenceLevel, CorrectiveAction
)
from src.retrieval.query_router import (
    QueryRouter, QueryType, QueryComplexity, RetrievalStrategy
)
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.reranker import Reranker
from src.retrieval.cache_manager import CacheManager, LRUCache, SemanticQueryCache
from src.retrieval.iterative_refiner import IterativeRefiner


# Test Data
SAMPLE_QUERY = "What is machine learning?"
SAMPLE_RESULTS = [
    {
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "score": 0.85,
        "metadata": {"source": "textbook", "author": "John Doe"}
    },
    {
        "content": "Deep learning is a type of machine learning using neural networks.",
        "score": 0.75,
        "metadata": {"source": "article"}
    },
    {
        "content": "Natural language processing uses machine learning for text analysis.",
        "score": 0.65,
        "metadata": {}
    }
]


class TestConfidenceScorer:
    """Test confidence scoring system."""

    def test_initialization(self):
        """Test scorer initialization."""
        scorer = ConfidenceScorer()
        assert scorer is not None
        assert scorer.thresholds is not None

    def test_score_result(self):
        """Test scoring a single result."""
        scorer = ConfidenceScorer()
        result = SAMPLE_RESULTS[0]

        score = scorer.score_result(
            query=SAMPLE_QUERY,
            result_content=result["content"],
            vector_score=result["score"],
            metadata=result["metadata"]
        )

        assert score is not None
        assert 0.0 <= score.overall_score <= 1.0
        assert score.level in [ConfidenceLevel.CORRECT, ConfidenceLevel.AMBIGUOUS, ConfidenceLevel.INCORRECT]
        assert score.corrective_action in CorrectiveAction
        assert score.reasoning

    def test_score_batch(self):
        """Test batch scoring."""
        scorer = ConfidenceScorer()

        scores = scorer.score_results_batch(SAMPLE_QUERY, SAMPLE_RESULTS)

        assert len(scores) == len(SAMPLE_RESULTS)
        for score in scores:
            assert 0.0 <= score.overall_score <= 1.0

    def test_confidence_levels(self):
        """Test confidence level determination."""
        scorer = ConfidenceScorer()

        # High confidence result
        high_conf_result = {
            "content": SAMPLE_QUERY + " " + SAMPLE_RESULTS[0]["content"],
            "score": 0.95,
            "metadata": {"verified": True, "author": "Expert"}
        }

        score = scorer.score_result(
            query=SAMPLE_QUERY,
            result_content=high_conf_result["content"],
            vector_score=high_conf_result["score"],
            metadata=high_conf_result["metadata"]
        )

        assert score.level == ConfidenceLevel.CORRECT

        # Low confidence result
        low_conf_result = {
            "content": "Unrelated content about cooking recipes.",
            "score": 0.3,
            "metadata": {}
        }

        score = scorer.score_result(
            query=SAMPLE_QUERY,
            result_content=low_conf_result["content"],
            vector_score=low_conf_result["score"]
        )

        assert score.level in [ConfidenceLevel.AMBIGUOUS, ConfidenceLevel.INCORRECT]

    def test_aggregate_confidence(self):
        """Test confidence aggregation."""
        scorer = ConfidenceScorer()
        scores = scorer.score_results_batch(SAMPLE_QUERY, SAMPLE_RESULTS)

        aggregated = scorer.aggregate_confidence(scores)

        assert "average_confidence" in aggregated
        assert "confidence_distribution" in aggregated
        assert "recommended_action" in aggregated
        assert 0.0 <= aggregated["average_confidence"] <= 1.0


class TestQueryRouter:
    """Test query routing system."""

    def test_initialization(self):
        """Test router initialization."""
        router = QueryRouter()
        assert router is not None

    def test_query_analysis(self):
        """Test query analysis."""
        router = QueryRouter()

        # Factual query
        analysis = router.analyze_query("What is Python?")
        assert analysis.query_type in QueryType
        assert analysis.complexity in QueryComplexity
        assert analysis.recommended_strategy in RetrievalStrategy

        # Relationship query
        analysis = router.analyze_query("How is Python related to Java?")
        assert analysis.query_type in [QueryType.RELATIONSHIP, QueryType.COMPARISON]

        # Complex query
        analysis = router.analyze_query(
            "Compare and contrast the architectural differences between "
            "Python and Java virtual machines"
        )
        assert analysis.complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]

    def test_route_query(self):
        """Test query routing."""
        router = QueryRouter()

        strategy, analysis = router.route_query("What is machine learning?")

        assert strategy in RetrievalStrategy
        assert analysis.query_type in QueryType

    def test_performance_recording(self):
        """Test performance recording."""
        with tempfile.TemporaryDirectory() as tmpdir:
            perf_file = Path(tmpdir) / "performance.json"

            router = QueryRouter(
                enable_learning=True,
                performance_file=perf_file
            )

            # Record some performance
            router.record_performance(
                query="test query",
                strategy=RetrievalStrategy.HYBRID,
                query_type=QueryType.FACTUAL,
                success=True,
                confidence=0.8,
                num_results=5
            )

            stats = router.get_statistics()
            assert stats["total_tracked"] > 0


class TestQueryExpander:
    """Test query expansion system."""

    def test_initialization(self):
        """Test expander initialization."""
        expander = QueryExpander()
        assert expander is not None

    def test_expand_query(self):
        """Test query expansion."""
        expander = QueryExpander()

        expanded = expander.expand_query(SAMPLE_QUERY, strategy="comprehensive")

        assert expanded is not None
        assert expanded.original_query == SAMPLE_QUERY
        assert expanded.expanded_query
        assert len(expanded.expanded_query) >= len(SAMPLE_QUERY)

    def test_expansion_strategies(self):
        """Test different expansion strategies."""
        expander = QueryExpander()

        conservative = expander.expand_query(SAMPLE_QUERY, strategy="conservative")
        comprehensive = expander.expand_query(SAMPLE_QUERY, strategy="comprehensive")
        aggressive = expander.expand_query(SAMPLE_QUERY, strategy="aggressive")

        # Aggressive should generally be longest
        assert len(aggressive.expanded_query) >= len(conservative.expanded_query)

    def test_domain_expansion(self):
        """Test domain-specific expansion."""
        expander = QueryExpander()

        # Add custom domain expansion
        expander.add_domain_expansion("ml", ["machine learning", "AI"])

        expanded = expander.expand_query("What is ml?")
        assert any(
            term in expanded.expanded_query.lower()
            for term in ["machine learning", "ai"]
        )

    def test_suggest_related_queries(self):
        """Test related query suggestions."""
        expander = QueryExpander()

        suggestions = expander.suggest_related_queries(SAMPLE_QUERY, num_suggestions=3)

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3


class TestReranker:
    """Test result reranking system."""

    def test_initialization(self):
        """Test reranker initialization."""
        reranker = Reranker(enable_cross_encoder=False)  # Disable to avoid model download
        assert reranker is not None

    def test_rerank(self):
        """Test reranking."""
        reranker = Reranker(enable_cross_encoder=False)

        reranked = reranker.rerank(
            query=SAMPLE_QUERY,
            results=SAMPLE_RESULTS,
            top_k=3,
            enable_diversity=True
        )

        assert len(reranked) <= 3
        assert all(hasattr(r, 'reranked_score') for r in reranked)

        # Check ranking is applied
        for i, result in enumerate(reranked):
            assert result.rank == i

    def test_feature_computation(self):
        """Test feature computation."""
        reranker = Reranker(enable_cross_encoder=False)

        reranked = reranker.rerank(SAMPLE_QUERY, SAMPLE_RESULTS)

        # Check features are computed
        for result in reranked:
            assert result.features is not None
            assert 0.0 <= result.features.semantic_score <= 1.0
            assert 0.0 <= result.features.length_score <= 1.0

    def test_explain_ranking(self):
        """Test ranking explanation."""
        reranker = Reranker(enable_cross_encoder=False)

        reranked = reranker.rerank(SAMPLE_QUERY, SAMPLE_RESULTS)
        explanation = reranker.explain_ranking(reranked[0])

        assert "total_score" in explanation
        assert "rank" in explanation
        assert "contributions" in explanation
        assert "top_features" in explanation


class TestCacheManager:
    """Test caching system."""

    def test_lru_cache(self):
        """Test LRU cache."""
        cache = LRUCache(max_size=2)

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"

        # Add third item, should evict first
        cache.set("key3", "value3")
        assert cache.get("key3") == "value3"
        # key1 might be evicted
        assert len(cache.cache) <= 2

    def test_semantic_query_cache(self):
        """Test semantic query cache."""
        cache = SemanticQueryCache(max_size=10)

        # Cache a query
        query_emb = np.random.rand(384)
        cache.set("test query", query_emb, ["result1", "result2"])

        # Exact match
        results = cache.get("test query", query_emb)
        assert results is not None

    def test_cache_manager(self):
        """Test cache manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"

            manager = CacheManager(
                cache_dir=cache_dir,
                enable_disk_cache=True
            )

            # Test query caching
            query_emb = np.random.rand(384)
            manager.cache_query_results(SAMPLE_QUERY, query_emb, SAMPLE_RESULTS)

            cached = manager.get_query_results(SAMPLE_QUERY, query_emb)
            assert cached is not None
            assert len(cached) == len(SAMPLE_RESULTS)

            # Test embedding caching
            text_emb = np.random.rand(384)
            manager.cache_embedding("test text", text_emb)

            retrieved_emb = manager.get_embedding("test text")
            assert retrieved_emb is not None
            assert np.allclose(retrieved_emb, text_emb)

    def test_cache_statistics(self):
        """Test cache statistics."""
        manager = CacheManager()

        # Perform some cache operations
        manager.cache_result("key1", "value1")
        manager.get_result("key1")
        manager.get_result("nonexistent")

        stats = manager.get_statistics()

        assert "result_cache" in stats
        assert "hit_rate" in stats["result_cache"]


class TestIterativeRefiner:
    """Test iterative refinement system."""

    def test_initialization(self):
        """Test refiner initialization."""
        refiner = IterativeRefiner()
        assert refiner is not None

    def test_refine_retrieval(self):
        """Test refinement process."""
        refiner = IterativeRefiner(max_iterations=2)

        # Mock retrieval function
        call_count = [0]

        def mock_retrieval(query, context=None):
            call_count[0] += 1
            # Return slightly different results each time
            return [
                {
                    "content": f"Result {i} for {query}",
                    "score": 0.7 - i * 0.1,
                    "metadata": {}
                }
                for i in range(3)
            ]

        refined_results, steps = refiner.refine_retrieval(
            original_query=SAMPLE_QUERY,
            initial_results=SAMPLE_RESULTS,
            retrieval_func=mock_retrieval
        )

        assert refined_results is not None
        assert isinstance(steps, list)

    def test_gap_identification(self):
        """Test information gap identification."""
        refiner = IterativeRefiner()

        # Results with low confidence should trigger gap identification
        low_conf_results = [
            {
                "content": "Partial information about the topic.",
                "score": 0.4,
                "confidence": 0.3,
                "metadata": {}
            }
        ]

        # Mock confidence scores
        from src.retrieval.confidence_scorer import ConfidenceScore, ConfidenceSignals, ConfidenceLevel

        conf_scores = [
            ConfidenceScore(
                overall_score=0.3,
                level=ConfidenceLevel.INCORRECT,
                signals=ConfidenceSignals(0.3, 0.2, 0.3, 0.5, 0.5, 0.5),
                corrective_action=CorrectiveAction.QUERY_REWRITE,
                reasoning="Low confidence"
            )
        ]

        needs_refinement, reason = refiner._needs_refinement(
            [refiner._process_results(low_conf_results, 0, SAMPLE_QUERY, set())[0]],
            conf_scores
        )

        assert needs_refinement


class TestIntegration:
    """Integration tests for the complete system."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from query to results."""
        # Initialize components
        router = QueryRouter()
        expander = QueryExpander()
        scorer = ConfidenceScorer()
        reranker = Reranker(enable_cross_encoder=False)

        # Process query
        query = "What is deep learning?"

        # Route query
        strategy, analysis = router.route_query(query)
        assert strategy is not None

        # Expand query
        expanded = expander.expand_query(query)
        assert expanded.expanded_query

        # Score results (mock)
        scores = scorer.score_results_batch(query, SAMPLE_RESULTS)
        assert len(scores) == len(SAMPLE_RESULTS)

        # Rerank results
        reranked = reranker.rerank(query, SAMPLE_RESULTS)
        assert len(reranked) > 0

    def test_caching_workflow(self):
        """Test workflow with caching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            manager = CacheManager(cache_dir=cache_dir)

            query_emb = np.random.rand(384)

            # First retrieval (cache miss)
            cached = manager.get_query_results(SAMPLE_QUERY, query_emb)
            assert cached is None

            # Cache results
            manager.cache_query_results(SAMPLE_QUERY, query_emb, SAMPLE_RESULTS)

            # Second retrieval (cache hit)
            cached = manager.get_query_results(SAMPLE_QUERY, query_emb)
            assert cached is not None
            assert len(cached) == len(SAMPLE_RESULTS)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
