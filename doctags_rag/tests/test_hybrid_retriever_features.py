from dataclasses import replace

from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridSearchResult,
    SearchStrategy,
)
from src.retrieval.qdrant_manager import SearchResult as QdrantResult


def test_hybrid_retriever_cache():
    retriever = HybridRetriever(neo4j_manager=None, qdrant_manager=None)
    retriever.cache_enabled = True
    retriever.cache_max_size = 8
    retriever.cache_ttl = 60
    retriever._cache.clear()
    retriever.reranker = None

    vector_calls = {"count": 0}

    def fake_search_vector(self, query_vector, top_k, filters, collection_name):
        vector_calls["count"] += 1
        return [
            QdrantResult(
                id="1",
                score=0.9,
                vector=None,
                metadata={"text": "doc-1"},
            )
        ]

    def fake_fusion(self, vector_results, graph_results, top_k):
        return self._convert_vector_results(vector_results)

    retriever._search_vector = fake_search_vector.__get__(retriever, HybridRetriever)
    retriever._search_graph = lambda *args, **kwargs: []
    retriever._fusion_combine = fake_fusion.__get__(retriever, HybridRetriever)

    query_vec = [0.1, 0.2, 0.3]
    results1, metrics1 = retriever.search(
        query_vector=query_vec,
        query_text="test query",
        top_k=3,
        strategy=SearchStrategy.HYBRID,
    )

    assert vector_calls["count"] == 1
    assert metrics1.cache_hit is False
    assert results1

    # Subsequent call should hit cache (no additional vector search)
    retriever._search_vector = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Cache miss"))
    results2, metrics2 = retriever.search(
        query_vector=query_vec,
        query_text="test query",
        top_k=3,
        strategy=SearchStrategy.HYBRID,
    )

    assert metrics2.cache_hit is True
    assert len(results2) == len(results1)


def test_hybrid_retriever_applies_reranker():
    retriever = HybridRetriever(neo4j_manager=None, qdrant_manager=None)
    retriever.cache_enabled = False

    class FakeReranker:
        def rerank(self, query, results, top_k=None):
            ordered = sorted(results, key=lambda r: r.metadata.get("rank", 0))
            for idx, item in enumerate(ordered):
                ordered[idx] = replace(item, score=1.0 - idx * 0.1)
            return ordered

    retriever.reranker = FakeReranker()
    retriever.reranker_top_n = 10

    dummy_results = [
        HybridSearchResult(
            id=str(i),
            content=f"doc-{i}",
            score=0.2,
            confidence=0.2,
            source="vector",
            vector_score=0.2,
            graph_score=None,
            metadata={"rank": rank},
            graph_context=None,
        )
        for i, rank in enumerate([2, 0, 1])
    ]

    retriever._fusion_combine = lambda vector_results, graph_results, top_k: dummy_results
    retriever._search_vector = lambda *args, **kwargs: []
    retriever._search_graph = lambda *args, **kwargs: []

    results, _ = retriever.search(
        query_vector=[0.1, 0.2, 0.3],
        query_text="rerank",
        top_k=3,
        strategy=SearchStrategy.HYBRID,
    )

    ranks = [r.metadata.get("rank") for r in results]
    assert ranks == [0, 1, 2]
