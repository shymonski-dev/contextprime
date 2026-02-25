from dataclasses import replace

from contextprime.retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridSearchResult,
    SearchStrategy,
    GraphRetrievalPolicy,
    QueryType,
)
from contextprime.retrieval.qdrant_manager import SearchResult as QdrantResult


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
        ], None

    def fake_fusion(self, vector_results, graph_results, lexical_results, top_k):
        return self._convert_vector_results(vector_results)

    retriever._search_vector = fake_search_vector.__get__(retriever, HybridRetriever)
    retriever._search_graph = lambda *args, **kwargs: ([], None)
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
    retriever._search_vector = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Cache miss"))  # type: ignore[return-value]
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

    retriever._fusion_combine = lambda vector_results, graph_results, lexical_results, top_k: dummy_results
    retriever._search_vector = lambda *args, **kwargs: ([], None)
    retriever._search_graph = lambda *args, **kwargs: ([], None)

    results, _ = retriever.search(
        query_vector=[0.1, 0.2, 0.3],
        query_text="rerank",
        top_k=3,
        strategy=SearchStrategy.HYBRID,
    )

    ranks = [r.metadata.get("rank") for r in results]
    assert ranks == [0, 1, 2]


def test_hybrid_retriever_includes_lexical_signal():
    retriever = HybridRetriever(neo4j_manager=None, qdrant_manager=None)
    retriever.cache_enabled = False
    retriever.lexical_enabled = True
    retriever.lexical_weight = 0.3
    retriever.reranker = None

    retriever._search_vector = lambda *args, **kwargs: ([], None)
    retriever._search_graph = lambda *args, **kwargs: ([], None)
    retriever._search_lexical = lambda *args, **kwargs: [
        QdrantResult(
            id="lex-1",
            score=2.5,
            vector=None,
            metadata={"text": "lexical content"},
        )
    ]

    results, metrics = retriever.search(
        query_vector=[0.1, 0.2, 0.3],
        query_text="lexical query",
        top_k=3,
        strategy=SearchStrategy.HYBRID,
    )

    assert metrics.lexical_results == 1
    assert metrics.lexical_time_ms >= 0
    assert results
    assert results[0].source in {"lexical", "hybrid"}


def test_hybrid_retriever_graph_policies():
    class FakeGraphResult:
        def __init__(self, node_id, score, text, metadata=None, properties=None):
            self.node_id = node_id
            self.score = score
            self.labels = ["Chunk"]
            self.properties = {"text": text}
            if properties:
                self.properties.update(properties)
            self.metadata = metadata or {}

    class FakeNeo4j:
        def __init__(self):
            self.calls = []

        def vector_similarity_search(self, index_name, query_vector, top_k, filters=None):
            self.calls.append("vector")
            return [
                FakeGraphResult("seed-1", 0.9, "seed one"),
                FakeGraphResult("seed-2", 0.8, "seed two"),
            ]

        def expand_from_seed_nodes(self, seed_scores, max_depth=2, limit=100):
            self.calls.append("local_expand")
            return [
                FakeGraphResult(
                    "neighbor-1",
                    0.6,
                    "neighbor one",
                    metadata={"seed_node_id": "seed-1"},
                )
            ]

        def keyword_search_nodes(self, query_text, top_k=20, scan_limit=1500, max_terms=8):
            self.calls.append("global_keyword")
            return [
                FakeGraphResult("global-1", 0.7, f"global for {query_text}")
            ]

        def community_summary_search(
            self,
            query_text,
            top_k=20,
            version=None,
            scan_limit=500,
            max_terms=8,
        ):
            self.calls.append("community_summary")
            return [
                FakeGraphResult(
                    "community-1",
                    0.75,
                    f"community summary for {query_text}",
                    metadata={"community_id": "v1_1", "graph_signal": "community_summary"},
                    properties={"community_id": "v1_1"},
                )
            ]

        def community_member_search(
            self,
            community_scores,
            top_k=20,
            members_per_community=6,
        ):
            self.calls.append("community_member")
            return [
                FakeGraphResult(
                    "member-1",
                    0.7,
                    "community member evidence",
                    metadata={"community_id": "v1_1", "graph_signal": "community_membership"},
                )
            ]

    retriever = HybridRetriever(neo4j_manager=None, qdrant_manager=None)
    fake_neo4j = FakeNeo4j()
    retriever._ensure_neo4j = lambda: fake_neo4j

    local_results, _ = retriever._search_graph(
        query_vector=[0.1, 0.2, 0.3],
        query_text="graph policy test",
        top_k=3,
        filters=None,
        vector_index_name="chunk_embeddings",
        graph_policy=GraphRetrievalPolicy.LOCAL,
    )
    assert local_results
    assert "vector" in fake_neo4j.calls
    assert "local_expand" in fake_neo4j.calls

    fake_neo4j.calls.clear()
    global_results, _ = retriever._search_graph(
        query_vector=[0.1, 0.2, 0.3],
        query_text="graph policy test",
        top_k=3,
        filters=None,
        vector_index_name="chunk_embeddings",
        graph_policy=GraphRetrievalPolicy.GLOBAL,
    )
    assert global_results
    assert fake_neo4j.calls == ["global_keyword"]

    fake_neo4j.calls.clear()
    drift_results, _ = retriever._search_graph(
        query_vector=[0.1, 0.2, 0.3],
        query_text="graph policy test",
        top_k=3,
        filters=None,
        vector_index_name="chunk_embeddings",
        graph_policy=GraphRetrievalPolicy.DRIFT,
    )
    assert drift_results
    assert "vector" in fake_neo4j.calls
    assert "local_expand" in fake_neo4j.calls
    assert "global_keyword" in fake_neo4j.calls

    fake_neo4j.calls.clear()
    community_results, _ = retriever._search_graph(
        query_vector=[0.1, 0.2, 0.3],
        query_text="graph policy test",
        top_k=3,
        filters=None,
        vector_index_name="chunk_embeddings",
        graph_policy=GraphRetrievalPolicy.COMMUNITY,
    )
    assert community_results
    assert "community_summary" in fake_neo4j.calls
    assert "community_member" in fake_neo4j.calls
    assert "vector" in fake_neo4j.calls


def test_hybrid_retriever_backend_errors_surfaced_in_search_errors():
    """Backend failures must appear in metrics.search_errors, not be silently swallowed.

    _search_vector and _search_graph return (results, Optional[str]) — when an error
    string is returned the search() method must collect it in metrics.search_errors.
    """
    retriever = HybridRetriever(neo4j_manager=None, qdrant_manager=None)
    retriever.cache_enabled = False
    retriever.reranker = None

    # Mock both backends to return the error-tuple contract (results=[], error=str)
    retriever._search_vector = lambda *args, **kwargs: (
        [],
        "vector: TimeoutError: qdrant timed out",
    )
    retriever._search_graph = lambda *args, **kwargs: (
        [],
        "graph: ConnectionError: neo4j unreachable",
    )

    results, metrics = retriever.search(
        query_vector=[0.1, 0.2, 0.3],
        query_text="failure test",
        top_k=3,
        strategy=SearchStrategy.HYBRID,
    )

    # Both backends returned errors — results should be empty but no exception raised
    assert results == []
    # Both error messages must be collected
    assert len(metrics.search_errors) == 2
    assert any("vector" in e for e in metrics.search_errors)
    assert any("graph" in e for e in metrics.search_errors)


def test_hybrid_retriever_adaptive_policy_selection():
    retriever = HybridRetriever(neo4j_manager=None, qdrant_manager=None)

    community_policy = retriever._select_adaptive_graph_policy(
        query_text="show community cluster evidence patterns",
        query_type=QueryType.COMPLEX,
    )
    assert community_policy == GraphRetrievalPolicy.COMMUNITY

    global_policy = retriever._select_adaptive_graph_policy(
        query_text="provide an overall trend summary across all components",
        query_type=QueryType.COMPLEX,
    )
    assert global_policy == GraphRetrievalPolicy.GLOBAL
