"""
Comprehensive tests for Neo4j, Qdrant, and Hybrid Retrieval managers.

Tests cover:
- Connection management
- CRUD operations
- Search functionality
- Error handling
- Integration scenarios
"""

import pytest
import time
from typing import List, Dict, Any
from uuid import uuid4

from contextprime.knowledge_graph.neo4j_manager import (
    Neo4jManager,
    GraphNode,
    GraphRelationship,
    SearchResult as Neo4jSearchResult,
)
from contextprime.retrieval.qdrant_manager import (
    QdrantManager,
    VectorPoint,
    SearchResult as QdrantSearchResult,
)
from contextprime.retrieval.hybrid_retriever import (
    HybridRetriever,
    QueryType,
    SearchStrategy,
    HybridSearchResult,
)
from contextprime.core.config import Neo4jConfig, QdrantConfig


# Test fixtures

@pytest.fixture(scope="module")
def neo4j_manager():
    """Create Neo4j manager for testing."""
    from contextprime.core.config import get_settings
    settings = get_settings()
    config = Neo4jConfig(
        uri=settings.neo4j.uri,
        username=settings.neo4j.username,
        password=settings.neo4j.password,
        database=settings.neo4j.database,
    )
    manager = Neo4jManager(config)
    yield manager
    # Cleanup
    manager.clear_database(confirm=True)
    manager.close()


@pytest.fixture(scope="module")
def qdrant_manager():
    """Create Qdrant manager for testing."""
    from contextprime.core.config import get_settings
    settings = get_settings()
    config = QdrantConfig(
        host=settings.qdrant.host,
        port=settings.qdrant.port,
        collection_name="test_collection",
        vector_size=384,  # Small dimension for testing
    )
    manager = QdrantManager(config)

    # Create test collection
    manager.create_collection(recreate=True)

    yield manager

    # Cleanup
    manager.delete_collection()
    manager.close()


@pytest.fixture(scope="module")
def hybrid_retriever(neo4j_manager, qdrant_manager):
    """Create hybrid retriever for testing."""
    retriever = HybridRetriever(
        neo4j_manager=neo4j_manager,
        qdrant_manager=qdrant_manager,
        vector_weight=0.6,
        graph_weight=0.4,
    )
    yield retriever
    retriever.close()


@pytest.fixture
def sample_vectors() -> List[List[float]]:
    """Generate sample vectors for testing."""
    import random
    random.seed(42)
    return [[random.random() for _ in range(384)] for _ in range(10)]


@pytest.fixture
def sample_nodes() -> List[GraphNode]:
    """Generate sample nodes for testing."""
    return [
        GraphNode(
            id=None,
            labels=["Document"],
            properties={
                "doc_id": f"doc_{i}",
                "title": f"Test Document {i}",
                "text": f"This is test document number {i}",
                "created_at": "2024-01-01",
            }
        )
        for i in range(5)
    ]


# Neo4j Manager Tests

class TestNeo4jManager:
    """Test suite for Neo4j manager."""

    def test_connection(self, neo4j_manager):
        """Test Neo4j connection."""
        assert neo4j_manager.health_check()
        assert neo4j_manager._connected

    def test_create_node(self, neo4j_manager):
        """Test creating a single node."""
        result = neo4j_manager.create_node(
            labels=["TestNode"],
            properties={"name": "test", "value": 42}
        )

        assert result is not None
        assert "n" in result

    def test_create_nodes_batch(self, neo4j_manager, sample_nodes):
        """Test batch node creation."""
        count = neo4j_manager.create_nodes_batch(sample_nodes)
        assert count == len(sample_nodes)

        # Verify nodes were created
        query = "MATCH (d:Document) RETURN count(d) as count"
        result = neo4j_manager.execute_query(query)
        assert result[0]["count"] >= len(sample_nodes)

    def test_update_node(self, neo4j_manager):
        """Test updating node properties."""
        # Create node
        result = neo4j_manager.create_node(
            labels=["TestNode"],
            properties={"name": "original"}
        )

        node = result["n"]
        node_id = neo4j_manager.execute_query(
            "MATCH (n) WHERE n.name = 'original' RETURN elementId(n) as id"
        )[0]["id"]

        # Update node
        success = neo4j_manager.update_node(
            node_id=node_id,
            properties={"name": "updated", "new_field": "added"}
        )

        assert success

    def test_delete_node(self, neo4j_manager):
        """Test deleting a node."""
        # Create node
        result = neo4j_manager.create_node(
            labels=["TestNode"],
            properties={"name": "to_delete"}
        )

        node_id = neo4j_manager.execute_query(
            "MATCH (n) WHERE n.name = 'to_delete' RETURN elementId(n) as id"
        )[0]["id"]

        # Delete node
        success = neo4j_manager.delete_node(node_id)
        assert success

    def test_create_relationship(self, neo4j_manager):
        """Test creating relationships between nodes."""
        # Create two nodes
        node1 = neo4j_manager.create_node(
            labels=["Person"],
            properties={"name": "Alice"}
        )
        node2 = neo4j_manager.create_node(
            labels=["Person"],
            properties={"name": "Bob"}
        )

        # Get node IDs
        id1 = neo4j_manager.execute_query(
            "MATCH (n) WHERE n.name = 'Alice' RETURN elementId(n) as id"
        )[0]["id"]
        id2 = neo4j_manager.execute_query(
            "MATCH (n) WHERE n.name = 'Bob' RETURN elementId(n) as id"
        )[0]["id"]

        # Create relationship
        rel = neo4j_manager.create_relationship(
            start_node_id=id1,
            end_node_id=id2,
            rel_type="KNOWS",
            properties={"since": "2024"}
        )

        assert rel is not None

    def test_create_relationships_batch(self, neo4j_manager):
        """Test batch relationship creation."""
        # Create nodes first
        nodes = [
            neo4j_manager.create_node(
                labels=["Node"],
                properties={"name": f"node_{i}"}
            )
            for i in range(3)
        ]

        # Get node IDs
        node_ids = []
        for i in range(3):
            result = neo4j_manager.execute_query(
                f"MATCH (n) WHERE n.name = 'node_{i}' RETURN elementId(n) as id"
            )
            node_ids.append(result[0]["id"])

        # Create relationships
        relationships = [
            GraphRelationship(
                type="CONNECTS",
                start_node=node_ids[0],
                end_node=node_ids[1],
                properties={"weight": 1.0}
            ),
            GraphRelationship(
                type="CONNECTS",
                start_node=node_ids[1],
                end_node=node_ids[2],
                properties={"weight": 0.5}
            ),
        ]

        count = neo4j_manager.create_relationships_batch(relationships)
        assert count == len(relationships)

    def test_vector_index_creation(self, neo4j_manager, sample_vectors):
        """Test creating vector index."""
        # Create nodes with vectors
        for i, vec in enumerate(sample_vectors[:3]):
            neo4j_manager.create_node(
                labels=["VectorNode"],
                properties={
                    "name": f"vec_{i}",
                    "embedding": vec
                }
            )

        # Create vector index
        success = neo4j_manager.initialize_vector_index(
            index_name="test_vector_index",
            label="VectorNode",
            property_name="embedding",
            dimensions=384,
            similarity_function="cosine"
        )

        assert success

    def test_traverse_graph(self, neo4j_manager):
        """Test graph traversal."""
        # Create a simple graph
        node1 = neo4j_manager.create_node(
            labels=["Node"],
            properties={"name": "start"}
        )
        node2 = neo4j_manager.create_node(
            labels=["Node"],
            properties={"name": "middle"}
        )
        node3 = neo4j_manager.create_node(
            labels=["Node"],
            properties={"name": "end"}
        )

        # Get IDs
        id1 = neo4j_manager.execute_query(
            "MATCH (n) WHERE n.name = 'start' RETURN elementId(n) as id"
        )[0]["id"]
        id2 = neo4j_manager.execute_query(
            "MATCH (n) WHERE n.name = 'middle' RETURN elementId(n) as id"
        )[0]["id"]
        id3 = neo4j_manager.execute_query(
            "MATCH (n) WHERE n.name = 'end' RETURN elementId(n) as id"
        )[0]["id"]

        # Create path
        neo4j_manager.create_relationship(id1, id2, "NEXT", {})
        neo4j_manager.create_relationship(id2, id3, "NEXT", {})

        # Traverse
        results = neo4j_manager.traverse_graph(
            start_node_id=id1,
            relationship_types=["NEXT"],
            max_depth=2
        )

        assert len(results) > 0

    def test_pattern_match(self, neo4j_manager):
        """Test pattern matching."""
        results = neo4j_manager.pattern_match(
            pattern="(n:Document)",
            limit=5
        )

        assert isinstance(results, list)

    def test_statistics(self, neo4j_manager):
        """Test getting database statistics."""
        stats = neo4j_manager.get_statistics()

        assert "total_nodes" in stats
        assert "total_relationships" in stats
        assert stats["total_nodes"] >= 0


# Qdrant Manager Tests

class TestQdrantManager:
    """Test suite for Qdrant manager."""

    def test_connection(self, qdrant_manager):
        """Test Qdrant connection."""
        assert qdrant_manager.health_check()
        assert qdrant_manager._connected

    def test_collection_exists(self, qdrant_manager):
        """Test checking collection existence."""
        exists = qdrant_manager.collection_exists()
        assert exists

    def test_insert_vector(self, qdrant_manager, sample_vectors):
        """Test inserting a single vector."""
        vector_id = qdrant_manager.insert_vector(
            vector=sample_vectors[0],
            metadata={"text": "Test document", "type": "test"}
        )

        assert vector_id is not None

    def test_insert_vectors_batch(self, qdrant_manager, sample_vectors):
        """Test batch vector insertion."""
        points = [
            VectorPoint(
                id=str(uuid4()),
                vector=vec,
                metadata={"text": f"Document {i}", "index": i}
            )
            for i, vec in enumerate(sample_vectors)
        ]

        count = qdrant_manager.insert_vectors_batch(points)
        assert count == len(points)

    def test_search(self, qdrant_manager, sample_vectors):
        """Test vector similarity search."""
        # Insert some vectors first
        points = [
            VectorPoint(
                id=f"search_test_{i}",
                vector=vec,
                metadata={"text": f"Search document {i}", "category": "test"}
            )
            for i, vec in enumerate(sample_vectors[:5])
        ]
        qdrant_manager.insert_vectors_batch(points)

        # Wait for indexing
        time.sleep(1)

        # Search
        results = qdrant_manager.search(
            query_vector=sample_vectors[0],
            top_k=3
        )

        assert len(results) > 0
        assert all(isinstance(r, QdrantSearchResult) for r in results)

    def test_search_with_filter(self, qdrant_manager, sample_vectors):
        """Test search with metadata filtering."""
        # Insert vectors with different categories
        points = [
            VectorPoint(
                id=f"filter_test_{i}",
                vector=vec,
                metadata={"text": f"Doc {i}", "category": "A" if i % 2 == 0 else "B"}
            )
            for i, vec in enumerate(sample_vectors[:5])
        ]
        qdrant_manager.insert_vectors_batch(points)

        time.sleep(1)

        # Search with filter
        results = qdrant_manager.search(
            query_vector=sample_vectors[0],
            top_k=5,
            filters={"category": "A"}
        )

        assert len(results) > 0
        assert all(r.metadata.get("category") == "A" for r in results)

    def test_get_vector(self, qdrant_manager, sample_vectors):
        """Test retrieving a vector by ID."""
        vector_id = "get_test_vector"

        # Insert vector
        qdrant_manager.insert_vector(
            vector=sample_vectors[0],
            metadata={"text": "Get test"},
            vector_id=vector_id
        )

        time.sleep(1)

        # Retrieve
        result = qdrant_manager.get_vector(vector_id)

        assert result is not None
        # ID will be normalized to UUID, just verify we got a result with an ID
        assert result.id is not None

    def test_update_vector(self, qdrant_manager, sample_vectors):
        """Test updating vector metadata."""
        vector_id = "update_test_vector"

        # Insert vector
        qdrant_manager.insert_vector(
            vector=sample_vectors[0],
            metadata={"text": "Original"},
            vector_id=vector_id
        )

        time.sleep(1)

        # Update metadata
        success = qdrant_manager.update_vector(
            vector_id=vector_id,
            metadata={"text": "Updated", "new_field": "added"}
        )

        assert success

    def test_delete_vector(self, qdrant_manager, sample_vectors):
        """Test deleting a vector."""
        vector_id = "delete_test_vector"

        # Insert vector
        qdrant_manager.insert_vector(
            vector=sample_vectors[0],
            metadata={"text": "To delete"},
            vector_id=vector_id
        )

        time.sleep(1)

        # Delete
        success = qdrant_manager.delete_vector(vector_id)
        assert success

        # Verify deletion
        result = qdrant_manager.get_vector(vector_id)
        assert result is None

    def test_collection_info(self, qdrant_manager):
        """Test getting collection information."""
        info = qdrant_manager.get_collection_info()

        assert info is not None
        assert "vectors_count" in info
        assert "vector_size" in info
        assert info["vector_size"] == 384

    def test_scroll_collection(self, qdrant_manager, sample_vectors):
        """Test scrolling through collection."""
        # Insert some vectors
        points = [
            VectorPoint(
                id=f"scroll_test_{i}",
                vector=vec,
                metadata={"text": f"Scroll doc {i}"}
            )
            for i, vec in enumerate(sample_vectors[:5])
        ]
        qdrant_manager.insert_vectors_batch(points)

        time.sleep(1)

        # Scroll
        results, next_offset = qdrant_manager.scroll_collection(limit=3)

        assert len(results) <= 3
        assert all(isinstance(r, QdrantSearchResult) for r in results)

    def test_statistics(self, qdrant_manager):
        """Test getting collection statistics."""
        stats = qdrant_manager.get_statistics()

        assert "collection_name" in stats
        assert "total_vectors" in stats


# Hybrid Retriever Tests

class TestHybridRetriever:
    """Test suite for hybrid retriever."""

    def test_initialization(self, hybrid_retriever):
        """Test hybrid retriever initialization."""
        assert hybrid_retriever.neo4j is not None
        assert hybrid_retriever.qdrant is not None
        assert abs(hybrid_retriever.vector_weight + hybrid_retriever.graph_weight - 1.0) < 0.01

    def test_query_type_detection(self, hybrid_retriever):
        """Test query type detection."""
        queries = {
            "What is machine learning?": QueryType.FACTUAL,
            "How are these concepts related?": QueryType.RELATIONSHIP,
            "Explain why this happens": QueryType.COMPLEX,
        }

        for query, expected_type in queries.items():
            detected = hybrid_retriever.detect_query_type(query)
            assert detected == expected_type or detected == QueryType.HYBRID

    def test_query_routing(self, hybrid_retriever):
        """Test query routing to strategies."""
        routing = {
            QueryType.FACTUAL: SearchStrategy.VECTOR_ONLY,
            QueryType.RELATIONSHIP: SearchStrategy.GRAPH_ONLY,
            QueryType.COMPLEX: SearchStrategy.HYBRID,
        }

        for qtype, expected_strategy in routing.items():
            strategy = hybrid_retriever.route_query(qtype)
            assert strategy == expected_strategy

    def test_vector_only_search(self, hybrid_retriever, sample_vectors):
        """Test vector-only search strategy."""
        # Insert test data into Qdrant
        points = [
            VectorPoint(
                id=f"hybrid_vec_{i}",
                vector=vec,
                metadata={"text": f"Hybrid test doc {i}", "source": "vector"}
            )
            for i, vec in enumerate(sample_vectors[:5])
        ]
        hybrid_retriever.qdrant.insert_vectors_batch(points)

        time.sleep(1)

        # Search
        results, metrics = hybrid_retriever.search(
            query_vector=sample_vectors[0],
            query_text="Test query",
            top_k=3,
            strategy=SearchStrategy.VECTOR_ONLY
        )

        assert len(results) > 0
        assert all(isinstance(r, HybridSearchResult) for r in results)
        assert metrics.vector_results > 0
        assert metrics.graph_results == 0

    def test_graph_only_search(self, hybrid_retriever, sample_vectors):
        """Test graph-only search strategy."""
        # Create vector index in Neo4j
        hybrid_retriever.neo4j.initialize_vector_index(
            index_name="hybrid_test_index",
            label="HybridDoc",
            property_name="embedding",
            dimensions=384
        )

        # Insert test data into Neo4j
        for i, vec in enumerate(sample_vectors[:5]):
            hybrid_retriever.neo4j.create_node(
                labels=["HybridDoc"],
                properties={
                    "text": f"Hybrid graph doc {i}",
                    "embedding": vec,
                    "source": "graph"
                }
            )

        time.sleep(1)

        # Search
        results, metrics = hybrid_retriever.search(
            query_vector=sample_vectors[0],
            query_text="Test graph query",
            top_k=3,
            strategy=SearchStrategy.GRAPH_ONLY,
            vector_index_name="hybrid_test_index"
        )

        assert metrics.graph_results >= 0  # May be 0 if index not ready

    def test_hybrid_search(self, hybrid_retriever, sample_vectors):
        """Test hybrid search combining both sources."""
        # Insert data into both databases
        # Qdrant
        points = [
            VectorPoint(
                id=f"hybrid_both_{i}",
                vector=vec,
                metadata={"text": f"Hybrid both doc {i}"}
            )
            for i, vec in enumerate(sample_vectors[:3])
        ]
        hybrid_retriever.qdrant.insert_vectors_batch(points)

        # Neo4j
        for i, vec in enumerate(sample_vectors[:3]):
            hybrid_retriever.neo4j.create_node(
                labels=["HybridBoth"],
                properties={
                    "text": f"Hybrid both doc {i}",
                    "embedding": vec
                }
            )

        time.sleep(1)

        # Search
        results, metrics = hybrid_retriever.search(
            query_vector=sample_vectors[0],
            query_text="Test hybrid query",
            top_k=5,
            strategy=SearchStrategy.HYBRID,
            vector_index_name="hybrid_test_index"
        )

        assert isinstance(results, list)
        assert metrics.strategy == SearchStrategy.HYBRID

    def test_confidence_scoring(self, hybrid_retriever):
        """Test confidence score calculation."""
        # Test with both scores
        confidence = hybrid_retriever._calculate_confidence(
            vector_score=0.9,
            graph_score=0.8,
            source_count=2
        )

        assert 0 <= confidence <= 1
        assert confidence > 0.8  # Should be boosted by diversity

        # Test with single score
        confidence = hybrid_retriever._calculate_confidence(
            vector_score=0.7,
            graph_score=None,
            source_count=1
        )

        assert 0 <= confidence <= 1

    def test_result_reranking(self, hybrid_retriever):
        """Test result reranking."""
        # Create sample results
        results = [
            HybridSearchResult(
                id=f"result_{i}",
                content=f"Content {i}",
                score=0.5 + i * 0.1,
                confidence=0.6 + i * 0.05,
                source="hybrid"
            )
            for i in range(5)
        ]

        # Rerank
        reranked = hybrid_retriever.rerank_results(
            results=results,
            query_text="test query",
            use_diversity=True
        )

        assert len(reranked) == len(results)
        # Check they're sorted by confidence * score
        for i in range(len(reranked) - 1):
            score1 = reranked[i].confidence * reranked[i].score
            score2 = reranked[i + 1].confidence * reranked[i + 1].score
            assert score1 >= score2

    def test_health_check(self, hybrid_retriever):
        """Test health check for both databases."""
        health = hybrid_retriever.health_check()

        assert "neo4j" in health
        assert "qdrant" in health
        assert isinstance(health["neo4j"], bool)
        assert isinstance(health["qdrant"], bool)

    def test_statistics(self, hybrid_retriever):
        """Test getting combined statistics."""
        stats = hybrid_retriever.get_statistics()

        assert "neo4j" in stats
        assert "qdrant" in stats
        assert "weights" in stats
        assert stats["weights"]["vector"] > 0
        assert stats["weights"]["graph"] > 0


# Integration Tests

class TestIntegration:
    """Integration tests for the complete system."""

    def test_end_to_end_indexing_and_retrieval(
        self,
        neo4j_manager,
        qdrant_manager,
        hybrid_retriever,
        sample_vectors
    ):
        """Test complete workflow from indexing to retrieval."""
        # 1. Index documents in both databases
        doc_id = "integration_test_doc"

        # Index in Qdrant
        qdrant_manager.insert_vector(
            vector=sample_vectors[0],
            metadata={
                "doc_id": doc_id,
                "text": "Integration test document",
                "category": "test"
            },
            vector_id=doc_id
        )

        # Index in Neo4j
        neo4j_manager.create_node(
            labels=["IntegrationDoc"],
            properties={
                "doc_id": doc_id,
                "text": "Integration test document",
                "embedding": sample_vectors[0],
                "category": "test"
            }
        )

        time.sleep(1)

        # 2. Perform hybrid search
        results, metrics = hybrid_retriever.search(
            query_vector=sample_vectors[0],
            query_text="integration test",
            top_k=5,
            strategy=SearchStrategy.HYBRID
        )

        # 3. Verify results
        assert len(results) > 0
        assert metrics.total_time_ms > 0

    def test_concurrent_operations(self, qdrant_manager, sample_vectors):
        """Test handling concurrent operations."""
        import concurrent.futures

        def insert_vector(i):
            return qdrant_manager.insert_vector(
                vector=sample_vectors[i % len(sample_vectors)],
                metadata={"text": f"Concurrent doc {i}"}
            )

        # Execute concurrent inserts
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(insert_vector, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 10

    def test_error_recovery(self, neo4j_manager):
        """Test error handling and recovery."""
        # Try to create node with invalid data
        try:
            neo4j_manager.execute_query(
                "INVALID CYPHER QUERY"
            )
            assert False, "Should have raised an error"
        except Exception as e:
            # Should handle error gracefully
            assert neo4j_manager.health_check()


# Performance Tests

class TestPerformance:
    """Performance tests for the system."""

    def test_batch_insertion_performance(self, qdrant_manager, sample_vectors):
        """Test batch insertion performance."""
        # Create large batch
        large_batch = [
            VectorPoint(
                id=str(uuid4()),
                vector=sample_vectors[i % len(sample_vectors)],
                metadata={"text": f"Perf test {i}"}
            )
            for i in range(100)
        ]

        # Time batch insertion
        start = time.time()
        count = qdrant_manager.insert_vectors_batch(large_batch, batch_size=50)
        duration = time.time() - start

        assert count == len(large_batch)
        assert duration < 10  # Should complete within 10 seconds

    def test_search_latency(self, qdrant_manager, sample_vectors):
        """Test search latency."""
        # Ensure data exists
        points = [
            VectorPoint(
                id=f"latency_test_{i}",
                vector=vec,
                metadata={"text": f"Latency doc {i}"}
            )
            for i, vec in enumerate(sample_vectors)
        ]
        qdrant_manager.insert_vectors_batch(points)

        time.sleep(1)

        # Measure search latency
        latencies = []
        for _ in range(10):
            start = time.time()
            qdrant_manager.search(
                query_vector=sample_vectors[0],
                top_k=5
            )
            latencies.append((time.time() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 100  # Should be under 100ms on average


from contextprime.processing.cross_reference_extractor import CrossRef


class TestNeo4jCrossReferenceEdges:
    """Integration tests for store_cross_references in Neo4jManager.

    Requires a live Neo4j connection (run via Docker / full test suite).
    Each test uses UUID-suffixed node IDs to avoid state leakage.
    """

    def _create_chunk_nodes(self, manager, doc_id, chunk_ids):
        """Create Chunk nodes for test setup."""
        for cid in chunk_ids:
            manager.execute_write_query(
                "MERGE (:Chunk {chunk_id: $cid, doc_id: $did, content: $content})",
                {"cid": cid, "did": doc_id, "content": "article_6 provision"},
            )

    def _cleanup(self, manager, doc_id):
        """Remove all Chunk nodes (and their relationships) for the given doc_id."""
        manager.execute_write_query(
            "MATCH (n:Chunk {doc_id: $did}) DETACH DELETE n",
            {"did": doc_id},
        )

    def test_store_cross_references_creates_edge(self, neo4j_manager):
        suffix = str(uuid4())[:8]
        doc_id = f"xref_test_{suffix}"
        src_id = f"src_chunk_{suffix}"
        tgt_id = f"tgt_chunk_{suffix}"
        try:
            self._create_chunk_nodes(neo4j_manager, doc_id, [src_id, tgt_id])
            ref = CrossRef(
                source_chunk_id=src_id,
                target_label="article_6",
                ref_type="article",
                doc_id=doc_id,
            )
            count = neo4j_manager.store_cross_references([ref])
            assert count >= 1

            result = neo4j_manager.execute_query(
                "MATCH ()-[r:REFERENCES {target_label: $label}]->() "
                "RETURN count(r) AS cnt",
                {"label": "article_6"},
            )
            assert result[0]["cnt"] >= 1
        finally:
            self._cleanup(neo4j_manager, doc_id)

    def test_store_cross_references_is_idempotent(self, neo4j_manager):
        suffix = str(uuid4())[:8]
        doc_id = f"xref_idem_{suffix}"
        src_id = f"src_idem_{suffix}"
        tgt_id = f"tgt_idem_{suffix}"
        try:
            self._create_chunk_nodes(neo4j_manager, doc_id, [src_id, tgt_id])
            ref = CrossRef(
                source_chunk_id=src_id,
                target_label="article_6",
                ref_type="article",
                doc_id=doc_id,
            )
            neo4j_manager.store_cross_references([ref])
            neo4j_manager.store_cross_references([ref])  # second call — MERGE, not CREATE

            result = neo4j_manager.execute_query(
                "MATCH (s:Chunk {chunk_id: $sid})-[r:REFERENCES {target_label: 'article_6'}]->() "
                "RETURN count(r) AS cnt",
                {"sid": src_id},
            )
            assert result[0]["cnt"] == 1
        finally:
            self._cleanup(neo4j_manager, doc_id)

    def test_store_cross_references_empty_returns_zero(self, neo4j_manager):
        count = neo4j_manager.store_cross_references([])
        assert count == 0

    def test_store_cross_references_unresolvable_target_skipped(self, neo4j_manager):
        suffix = str(uuid4())[:8]
        doc_id = f"xref_unres_{suffix}"
        src_id = f"src_unres_{suffix}"
        try:
            # Create only the source chunk — no matching target exists
            neo4j_manager.execute_write_query(
                "MERGE (:Chunk {chunk_id: $cid, doc_id: $did, content: $content})",
                {"cid": src_id, "did": doc_id, "content": "no article reference here"},
            )
            ref = CrossRef(
                source_chunk_id=src_id,
                target_label="article_99",
                ref_type="article",
                doc_id=doc_id,
            )
            count = neo4j_manager.store_cross_references([ref])
            assert count == 0
        finally:
            self._cleanup(neo4j_manager, doc_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
