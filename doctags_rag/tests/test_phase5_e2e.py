"""
Phase 5 End-to-End Tests
Comprehensive real-world RAG workflow validation.
"""

import pytest
from pathlib import Path
import time

from src.processing.pipeline import DocumentProcessingPipeline
from src.retrieval.qdrant_manager import QdrantManager
from src.knowledge_graph.neo4j_manager import Neo4jManager
from src.knowledge_graph.kg_pipeline import KnowledgeGraphPipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.advanced_pipeline import AdvancedRetrievalPipeline
from src.core.config import Neo4jConfig, QdrantConfig, get_settings


@pytest.fixture
def test_docs_dir():
    """Directory containing Phase 5 test documents."""
    return Path("tests/fixtures/phase5_e2e")


@pytest.fixture(scope="module")
def qdrant_manager():
    """Qdrant manager for tests."""
    config = QdrantConfig(
        host="localhost",
        port=6333,
        collection_name="phase5_e2e_test",
        vector_size=384,
    )
    manager = QdrantManager(config)
    yield manager
    # Cleanup
    try:
        manager.delete_collection()
    except:
        pass
    manager.close()


@pytest.fixture(scope="module")
def neo4j_manager():
    """Neo4j manager for tests."""
    settings = get_settings()
    config = Neo4jConfig(
        uri=settings.neo4j.uri,
        username=settings.neo4j.username,
        password=settings.neo4j.password,
        database=settings.neo4j.database,
    )
    manager = Neo4jManager(config)
    yield manager
    # Note: Not clearing database to avoid affecting other tests
    manager.close()


@pytest.fixture
def phase5_collection(qdrant_manager):
    """Create and cleanup Phase 5 test collection."""
    collection_name = "phase5_e2e_test"

    # Create fresh collection
    qdrant_manager.create_collection(recreate=True)

    yield collection_name

    # Cleanup
    try:
        qdrant_manager.delete_collection()
    except:
        pass


class TestPhase5SingleDocumentRAG:
    """Test 1: Single Document RAG - Index → Query → Retrieve → Answer."""

    @pytest.mark.asyncio
    async def test_single_document_workflow(
        self,
        test_docs_dir,
        qdrant_manager,
        phase5_collection
    ):
        """Complete single document RAG workflow."""

        # Step 1: Process document
        doc_path = test_docs_dir / "ml_basics.md"
        assert doc_path.exists(), f"Test document not found: {doc_path}"

        pipeline = DocumentProcessingPipeline()
        result = pipeline.process_file(str(doc_path))

        assert result.success, f"Document processing failed: {result.error}"
        assert len(result.chunks) > 0, "No chunks generated"

        chunks = result.chunks
        print(f"\nProcessed document: {len(chunks)} chunks created")

        # Step 2: Index to Qdrant
        index_start = time.time()

        for i, chunk in enumerate(chunks):
            await qdrant_manager.add_document(
                collection_name=phase5_collection,
                doc_id=chunk.chunk_id,
                content=chunk.content,
                metadata={"chunk_index": i, "doc_id": chunk.doc_id}
            )

        index_time = (time.time() - index_start) * 1000
        print(f"Indexing time: {index_time:.0f}ms for {len(chunks)} chunks")

        # Step 3: Test retrieval with multiple queries
        test_queries = [
            "What is supervised learning?",
            "Explain neural networks",
            "What are evaluation metrics for classification?"
        ]

        for query in test_queries:
            query_start = time.time()

            results = await qdrant_manager.search(
                collection_name=phase5_collection,
                query_text=query,
                top_k=5
            )

            query_time = (time.time() - query_start) * 1000

            assert len(results) > 0, f"No results for query: {query}"
            assert query_time < 1000, f"Query too slow: {query_time:.0f}ms"

            print(f"Query: '{query[:40]}...' → {len(results)} results ({query_time:.0f}ms)")

        # Step 4: Verify result quality
        quality_query = "What is the difference between supervised and unsupervised learning?"
        quality_results = await qdrant_manager.search(
            collection_name=phase5_collection,
            query_text=quality_query,
            top_k=3
        )

        # Check relevance
        relevant_count = 0
        for result in quality_results:
            content_lower = result['content'].lower()
            if 'supervised' in content_lower or 'unsupervised' in content_lower:
                relevant_count += 1

        relevance_ratio = relevant_count / len(quality_results)
        assert relevance_ratio >= 0.5, f"Low relevance: {relevance_ratio:.0%}"

        print(f"Quality check: {relevant_count}/{len(quality_results)} results relevant ({relevance_ratio:.0%})")


class TestPhase5MultiDocumentKnowledgeGraph:
    """Test 2: Multi-Document RAG with Knowledge Graph."""

    @pytest.mark.asyncio
    async def test_multi_document_kg_workflow(
        self,
        test_docs_dir,
        qdrant_manager,
        neo4j_manager,
        phase5_collection
    ):
        """Multi-document RAG with knowledge graph construction."""

        # Step 1: Process all documents
        doc_files = list(test_docs_dir.glob("*.md"))
        assert len(doc_files) >= 3, f"Expected at least 3 test documents, found {len(doc_files)}"

        pipeline = DocumentProcessingPipeline()
        all_chunks = []

        for doc_path in doc_files:
            result = await pipeline.process_file(str(doc_path))
            if result.success:
                all_chunks.extend(result.chunks)
                print(f"Processed {doc_path.name}: {len(result.chunks)} chunks")

        assert len(all_chunks) > 0, "No chunks generated from any document"

        # Step 2: Build knowledge graph
        kg_pipeline = KnowledgeGraphPipeline(neo4j_manager=neo4j_manager)

        total_entities = 0
        total_relationships = 0

        for doc_path in doc_files:
            doc_text = doc_path.read_text()

            kg_result = await kg_pipeline.process_document(
                doc_text,
                doc_id=doc_path.stem,
                metadata={"filename": doc_path.name}
            )

            entities_count = len(kg_result.get('entities', []))
            rels_count = len(kg_result.get('relationships', []))

            total_entities += entities_count
            total_relationships += rels_count

            print(f"KG for {doc_path.name}: {entities_count} entities, {rels_count} relationships")

        assert total_entities > 0, "No entities extracted"
        print(f"\nTotal: {total_entities} entities, {total_relationships} relationships")

        # Step 3: Index chunks to Qdrant
        for chunk in all_chunks:
            await qdrant_manager.add_document(
                collection_name=phase5_collection,
                doc_id=chunk.chunk_id,
                content=chunk.content,
                metadata={"doc_id": chunk.doc_id}
            )

        # Step 4: Test hybrid retrieval
        hybrid = HybridRetriever(
            qdrant_manager=qdrant_manager,
            graph_manager=neo4j_manager
        )

        test_query = "How do neural networks relate to deep learning?"
        hybrid_start = time.time()

        hybrid_results = await hybrid.search(
            query=test_query,
            collection_name=phase5_collection,
            top_k=5,
            strategy="hybrid"
        )

        hybrid_time = (time.time() - hybrid_start) * 1000

        assert len(hybrid_results) > 0, "No hybrid search results"
        assert hybrid_time < 2000, f"Hybrid search too slow: {hybrid_time:.0f}ms"

        print(f"Hybrid search: {len(hybrid_results)} results ({hybrid_time:.0f}ms)")

        await hybrid.close()


class TestPhase5ComplexQueries:
    """Test 5: Complex Multi-Step Analytical Queries."""

    @pytest.mark.asyncio
    async def test_complex_analytical_queries(
        self,
        test_docs_dir,
        qdrant_manager,
        phase5_collection
    ):
        """Test complex queries requiring multi-step reasoning."""

        # Setup: Index all documents
        doc_files = list(test_docs_dir.glob("*.md"))
        pipeline = DocumentProcessingPipeline()

        for doc_path in doc_files:
            result = await pipeline.process_file(str(doc_path))
            if result.success:
                for chunk in result.chunks:
                    await qdrant_manager.add_document(
                        collection_name=phase5_collection,
                        doc_id=chunk.chunk_id,
                        content=chunk.content,
                        metadata={"doc_id": chunk.doc_id}
                    )

        # Initialize advanced retrieval
        advanced = AdvancedRetrievalPipeline(
            qdrant_manager=qdrant_manager,
            collection_name=phase5_collection
        )

        # Test queries of increasing complexity
        test_cases = [
            {
                "query": "What are the main types of machine learning?",
                "type": "factual",
                "min_results": 3
            },
            {
                "query": "Compare CNNs and RNNs in deep learning",
                "type": "comparative",
                "min_results": 3
            },
            {
                "query": "How does backpropagation work and why is it important?",
                "type": "analytical",
                "min_results": 3
            }
        ]

        for test in test_cases:
            query_start = time.time()

            results = await advanced.retrieve(
                query=test['query'],
                top_k=10,
                enable_expansion=True,
                enable_reranking=True
            )

            query_time = (time.time() - query_start) * 1000

            assert len(results) >= test['min_results'], \
                f"{test['type']} query returned {len(results)} results, expected >= {test['min_results']}"

            assert query_time < 3000, f"Query too slow: {query_time:.0f}ms"

            avg_score = sum(r.get('score', 0) for r in results) / len(results)
            assert avg_score >= 0.3, f"Low average score: {avg_score:.3f}"

            print(f"{test['type'].capitalize()} query: {len(results)} results, "
                  f"avg score: {avg_score:.3f}, {query_time:.0f}ms")


class TestPhase5Performance:
    """Performance and quality metrics for Phase 5."""

    @pytest.mark.asyncio
    async def test_latency_requirements(
        self,
        test_docs_dir,
        qdrant_manager,
        phase5_collection
    ):
        """Verify latency requirements for real-world usage."""

        # Setup: Index one document
        doc_path = test_docs_dir / "ml_basics.md"
        pipeline = DocumentProcessingPipeline()
        result = pipeline.process_file(str(doc_path))

        for chunk in result.chunks:
            await qdrant_manager.add_document(
                collection_name=phase5_collection,
                doc_id=chunk.chunk_id,
                content=chunk.content,
                metadata={}
            )

        # Test: Multiple queries to measure consistent latency
        queries = [
            "What is machine learning?",
            "Explain supervised learning",
            "What is deep learning?",
            "How do neural networks work?",
            "What is reinforcement learning?"
        ]

        latencies = []

        for query in queries:
            start = time.time()
            results = await qdrant_manager.search(
                collection_name=phase5_collection,
                query_text=query,
                top_k=5
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            assert len(results) > 0, f"No results for: {query}"

        # Performance assertions
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\nLatency metrics:")
        print(f"  Average: {avg_latency:.0f}ms")
        print(f"  P95: {p95_latency:.0f}ms")
        print(f"  Max: {max_latency:.0f}ms")

        # Requirements
        assert avg_latency < 500, f"Average latency too high: {avg_latency:.0f}ms"
        assert p95_latency < 1000, f"P95 latency too high: {p95_latency:.0f}ms"

    @pytest.mark.asyncio
    async def test_result_quality(
        self,
        test_docs_dir,
        qdrant_manager,
        phase5_collection
    ):
        """Verify retrieval result quality."""

        # Setup: Index documents
        doc_files = list(test_docs_dir.glob("*.md"))
        pipeline = DocumentProcessingPipeline()

        for doc_path in doc_files:
            result = await pipeline.process_file(str(doc_path))
            if result.success:
                for chunk in result.chunks:
                    await qdrant_manager.add_document(
                        collection_name=phase5_collection,
                        doc_id=chunk.chunk_id,
                        content=chunk.content,
                        metadata={"source": doc_path.name}
                    )

        # Test: Specific queries with known expected content
        quality_tests = [
            {
                "query": "What is supervised learning?",
                "expected_keywords": ["supervised", "labeled", "training"],
                "min_keyword_hits": 2
            },
            {
                "query": "Explain convolutional neural networks",
                "expected_keywords": ["cnn", "convolutional", "image", "filter"],
                "min_keyword_hits": 2
            }
        ]

        for test in quality_tests:
            results = await qdrant_manager.search(
                collection_name=phase5_collection,
                query_text=test['query'],
                top_k=3
            )

            assert len(results) > 0, f"No results for: {test['query']}"

            # Check if top results contain expected keywords
            top_content = " ".join([r['content'].lower() for r in results[:3]])

            keyword_hits = sum(
                1 for keyword in test['expected_keywords']
                if keyword.lower() in top_content
            )

            assert keyword_hits >= test['min_keyword_hits'], \
                f"Expected {test['min_keyword_hits']} keywords, found {keyword_hits}"

            print(f"Quality test '{test['query'][:30]}...': "
                  f"{keyword_hits}/{len(test['expected_keywords'])} keywords found")
