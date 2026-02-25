#!/usr/bin/env python3
"""
Full smoke test for doctags_rag system
Tests ingestion, storage, retrieval, and agentic processing
"""

import os
import sys
from pathlib import Path
import asyncio
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_ingestion():
    """Test document ingestion pipeline"""
    print("\n" + "="*60)
    print("STEP 1: DOCUMENT INGESTION TEST")
    print("="*60)

    from contextprime.embeddings import OpenAIEmbeddingModel
    from contextprime.pipelines.document_ingestion import DocumentIngestionPipeline

    embedder = OpenAIEmbeddingModel("text-embedding-3-small")
    pipeline = DocumentIngestionPipeline(embeddings_model=embedder)

    samples = [
        Path("data/samples/sample_text.txt"),
        Path("data/samples/sample_markdown.md"),
    ]

    print(f"Processing {len(samples)} sample documents...")
    report = pipeline.process_files(samples)

    report_dict = report.to_dict()
    print("\nIngestion Report:")
    print(json.dumps(report_dict, indent=2))

    pipeline.close()

    assert report_dict['processed_documents'] == 2, f"Expected 2 documents, got {report_dict['processed_documents']}"
    assert report_dict['chunks_ingested'] > 0, f"Expected chunks > 0, got {report_dict['chunks_ingested']}"
    print("\n✓ Ingestion test PASSED")
    return report_dict

def test_qdrant():
    """Test Qdrant vector store"""
    print("\n" + "="*60)
    print("STEP 2: QDRANT VECTOR STORE TEST")
    print("="*60)

    import requests

    # Check for points in the collection
    url = "http://localhost:6333/collections/doctags_vectors/points/scroll"
    payload = {"limit": 5}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        points_count = len(data.get('result', {}).get('points', []))
        print(f"\nQdrant contains {points_count} points (showing up to 5)")

        if points_count > 0:
            print("\nSample point IDs:")
            for point in data['result']['points'][:3]:
                print(f"  - ID: {point['id']}")
                if 'payload' in point and 'doc_id' in point['payload']:
                    print(f"    Doc: {point['payload']['doc_id']}")
                    print(f"    Section: {point['payload'].get('section', 'N/A')}")

        assert points_count > 0, "No points found in Qdrant"
        print("\n✓ Qdrant test PASSED")
        return points_count
    except Exception as e:
        print(f"Error checking Qdrant: {e}")
        return 0

def test_neo4j():
    """Test Neo4j graph database"""
    print("\n" + "="*60)
    print("STEP 3: NEO4J GRAPH DATABASE TEST")
    print("="*60)

    from contextprime.knowledge_graph.neo4j_manager import Neo4jManager

    neo = Neo4jManager()
    docs = neo.execute_query("MATCH (d:Document) RETURN d.doc_id AS id, d.title AS title")

    print(f"\nNeo4j contains {len(docs)} documents:")
    for doc in docs[:5]:  # Show first 5
        print(f"  - ID: {doc['id']}")
        print(f"    Title: {doc['title']}")

    neo.close()

    assert len(docs) > 0, "No documents found in Neo4j"
    print("\n✓ Neo4j test PASSED")
    return len(docs)

def test_hybrid_search():
    """Test hybrid search functionality"""
    print("\n" + "="*60)
    print("STEP 4: HYBRID SEARCH TEST")
    print("="*60)

    from contextprime.retrieval.hybrid_retriever import HybridRetriever, SearchStrategy
    from contextprime.embeddings import OpenAIEmbeddingModel

    embedder = OpenAIEmbeddingModel("text-embedding-3-small")
    query = "What's covered in the onboarding materials?"
    print(f"\nQuery: {query}")

    vector = embedder.encode([query])[0]

    retriever = HybridRetriever()
    results, metrics = retriever.search(
        query_text=query,
        query_vector=vector,
        top_k=5,
        strategy=SearchStrategy.HYBRID
    )

    print("\nSearch Metrics:")
    # Convert metrics to dict if it's a dataclass
    if hasattr(metrics, '__dict__'):
        metrics_dict = {k: v for k, v in metrics.__dict__.items() if not k.startswith('_')}
        print(json.dumps(metrics_dict, indent=2, default=str))
    else:
        print(metrics)

    print(f"\nFound {len(results)} results:")
    for item in results[:5]:
        print(f"  - Doc ID: {item.metadata.get('doc_id', 'N/A')}")
        print(f"    Section: {item.metadata.get('section', 'N/A')}")
        print(f"    Score: {item.score:.4f}")
        print(f"    Content: {item.content[:100]}...")

    retriever.close()

    assert len(results) > 0, "No search results found"
    print("\n✓ Hybrid search test PASSED")
    return len(results)

async def test_agentic():
    """Test agentic processing pipeline"""
    print("\n" + "="*60)
    print("STEP 5: AGENTIC PROCESSING TEST")
    print("="*60)

    from contextprime.agents.agentic_pipeline import AgenticPipeline, AgenticMode

    agentic = AgenticPipeline(mode=AgenticMode.FAST)

    query = "Summarise the onboarding packet"
    print(f"\nQuery: {query}")

    result = await agentic.process_query(query)

    print(f"\nAnswer (first 500 chars):")
    print(result.answer[:500])

    print(f"\nAssessment Score: {result.assessment.overall_score}")
    if hasattr(result.assessment, 'confidence'):
        print(f"Confidence: {result.assessment.confidence}")
    if hasattr(result.assessment, 'query_intent'):
        print(f"Query Intent: {result.assessment.query_intent}")

    assert result.assessment.overall_score > 0, "Score should be non-zero"
    assert len(result.answer) > 0, "Answer should not be empty"
    print("\n✓ Agentic processing test PASSED")

    return result

def main():
    """Run all smoke tests"""
    print("\n" + "="*60)
    print("DOCTAGS_RAG FULL SMOKE TEST")
    print("="*60)

    try:
        # Test 1: Ingestion
        ingestion_report = test_ingestion()

        # Test 2: Qdrant
        qdrant_points = test_qdrant()

        # Test 3: Neo4j
        neo4j_docs = test_neo4j()

        # Test 4: Hybrid Search
        search_results = test_hybrid_search()

        # Test 5: Agentic Processing
        agentic_result = asyncio.run(test_agentic())

        print("\n" + "="*60)
        print("ALL TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        print("\nSummary:")
        print(f"  ✓ Ingested {ingestion_report['processed_documents']} documents")
        print(f"  ✓ Created {ingestion_report['chunks_ingested']} chunks")
        print(f"  ✓ Qdrant has {qdrant_points} points")
        print(f"  ✓ Neo4j has {neo4j_docs} documents")
        print(f"  ✓ Hybrid search returned {search_results} results")
        print(f"  ✓ Agentic processing score: {agentic_result.assessment.overall_score}")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
