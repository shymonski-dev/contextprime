#!/usr/bin/env python3
"""
Reset collections and run full smoke test
"""

import os
import sys
from pathlib import Path
import asyncio
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def reset_qdrant():
    """Reset Qdrant collections"""
    print("\nResetting Qdrant collections...")
    import requests

    # Delete existing collections
    for collection in ["doctags_vectors", "doctags_chunks"]:
        try:
            response = requests.delete(f"http://localhost:6333/collections/{collection}")
            if response.status_code == 200:
                print(f"  Deleted collection: {collection}")
        except:
            pass

    print("✓ Qdrant reset complete")

def reset_neo4j():
    """Clear Neo4j database"""
    print("\nResetting Neo4j database...")
    from contextprime.knowledge_graph.neo4j_manager import Neo4jManager

    neo = Neo4jManager()
    # Clear all nodes and relationships
    neo.execute_query("MATCH (n) DETACH DELETE n")
    neo.close()

    print("✓ Neo4j reset complete")

def test_ingestion():
    """Test document ingestion pipeline"""
    print("\n" + "="*60)
    print("DOCUMENT INGESTION TEST")
    print("="*60)

    from contextprime.pipelines.document_ingestion import DocumentIngestionPipeline

    pipeline = DocumentIngestionPipeline()

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
    return report_dict

def test_hybrid_search():
    """Test hybrid search functionality"""
    print("\n" + "="*60)
    print("HYBRID SEARCH TEST")
    print("="*60)

    from contextprime.retrieval.hybrid_retriever import HybridRetriever, SearchStrategy
    from contextprime.embeddings import OpenAIEmbeddingModel

    embedder = OpenAIEmbeddingModel("text-embedding-3-small")
    query = "What's covered in the onboarding materials?"
    print(f"\nQuery: {query}")

    vector = embedder.encode([query])[0]
    print(f"Vector dimensions: {len(vector)}")

    retriever = HybridRetriever()
    results, metrics = retriever.search(
        query_text=query,
        query_vector=vector,
        top_k=5,
        strategy=SearchStrategy.HYBRID
    )

    print("\nSearch Metrics:")
    if hasattr(metrics, '__dict__'):
        metrics_dict = {k: str(v) for k, v in metrics.__dict__.items() if not k.startswith('_')}
        print(json.dumps(metrics_dict, indent=2))
    else:
        print(metrics)

    print(f"\nFound {len(results)} results:")
    for item in results[:5]:
        print(f"  - Doc ID: {item.metadata.get('doc_id', 'N/A')}")
        print(f"    Section: {item.metadata.get('section', 'N/A')}")
        print(f"    Score: {item.score:.4f}")
        print(f"    Content: {item.content[:100]}...")

    retriever.close()
    return len(results)

async def test_agentic():
    """Test agentic processing pipeline"""
    print("\n" + "="*60)
    print("AGENTIC PROCESSING TEST")
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

    return result

def main():
    """Run all tests with reset"""
    print("\n" + "="*60)
    print("DOCTAGS_RAG RESET AND TEST")
    print("="*60)

    try:
        # Reset databases
        reset_qdrant()
        reset_neo4j()

        # Run ingestion
        ingestion_report = test_ingestion()

        # Run search test
        search_results = test_hybrid_search()

        # Run agentic test
        agentic_result = asyncio.run(test_agentic())

        print("\n" + "="*60)
        print("ALL TESTS COMPLETED!")
        print("="*60)
        print("\nSummary:")
        print(f"  ✓ Ingested {ingestion_report['processed_documents']} documents")
        print(f"  ✓ Created {ingestion_report['chunks_ingested']} chunks")
        print(f"  ✓ Hybrid search returned {search_results} results")
        print(f"  ✓ Agentic processing score: {agentic_result.assessment.overall_score}")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
