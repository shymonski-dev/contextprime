#!/usr/bin/env python3
"""
Simple direct test of the system
"""

import sys
from pathlib import Path
import asyncio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("\n" + "="*60)
print("SIMPLE SMOKE TEST")
print("="*60)

# Test 1: Check what's in Qdrant
print("\n1. Checking Qdrant collections...")
import requests
response = requests.get("http://localhost:6333/collections")
collections = response.json()['result']['collections']
for col in collections:
    name = col['name']
    info = requests.get(f"http://localhost:6333/collections/{name}").json()
    size = info['result']['config']['params']['vectors']['size']
    count = info['result']['points_count']
    print(f"  - {name}: {size} dims, {count} points")

# Test 2: Check Neo4j
print("\n2. Checking Neo4j documents...")
from contextprime.knowledge_graph.neo4j_manager import Neo4jManager
neo = Neo4jManager()
docs = neo.execute_query("MATCH (d:Document) RETURN d.doc_id AS id, d.title AS title LIMIT 5")
print(f"  Found {len(docs)} documents")
for doc in docs[:3]:
    print(f"    - {doc['id']}: {doc['title']}")
neo.close()

# Test 3: Direct Qdrant search in doctags_vectors
print("\n3. Direct search in doctags_vectors collection...")
from contextprime.embeddings import OpenAIEmbeddingModel
embedder = OpenAIEmbeddingModel("text-embedding-3-small")
query = "onboarding"
vector = embedder.encode([query])[0]

search_payload = {
    "vector": vector,
    "limit": 3,
    "with_payload": True
}
response = requests.post("http://localhost:6333/collections/doctags_vectors/points/search", json=search_payload)
if response.status_code == 200:
    results = response.json()['result']
    print(f"  Found {len(results)} results:")
    for r in results[:3]:
        print(f"    - Score: {r['score']:.3f}, Doc: {r.get('payload', {}).get('doc_id', 'N/A')}")
else:
    print(f"  Search failed: {response.text}")

# Test 4: Try retrieval with QdrantManager
print("\n4. Testing retrieval with QdrantManager...")
print("  (Skipped - QdrantManager uses config collection name)")

from contextprime.retrieval.hybrid_retriever import HybridRetriever, SearchStrategy
print("\n4. Testing HybridRetriever...")
try:
    retriever = HybridRetriever()
    results, metrics = retriever.search(
        query_text="onboarding",
        query_vector=vector,
        top_k=3,
        strategy=SearchStrategy.HYBRID
    )
    print(f"  Hybrid search found {len(results)} results")
    retriever.close()
except Exception as e:
    print(f"  Hybrid retriever test failed: {e}")

# Test 5: Try basic agentic query
print("\n5. Testing basic agentic query...")
try:
    from contextprime.agents.agentic_pipeline import AgenticPipeline, AgenticMode

    async def run_query():
        agentic = AgenticPipeline(mode=AgenticMode.FAST)
        result = await agentic.process_query("What documents are available?")
        return result

    result = asyncio.run(run_query())
    print(f"  Response length: {len(result.answer)} chars")
    print(f"  Score: {result.assessment.overall_score}")
    print(f"  First 200 chars: {result.answer[:200]}...")
except Exception as e:
    print(f"  Agentic test failed: {e}")

print("\n" + "="*60)
print("SIMPLE TEST COMPLETE")
print("="*60)
