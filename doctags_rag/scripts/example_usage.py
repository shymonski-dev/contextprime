#!/usr/bin/env python3
"""
Example usage of the dual indexing infrastructure.

Demonstrates:
1. Indexing documents in both databases
2. Performing hybrid search
3. Working with graph relationships
4. Query routing
"""

import sys
from pathlib import Path
from typing import List
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from contextprime.knowledge_graph.neo4j_manager import Neo4jManager, GraphNode, GraphRelationship
from contextprime.retrieval.qdrant_manager import QdrantManager, VectorPoint
from contextprime.retrieval.hybrid_retriever import HybridRetriever, SearchStrategy


def generate_dummy_embedding(dimension: int = 1536) -> List[float]:
    """Generate a random embedding for demonstration."""
    random.seed(42)
    return [random.random() for _ in range(dimension)]


def example_indexing():
    """Example: Index documents in both databases."""
    logger.info("=== Example 1: Indexing Documents ===")

    neo4j = Neo4jManager()
    qdrant = QdrantManager()

    try:
        # Sample documents
        documents = [
            {
                "doc_id": "doc_001",
                "title": "Introduction to Machine Learning",
                "text": "Machine learning is a subset of artificial intelligence...",
                "author": "John Doe",
                "date": "2024-01-01"
            },
            {
                "doc_id": "doc_002",
                "title": "Deep Learning Fundamentals",
                "text": "Deep learning uses neural networks with multiple layers...",
                "author": "Jane Smith",
                "date": "2024-01-15"
            },
            {
                "doc_id": "doc_003",
                "title": "Natural Language Processing",
                "text": "NLP enables computers to understand and process human language...",
                "author": "Bob Johnson",
                "date": "2024-02-01"
            }
        ]

        # Index in Neo4j (with graph structure)
        logger.info("Indexing in Neo4j...")
        for doc in documents:
            # Create document node
            embedding = generate_dummy_embedding()

            node_data = neo4j.create_node(
                labels=["Document"],
                properties={
                    **doc,
                    "embedding": embedding
                }
            )
            logger.info(f"Created Neo4j node for {doc['doc_id']}")

        # Create relationships between documents
        logger.info("Creating document relationships...")

        # Get document IDs from Neo4j
        query = "MATCH (d:Document) RETURN elementId(d) as id, d.doc_id as doc_id ORDER BY d.doc_id"
        doc_nodes = neo4j.execute_query(query)

        if len(doc_nodes) >= 2:
            # Create "REFERENCES" relationship
            neo4j.create_relationship(
                start_node_id=doc_nodes[0]["id"],
                end_node_id=doc_nodes[1]["id"],
                rel_type="REFERENCES",
                properties={"weight": 0.8}
            )
            logger.info(f"Created relationship: {doc_nodes[0]['doc_id']} -> {doc_nodes[1]['doc_id']}")

        # Index in Qdrant (vector search)
        logger.info("Indexing in Qdrant...")
        points = []
        for doc in documents:
            embedding = generate_dummy_embedding()

            point = VectorPoint(
                id=doc["doc_id"],
                vector=embedding,
                metadata={
                    "title": doc["title"],
                    "text": doc["text"],
                    "author": doc["author"],
                    "date": doc["date"],
                    "doc_type": "article"
                }
            )
            points.append(point)

        count = qdrant.insert_vectors_batch(points)
        logger.info(f"Indexed {count} vectors in Qdrant")

        logger.success("Indexing complete!")

    finally:
        neo4j.close()
        qdrant.close()


def example_hybrid_search():
    """Example: Perform hybrid search."""
    logger.info("\n=== Example 2: Hybrid Search ===")

    retriever = HybridRetriever()

    try:
        # Query
        query_text = "What is machine learning?"
        query_embedding = generate_dummy_embedding()

        logger.info(f"Query: {query_text}")

        # Perform search with different strategies
        strategies = [
            SearchStrategy.VECTOR_ONLY,
            SearchStrategy.GRAPH_ONLY,
            SearchStrategy.HYBRID
        ]

        for strategy in strategies:
            logger.info(f"\nUsing strategy: {strategy.value}")

            results, metrics = retriever.search(
                query_vector=query_embedding,
                query_text=query_text,
                top_k=3,
                strategy=strategy,
                vector_index_name="document_embeddings"
            )

            logger.info(f"Found {len(results)} results in {metrics.total_time_ms:.2f}ms")
            logger.info(f"Vector results: {metrics.vector_results}, Graph results: {metrics.graph_results}")

            for i, result in enumerate(results[:3], 1):
                logger.info(
                    f"{i}. Score: {result.score:.3f}, Confidence: {result.confidence:.3f}, "
                    f"Source: {result.source}"
                )
                if result.metadata:
                    logger.info(f"   Title: {result.metadata.get('title', 'N/A')}")

    finally:
        retriever.close()


def example_query_routing():
    """Example: Automatic query routing."""
    logger.info("\n=== Example 3: Query Routing ===")

    retriever = HybridRetriever()

    try:
        # Different query types
        queries = [
            "What is deep learning?",  # Factual -> Vector search
            "How are ML and DL related?",  # Relationship -> Graph search
            "Explain the connection between NLP and machine learning",  # Complex -> Hybrid
        ]

        for query in queries:
            logger.info(f"\nQuery: {query}")

            # Detect query type
            query_type = retriever.detect_query_type(query)
            strategy = retriever.route_query(query_type)

            logger.info(f"Detected type: {query_type.value}")
            logger.info(f"Routed to strategy: {strategy.value}")

            # Perform search
            query_embedding = generate_dummy_embedding()
            results, metrics = retriever.search(
                query_vector=query_embedding,
                query_text=query,
                top_k=3,
                strategy=strategy,
                vector_index_name="document_embeddings"
            )

            logger.info(f"Results: {len(results)}, Time: {metrics.total_time_ms:.2f}ms")

    finally:
        retriever.close()


def example_graph_traversal():
    """Example: Graph traversal and context enrichment."""
    logger.info("\n=== Example 4: Graph Traversal ===")

    neo4j = Neo4jManager()

    try:
        # Get a document node
        query = "MATCH (d:Document) RETURN elementId(d) as id, d.doc_id as doc_id LIMIT 1"
        result = neo4j.execute_query(query)

        if result:
            node_id = result[0]["id"]
            doc_id = result[0]["doc_id"]

            logger.info(f"Starting from document: {doc_id}")

            # Traverse graph
            logger.info("Traversing relationships...")
            paths = neo4j.traverse_graph(
                start_node_id=node_id,
                relationship_types=["REFERENCES", "MENTIONS"],
                direction="both",
                max_depth=2,
                limit=10
            )

            logger.info(f"Found {len(paths)} paths")

            # Get statistics
            stats = neo4j.get_statistics()
            logger.info(f"\nGraph statistics:")
            logger.info(f"Total nodes: {stats['total_nodes']}")
            logger.info(f"Total relationships: {stats['total_relationships']}")
            logger.info(f"Nodes by label: {stats['nodes_by_label']}")

    finally:
        neo4j.close()


def example_batch_operations():
    """Example: Efficient batch operations."""
    logger.info("\n=== Example 5: Batch Operations ===")

    neo4j = Neo4jManager()
    qdrant = QdrantManager()

    try:
        # Create batch of nodes
        logger.info("Creating batch of entities...")
        entities = [
            GraphNode(
                id=None,
                labels=["Entity", "Concept"],
                properties={
                    "name": f"Concept_{i}",
                    "type": "technical",
                    "description": f"Technical concept number {i}"
                }
            )
            for i in range(10)
        ]

        count = neo4j.create_nodes_batch(entities, batch_size=5)
        logger.info(f"Created {count} entity nodes")

        # Create batch of vectors
        logger.info("Creating batch of vectors...")
        vectors = [
            VectorPoint(
                id=f"concept_{i}",
                vector=generate_dummy_embedding(),
                metadata={
                    "name": f"Concept_{i}",
                    "type": "technical"
                }
            )
            for i in range(10)
        ]

        count = qdrant.insert_vectors_batch(vectors, batch_size=5)
        logger.info(f"Inserted {count} vectors")

        logger.success("Batch operations complete!")

    finally:
        neo4j.close()
        qdrant.close()


def example_health_monitoring():
    """Example: Health checks and statistics."""
    logger.info("\n=== Example 6: Health Monitoring ===")

    retriever = HybridRetriever()

    try:
        # Health check
        health = retriever.health_check()
        logger.info(f"System health:")
        logger.info(f"  Neo4j: {'✓' if health['neo4j'] else '✗'}")
        logger.info(f"  Qdrant: {'✓' if health['qdrant'] else '✗'}")

        # Statistics
        stats = retriever.get_statistics()
        logger.info(f"\nNeo4j statistics:")
        for key, value in stats['neo4j'].items():
            logger.info(f"  {key}: {value}")

        logger.info(f"\nQdrant statistics:")
        for key, value in stats['qdrant'].items():
            logger.info(f"  {key}: {value}")

        logger.info(f"\nRetrieval weights:")
        logger.info(f"  Vector: {stats['weights']['vector']:.2f}")
        logger.info(f"  Graph: {stats['weights']['graph']:.2f}")

    finally:
        retriever.close()


def main():
    """Run all examples."""
    logger.info("Contextprime - Dual Indexing Infrastructure Examples\n")

    try:
        # Run examples
        example_indexing()
        example_hybrid_search()
        example_query_routing()
        example_graph_traversal()
        example_batch_operations()
        example_health_monitoring()

        logger.success("\nAll examples completed successfully!")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
