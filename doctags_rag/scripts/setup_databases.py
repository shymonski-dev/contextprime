#!/usr/bin/env python3
"""
Setup script for initializing Neo4j and Qdrant databases.

This script:
1. Tests database connections
2. Creates necessary indexes and collections
3. Sets up schema constraints
4. Validates the setup
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from contextprime.core.config import get_settings
from contextprime.knowledge_graph.neo4j_manager import Neo4jManager
from contextprime.retrieval.qdrant_manager import QdrantManager


def setup_neo4j(manager: Neo4jManager) -> bool:
    """
    Setup Neo4j database with indexes and constraints.

    Args:
        manager: Neo4j manager instance

    Returns:
        True if setup successful
    """
    logger.info("Setting up Neo4j database...")

    try:
        # 1. Health check
        if not manager.health_check():
            logger.error("Neo4j health check failed")
            return False

        logger.success("Neo4j connection healthy")

        # 2. Create constraints
        logger.info("Creating schema constraints...")
        manager.create_schema_constraints()

        # 3. Create indexes
        logger.info("Creating indexes...")
        manager.create_indexes()

        # 4. Create vector indexes
        logger.info("Creating vector indexes...")

        # Document embeddings index
        manager.initialize_vector_index(
            index_name="document_embeddings",
            label="Document",
            property_name="embedding",
            dimensions=1536,  # OpenAI embeddings
            similarity_function="cosine"
        )

        # Section embeddings index
        manager.initialize_vector_index(
            index_name="section_embeddings",
            label="Section",
            property_name="embedding",
            dimensions=1536,
            similarity_function="cosine"
        )

        # Chunk embeddings index
        manager.initialize_vector_index(
            index_name="chunk_embeddings",
            label="Chunk",
            property_name="embedding",
            dimensions=1536,
            similarity_function="cosine"
        )

        # Entity embeddings index (if using entity embeddings)
        manager.initialize_vector_index(
            index_name="entity_embeddings",
            label="Entity",
            property_name="embedding",
            dimensions=1536,
            similarity_function="cosine"
        )

        # 5. Get statistics
        stats = manager.get_statistics()
        logger.info(f"Neo4j setup complete. Stats: {stats}")

        return True

    except Exception as e:
        logger.error(f"Neo4j setup failed: {e}")
        return False


def setup_qdrant(manager: QdrantManager) -> bool:
    """
    Setup Qdrant database with collections.

    Args:
        manager: Qdrant manager instance

    Returns:
        True if setup successful
    """
    logger.info("Setting up Qdrant database...")

    try:
        # 1. Health check
        if not manager.health_check():
            logger.error("Qdrant health check failed")
            return False

        logger.success("Qdrant connection healthy")

        # 2. Create main collection
        logger.info("Creating main vectors collection...")
        manager.create_collection(recreate=False)

        # 3. Create additional collections for different content types
        # Chunk-level embeddings
        logger.info("Creating chunks collection...")
        manager.create_collection(
            collection_name="doctags_chunks",
            vector_size=1536,
            distance_metric="cosine"
        )

        # Summary embeddings (RAPTOR)
        logger.info("Creating summaries collection...")
        manager.create_collection(
            collection_name="doctags_summaries",
            vector_size=1536,
            distance_metric="cosine"
        )

        # Entity embeddings
        logger.info("Creating entities collection...")
        manager.create_collection(
            collection_name="doctags_entities",
            vector_size=1536,
            distance_metric="cosine"
        )

        # 4. Get statistics
        stats = manager.get_statistics()
        logger.info(f"Qdrant setup complete. Stats: {stats}")

        return True

    except Exception as e:
        logger.error(f"Qdrant setup failed: {e}")
        return False


def main():
    """Main setup function."""
    logger.info("Starting database setup...")

    # Load configuration
    settings = get_settings()

    # Initialize managers
    neo4j_manager = Neo4jManager()
    qdrant_manager = QdrantManager()

    success = True

    try:
        # Setup Neo4j
        if not setup_neo4j(neo4j_manager):
            logger.error("Neo4j setup failed")
            success = False

        # Setup Qdrant
        if not setup_qdrant(qdrant_manager):
            logger.error("Qdrant setup failed")
            success = False

        if success:
            logger.success("All databases setup successfully!")
            logger.info("\nYou can now:")
            logger.info("1. Start indexing documents")
            logger.info("2. Run the test suite: pytest tests/test_indexing.py")
            logger.info("3. Use the hybrid retriever for search")

            return 0
        else:
            logger.error("Database setup completed with errors")
            return 1

    except Exception as e:
        logger.error(f"Setup failed with exception: {e}")
        return 1

    finally:
        # Cleanup
        neo4j_manager.close()
        qdrant_manager.close()


if __name__ == "__main__":
    sys.exit(main())
