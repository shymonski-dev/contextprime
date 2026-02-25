"""
Demo script for RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) system.

This script demonstrates:
1. Building a hierarchical tree from sample documents
2. Visualizing tree structure
3. Multi-level retrieval
4. Comparing retrieval strategies
5. Performance metrics
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from contextprime.processing.chunker import Chunk
from contextprime.processing.doctags_processor import DocTagsDocument, DocTag, DocTagType
from contextprime.knowledge_graph.neo4j_manager import Neo4jManager
from contextprime.retrieval.qdrant_manager import QdrantManager
from contextprime.summarization.cluster_manager import ClusterManager, ClusteringMethod
from contextprime.summarization.summary_generator import SummaryGenerator
from contextprime.summarization.tree_builder import TreeBuilder
from contextprime.summarization.tree_storage import TreeStorage
from contextprime.summarization.hierarchical_retriever import HierarchicalRetriever, RetrievalStrategy
from contextprime.summarization.raptor_pipeline import RAPTORPipeline, PipelineConfig
from contextprime.summarization.tree_visualizer import TreeVisualizer
from contextprime.core.config import get_settings


def create_sample_document() -> DocTagsDocument:
    """Create a sample document about machine learning."""
    tags = [
        DocTag(
            tag_id="doc_1",
            tag_type=DocTagType.DOCUMENT,
            content="Machine Learning Overview",
            level=0
        ),
        DocTag(
            tag_id="sec_1",
            tag_type=DocTagType.SECTION,
            content="Introduction to Machine Learning",
            level=1
        ),
        DocTag(
            tag_id="p_1",
            tag_type=DocTagType.PARAGRAPH,
            content=(
                "Machine learning is a subset of artificial intelligence that enables "
                "computers to learn from data without being explicitly programmed. "
                "It has revolutionized many industries including healthcare, finance, "
                "and transportation. The field has grown exponentially in recent years "
                "due to advances in computing power and data availability."
            ),
            level=2
        ),
        DocTag(
            tag_id="p_2",
            tag_type=DocTagType.PARAGRAPH,
            content=(
                "There are three main types of machine learning: supervised learning, "
                "unsupervised learning, and reinforcement learning. Each type has its "
                "own applications and methodologies. Supervised learning uses labeled "
                "data to train models, while unsupervised learning finds patterns in "
                "unlabeled data. Reinforcement learning trains agents through rewards."
            ),
            level=2
        ),
        DocTag(
            tag_id="sec_2",
            tag_type=DocTagType.SECTION,
            content="Neural Networks and Deep Learning",
            level=1
        ),
        DocTag(
            tag_id="p_3",
            tag_type=DocTagType.PARAGRAPH,
            content=(
                "Neural networks are computing systems inspired by biological neural "
                "networks. They consist of layers of interconnected nodes that process "
                "information. Deep learning refers to neural networks with many layers, "
                "enabling them to learn complex patterns. These networks have achieved "
                "remarkable success in image recognition, natural language processing, "
                "and game playing."
            ),
            level=2
        ),
        DocTag(
            tag_id="p_4",
            tag_type=DocTagType.PARAGRAPH,
            content=(
                "Convolutional Neural Networks (CNNs) are specialized for processing "
                "grid-like data such as images. They use convolutional layers to detect "
                "features at different scales. Recurrent Neural Networks (RNNs) are "
                "designed for sequential data like text and time series. Long Short-Term "
                "Memory (LSTM) networks are a type of RNN that can learn long-term "
                "dependencies."
            ),
            level=2
        ),
        DocTag(
            tag_id="sec_3",
            tag_type=DocTagType.SECTION,
            content="Applications and Impact",
            level=1
        ),
        DocTag(
            tag_id="p_5",
            tag_type=DocTagType.PARAGRAPH,
            content=(
                "Machine learning has transformed healthcare through improved diagnostics "
                "and personalized treatment plans. In finance, it powers fraud detection "
                "and algorithmic trading. Autonomous vehicles rely heavily on machine "
                "learning for perception and decision-making. The technology also enables "
                "recommendation systems used by streaming services and e-commerce platforms."
            ),
            level=2
        ),
        DocTag(
            tag_id="p_6",
            tag_type=DocTagType.PARAGRAPH,
            content=(
                "Natural language processing applications include chatbots, machine "
                "translation, and sentiment analysis. Computer vision enables facial "
                "recognition, object detection, and medical image analysis. Voice "
                "assistants like Siri and Alexa use speech recognition powered by "
                "machine learning. These applications continue to improve as models "
                "become more sophisticated."
            ),
            level=2
        ),
        DocTag(
            tag_id="sec_4",
            tag_type=DocTagType.SECTION,
            content="Challenges and Future Directions",
            level=1
        ),
        DocTag(
            tag_id="p_7",
            tag_type=DocTagType.PARAGRAPH,
            content=(
                "Despite its successes, machine learning faces several challenges. "
                "Models can be biased if trained on biased data, leading to unfair "
                "outcomes. Interpretability remains a major concern, as complex models "
                "often act as black boxes. Privacy and security issues arise when "
                "dealing with sensitive data. Energy consumption of large models is "
                "also becoming a concern."
            ),
            level=2
        ),
        DocTag(
            tag_id="p_8",
            tag_type=DocTagType.PARAGRAPH,
            content=(
                "Future directions include developing more efficient algorithms, "
                "improving model interpretability, and ensuring fairness and ethics "
                "in AI systems. Transfer learning and few-shot learning aim to reduce "
                "data requirements. Quantum machine learning explores the intersection "
                "of quantum computing and ML. AutoML seeks to automate the model "
                "development process."
            ),
            level=2
        )
    ]

    return DocTagsDocument(
        doc_id="ml_overview",
        title="Machine Learning Overview",
        tags=tags,
        metadata={
            'author': 'Demo Author',
            'date': '2025-10-01'
        }
    )


def setup_components(settings):
    """Setup required components."""
    logger.info("Setting up components...")

    # Initialize embeddings model
    logger.info("Loading embeddings model...")
    embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Initialize Neo4j (optional for demo)
    try:
        neo4j_manager = Neo4jManager(settings.neo4j)
        logger.info("Neo4j connected")
    except Exception as e:
        logger.warning(f"Neo4j not available: {e}")
        neo4j_manager = None

    # Initialize Qdrant (optional for demo)
    try:
        qdrant_manager = QdrantManager(settings.qdrant)
        logger.info("Qdrant connected")
    except Exception as e:
        logger.warning(f"Qdrant not available: {e}")
        qdrant_manager = None

    return embeddings_model, neo4j_manager, qdrant_manager


def demo_tree_building(doc, embeddings_model):
    """Demonstrate tree building."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 1: Building Hierarchical Tree")
    logger.info("="*60)

    # Create components
    cluster_manager = ClusterManager(
        method=ClusteringMethod.KMEANS,
        min_cluster_size=2,
        max_cluster_size=20
    )

    summary_generator = SummaryGenerator(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.1
    )

    tree_builder = TreeBuilder(
        cluster_manager=cluster_manager,
        summary_generator=summary_generator,
        max_depth=3,
        target_branching_factor=3
    )

    # Chunk document
    from contextprime.processing.chunker import StructurePreservingChunker

    chunker = StructurePreservingChunker(chunk_size=300, chunk_overlap=50)
    chunks = chunker.chunk_document(doc)

    logger.info(f"Created {len(chunks)} chunks from document")

    # Generate embeddings
    logger.info("Generating embeddings...")
    chunk_texts = [chunk.content for chunk in chunks]
    embeddings = embeddings_model.encode(chunk_texts, show_progress_bar=True)

    # Build tree
    logger.info("Building tree...")
    root, all_nodes = tree_builder.build_tree(
        chunks=[c.to_dict() for c in chunks],
        embeddings=embeddings,
        doc_id=doc.doc_id
    )

    # Compute statistics
    stats = tree_builder.compute_tree_stats(root, all_nodes)

    logger.info(f"\nTree built successfully!")
    logger.info(f"Total nodes: {stats.total_nodes}")
    logger.info(f"Leaf nodes: {stats.leaf_nodes}")
    logger.info(f"Internal nodes: {stats.internal_nodes}")
    logger.info(f"Max depth: {stats.max_depth}")
    logger.info(f"Avg branching factor: {stats.avg_branching_factor:.2f}")

    return root, all_nodes, embeddings_model


def demo_visualization(root, all_nodes):
    """Demonstrate tree visualization."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 2: Tree Visualization")
    logger.info("="*60)

    visualizer = TreeVisualizer(max_content_length=60)

    # ASCII visualization
    logger.info("\nASCII Tree Structure:")
    logger.info("-" * 60)
    ascii_tree = visualizer.visualize_ascii(
        root=root,
        all_nodes=all_nodes,
        show_content=True,
        show_scores=True,
        max_depth=2  # Limit depth for readability
    )
    print(ascii_tree)

    # Statistics visualization
    stats = TreeBuilder(
        cluster_manager=ClusterManager(),
        summary_generator=SummaryGenerator(),
        max_depth=3
    ).compute_tree_stats(root, all_nodes)

    stats_output = visualizer.visualize_stats(stats, all_nodes)
    print(stats_output)

    # Level visualization
    logger.info("\nLevel 1 Nodes:")
    logger.info("-" * 60)
    level_output = visualizer.visualize_level(
        level=1,
        all_nodes=all_nodes,
        show_content=True
    )
    print(level_output[:500] + "..." if len(level_output) > 500 else level_output)

    # Export to HTML
    html_path = "/tmp/raptor_tree.html"
    success = visualizer.export_to_html(root, all_nodes, html_path)
    if success:
        logger.info(f"\nTree exported to HTML: {html_path}")


def demo_retrieval(root, all_nodes, embeddings_model, neo4j_manager, qdrant_manager):
    """Demonstrate hierarchical retrieval."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 3: Hierarchical Retrieval")
    logger.info("="*60)

    # Setup storage (mock if services not available)
    if neo4j_manager and qdrant_manager:
        tree_storage = TreeStorage(
            neo4j_manager=neo4j_manager,
            qdrant_manager=qdrant_manager
        )

        # Save tree
        tree_id = "demo_ml_tree"
        logger.info(f"Saving tree with ID: {tree_id}")
        tree_storage.save_tree(
            root=root,
            all_nodes=all_nodes,
            tree_id=tree_id,
            metadata={'demo': True}
        )
    else:
        logger.warning("Storage services not available, using in-memory only")
        tree_storage = None

    # Create retriever
    if tree_storage:
        retriever = HierarchicalRetriever(
            tree_storage=tree_storage,
            strategy=RetrievalStrategy.ADAPTIVE,
            top_k=5
        )

        # Test queries
        queries = [
            "What is deep learning?",
            "How is machine learning used in healthcare?",
            "What are the challenges in machine learning?"
        ]

        for query in queries:
            logger.info(f"\nQuery: '{query}'")
            logger.info("-" * 60)

            # Generate query embedding
            query_embedding = embeddings_model.encode([query])[0]

            # Retrieve results
            results = retriever.retrieve(
                query_embedding=query_embedding,
                tree_id=tree_id,
                query_text=query
            )

            # Display results
            for i, result in enumerate(results[:3], 1):
                logger.info(f"\nResult {i} (Score: {result.score:.3f}):")
                logger.info(f"Level: {result.level}, Leaf: {result.node.is_leaf}")
                content_preview = result.node.content[:150]
                logger.info(f"Content: {content_preview}...")
    else:
        # Simple in-memory retrieval
        logger.info("Performing simple similarity search...")

        queries = [
            "What is deep learning?",
            "How is machine learning used in healthcare?"
        ]

        for query in queries:
            logger.info(f"\nQuery: '{query}'")
            logger.info("-" * 60)

            query_embedding = embeddings_model.encode([query])[0]

            # Compute similarities with all nodes
            similarities = []
            for node in all_nodes.values():
                if node.embedding is not None:
                    sim = np.dot(query_embedding, node.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(node.embedding)
                    )
                    similarities.append((node, sim))

            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Display top results
            for i, (node, sim) in enumerate(similarities[:3], 1):
                logger.info(f"\nResult {i} (Similarity: {sim:.3f}):")
                logger.info(f"Level: {node.level}, Leaf: {node.is_leaf}")
                content_preview = node.content[:150]
                logger.info(f"Content: {content_preview}...")


def demo_comparison(root, all_nodes, embeddings_model):
    """Compare flat vs hierarchical retrieval."""
    logger.info("\n" + "="*60)
    logger.info("DEMO 4: Flat vs Hierarchical Retrieval Comparison")
    logger.info("="*60)

    query = "What are neural networks?"
    logger.info(f"Query: '{query}'")

    query_embedding = embeddings_model.encode([query])[0]

    # Flat retrieval (only leaves)
    logger.info("\n1. Flat Retrieval (Leaf Nodes Only):")
    logger.info("-" * 60)

    leaf_nodes = [n for n in all_nodes.values() if n.is_leaf]
    leaf_similarities = []

    for node in leaf_nodes:
        if node.embedding is not None:
            sim = np.dot(query_embedding, node.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node.embedding)
            )
            leaf_similarities.append((node, sim))

    leaf_similarities.sort(key=lambda x: x[1], reverse=True)

    for i, (node, sim) in enumerate(leaf_similarities[:3], 1):
        logger.info(f"Result {i} (Sim: {sim:.3f}): {node.content[:100]}...")

    # Hierarchical retrieval (all levels)
    logger.info("\n2. Hierarchical Retrieval (All Levels):")
    logger.info("-" * 60)

    all_similarities = []

    for node in all_nodes.values():
        if node.embedding is not None:
            sim = np.dot(query_embedding, node.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node.embedding)
            )
            all_similarities.append((node, sim))

    all_similarities.sort(key=lambda x: x[1], reverse=True)

    for i, (node, sim) in enumerate(all_similarities[:3], 1):
        logger.info(
            f"Result {i} (Sim: {sim:.3f}, Level: {node.level}): "
            f"{node.content[:100]}..."
        )

    # Analysis
    logger.info("\nAnalysis:")
    logger.info("-" * 60)
    logger.info("Hierarchical retrieval can provide:")
    logger.info("- High-level summaries for broad understanding")
    logger.info("- Specific details when needed")
    logger.info("- Contextual information from parent/sibling nodes")
    logger.info("- Better coverage of document structure")


def main():
    """Main demo function."""
    logger.info("RAPTOR System Demo")
    logger.info("=" * 60)

    # Load settings
    settings = get_settings()

    # Setup components
    embeddings_model, neo4j_manager, qdrant_manager = setup_components(settings)

    # Create sample document
    doc = create_sample_document()
    logger.info(f"Created sample document: {doc.title}")
    logger.info(f"Document has {len(doc.tags)} tags")

    # Demo 1: Tree Building
    root, all_nodes, embeddings_model = demo_tree_building(doc, embeddings_model)

    # Demo 2: Visualization
    demo_visualization(root, all_nodes)

    # Demo 3: Retrieval
    demo_retrieval(root, all_nodes, embeddings_model, neo4j_manager, qdrant_manager)

    # Demo 4: Comparison
    demo_comparison(root, all_nodes, embeddings_model)

    logger.info("\n" + "="*60)
    logger.info("Demo Complete!")
    logger.info("="*60)
    logger.info("\nThe RAPTOR system has demonstrated:")
    logger.info("1. Hierarchical tree construction from documents")
    logger.info("2. Multi-level summarization")
    logger.info("3. Tree visualization in multiple formats")
    logger.info("4. Hierarchical retrieval with multiple strategies")
    logger.info("5. Comparison of flat vs hierarchical approaches")


if __name__ == "__main__":
    main()
