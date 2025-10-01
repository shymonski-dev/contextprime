"""
Sample Knowledge Graph Builder Script.

Demonstrates the complete knowledge graph construction pipeline:
1. Load sample documents
2. Extract entities and relationships
3. Resolve entities
4. Build graph in Neo4j
5. Query and visualize results
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime
from typing import List, Dict, Any

from loguru import logger

from src.knowledge_graph import (
    KnowledgeGraphPipeline,
    PipelineConfig,
    GraphQueryInterface,
)


# Sample documents for demonstration
SAMPLE_DOCUMENTS = [
    {
        "text": """
        Artificial Intelligence (AI) is transforming the technology industry. Companies like
        OpenAI, Google DeepMind, and Anthropic are leading the development of large language
        models (LLMs). OpenAI created GPT-4, one of the most advanced AI systems. Google
        DeepMind developed AlphaGo, which defeated human champions in the game of Go.
        Anthropic focuses on AI safety and created Claude, a helpful and harmless AI assistant.

        These companies are headquartered in San Francisco, London, and the Bay Area.
        Sam Altman leads OpenAI, while Demis Hassabis leads Google DeepMind. Dario Amodei
        and Daniela Amodei founded Anthropic in 2021.
        """,
        "doc_id": "ai_companies",
        "metadata": {
            "title": "AI Companies and Leadership",
            "source": "tech_news",
            "content_type": "article",
            "tags": ["AI", "technology", "companies"]
        }
    },
    {
        "text": """
        Machine Learning is a subset of Artificial Intelligence that enables computers to
        learn from data. Deep Learning, a branch of Machine Learning, uses neural networks
        with multiple layers. Transformers, introduced in the paper "Attention is All You Need"
        by Google researchers, revolutionized natural language processing.

        GPT (Generative Pre-trained Transformer) architecture, developed by OpenAI, uses
        transformers for text generation. BERT (Bidirectional Encoder Representations from
        Transformers) by Google improved language understanding. These models are trained on
        massive datasets and require significant computational resources.
        """,
        "doc_id": "ml_concepts",
        "metadata": {
            "title": "Machine Learning Concepts",
            "source": "tech_docs",
            "content_type": "documentation",
            "tags": ["ML", "deep learning", "transformers"]
        }
    },
    {
        "text": """
        Python is the most popular programming language for AI and Machine Learning.
        TensorFlow and PyTorch are the leading deep learning frameworks. TensorFlow was
        developed by Google, while PyTorch was created by Meta (formerly Facebook).

        NumPy provides numerical computing capabilities, while Pandas is used for data
        manipulation. Scikit-learn offers traditional machine learning algorithms.
        Hugging Face provides a platform for sharing pre-trained models and has become
        essential for NLP tasks. The Hugging Face Transformers library supports models
        like BERT, GPT, and T5.
        """,
        "doc_id": "ml_tools",
        "metadata": {
            "title": "Machine Learning Tools and Frameworks",
            "source": "tech_docs",
            "content_type": "documentation",
            "tags": ["Python", "frameworks", "tools"]
        }
    },
    {
        "text": """
        Natural Language Processing (NLP) enables computers to understand and generate
        human language. Named Entity Recognition (NER) identifies entities like people,
        organizations, and locations. Sentiment Analysis determines the emotional tone
        of text. Machine Translation translates text between languages.

        Recent advances in NLP include BERT for understanding, GPT for generation, and
        T5 for text-to-text tasks. These models use attention mechanisms to process
        sequential data. OpenAI's ChatGPT and Google's Bard are examples of conversational
        AI systems built on these technologies.
        """,
        "doc_id": "nlp_applications",
        "metadata": {
            "title": "NLP Applications and Techniques",
            "source": "tech_docs",
            "content_type": "documentation",
            "tags": ["NLP", "language models", "applications"]
        }
    }
]


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_statistics(stats: Dict[str, Any], indent: int = 0):
    """Pretty print statistics."""
    prefix = "  " * indent
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_statistics(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")


def main():
    """Main function to demonstrate knowledge graph construction."""
    print_section("DocTags RAG - Knowledge Graph Construction Demo")

    # Configure pipeline
    print("Configuring knowledge graph pipeline...")
    config = PipelineConfig(
        extract_entities=True,
        extract_relationships=True,
        resolve_entities=True,
        use_llm=False,  # Set to True if you have OpenAI/Anthropic API key
        batch_size=10,
        confidence_threshold=0.7,
        enable_progress_bar=True
    )

    print(f"  - Entity extraction: {config.extract_entities}")
    print(f"  - Relationship extraction: {config.extract_relationships}")
    print(f"  - Entity resolution: {config.resolve_entities}")
    print(f"  - Use LLM: {config.use_llm}")
    print(f"  - Confidence threshold: {config.confidence_threshold}")

    # Initialize pipeline
    try:
        print("\nInitializing pipeline...")
        pipeline = KnowledgeGraphPipeline(config=config)
        print("  ✓ Pipeline initialized successfully")
    except Exception as e:
        print(f"  ✗ Failed to initialize pipeline: {e}")
        print("\nMake sure Neo4j is running:")
        print("  docker run -p 7687:7687 -p 7474:7474 neo4j:latest")
        return

    # Clear existing data (optional)
    print("\nClearing existing graph data...")
    try:
        pipeline.graph_builder.clear_graph(confirm=True)
        print("  ✓ Graph cleared")
    except Exception as e:
        print(f"  ⚠ Warning: Could not clear graph: {e}")

    # Process documents
    print_section("Processing Documents")
    print(f"Processing {len(SAMPLE_DOCUMENTS)} sample documents...")

    start_time = datetime.now()

    try:
        result = pipeline.process_documents_batch(SAMPLE_DOCUMENTS)
        processing_time = (datetime.now() - start_time).total_seconds()

        print(f"\n✓ Processing complete in {processing_time:.2f}s")
        print(f"\nResults:")
        print(f"  - Documents processed: {result.documents_processed}")
        print(f"  - Total entities extracted: {result.total_entities}")
        print(f"  - Unique entities: {result.unique_entities}")
        print(f"  - Total relationships: {result.total_relationships}")
        print(f"  - Nodes created: {result.nodes_created}")
        print(f"  - Edges created: {result.edges_created}")
        print(f"  - Processing time: {result.processing_time:.2f}s")

        if result.errors:
            print(f"\nErrors encountered: {len(result.errors)}")
            for error in result.errors[:5]:  # Show first 5 errors
                print(f"  - {error}")

    except Exception as e:
        print(f"\n✗ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Query the graph
    print_section("Querying Knowledge Graph")

    query_interface = GraphQueryInterface(neo4j_manager=pipeline.neo4j_manager)

    # 1. Get entity statistics
    print("1. Entity Statistics:")
    entity_stats = query_interface.get_entity_statistics()
    print_statistics(entity_stats, indent=1)

    # 2. Get relationship statistics
    print("\n2. Relationship Statistics:")
    rel_stats = query_interface.get_relationship_statistics()
    print_statistics(rel_stats, indent=1)

    # 3. Search for specific entities
    print("\n3. Searching for AI-related entities:")
    ai_entities = query_interface.find_entity("OpenAI", fuzzy=True)
    print(f"   Found {ai_entities.count} matches:")
    for entity in ai_entities.results[:5]:
        print(f"     - {entity['name']} ({entity['type']}) - confidence: {entity.get('confidence', 'N/A')}")

    # 4. Find most connected entities
    print("\n4. Most Connected Entities:")
    connected = query_interface.get_most_connected_entities(limit=10)
    print(f"   Top {min(10, connected.count)} entities by connections:")
    for entity in connected.results[:10]:
        print(f"     - {entity['name']} ({entity['type']}) - {entity['degree']} connections")

    # 5. Find entities by type
    print("\n5. Organizations in the Graph:")
    orgs = query_interface.search_entities(entity_type="ORGANIZATION", limit=10)
    if orgs.count > 0:
        for org in orgs.results:
            print(f"     - {org['name']}")
    else:
        print("     No organizations found")

    # 6. Find relationships between entities
    print("\n6. Finding Relationships:")
    try:
        rels = query_interface.find_relationships("OpenAI")
        if rels.count > 0:
            print(f"   Found {rels.count} relationships involving OpenAI:")
            for rel in rels.results[:5]:
                print(f"     - {rel['source']} --[{rel['relationship']}]--> {rel['target']}")
        else:
            print("   No relationships found")
    except Exception as e:
        print(f"   Could not find relationships: {e}")

    # 7. Document queries
    print("\n7. Document Queries:")
    doc_entities = query_interface.get_document_entities("ai_companies")
    print(f"   Entities in document 'ai_companies': {doc_entities.count}")
    for entity in doc_entities.results[:10]:
        print(f"     - {entity['name']} ({entity['type']})")

    # 8. Find similar documents
    print("\n8. Similar Documents:")
    similar = query_interface.find_similar_documents("ai_companies", min_shared_entities=2, limit=5)
    if similar.count > 0:
        print(f"   Documents similar to 'ai_companies':")
        for doc in similar.results:
            print(f"     - {doc['doc_id']} ({doc.get('title', 'N/A')}) - {doc['shared_entities']} shared entities")
    else:
        print("   No similar documents found")

    # Get overall pipeline statistics
    print_section("Pipeline Statistics")
    pipeline_stats = pipeline.get_pipeline_statistics()
    print_statistics(pipeline_stats)

    # Export results
    print_section("Exporting Results")

    output_dir = Path(__file__).parent.parent / "data" / "kg_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export pipeline result
    result_file = output_dir / f"kg_build_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(result_file, 'w') as f:
        json.dump({
            "documents_processed": result.documents_processed,
            "total_entities": result.total_entities,
            "unique_entities": result.unique_entities,
            "total_relationships": result.total_relationships,
            "nodes_created": result.nodes_created,
            "edges_created": result.edges_created,
            "processing_time": result.processing_time,
            "statistics": result.statistics,
            "errors": result.errors
        }, f, indent=2)

    print(f"✓ Results exported to: {result_file}")

    # Export entity statistics
    stats_file = output_dir / f"entity_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(stats_file, 'w') as f:
        json.dump({
            "entity_stats": entity_stats,
            "relationship_stats": rel_stats
        }, f, indent=2)

    print(f"✓ Statistics exported to: {stats_file}")

    print_section("Demo Complete")
    print("Knowledge graph construction completed successfully!")
    print("\nNext steps:")
    print("  1. Open Neo4j Browser: http://localhost:7474")
    print("  2. Run Cypher queries to explore the graph")
    print("  3. Visualize entity relationships")
    print("\nExample Cypher queries:")
    print("  MATCH (n) RETURN n LIMIT 25")
    print("  MATCH (e:Entity)-[r]-(other:Entity) RETURN e, r, other LIMIT 50")
    print("  MATCH (d:Document)-[:CONTAINS]->(e:Entity) RETURN d, e LIMIT 50")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
