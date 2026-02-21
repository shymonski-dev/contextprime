"""
Demonstration script for the community detection system.

Shows:
- Community detection on sample knowledge graph
- Graph analysis and metrics
- Community summarization
- Global query answering
- Visualization
- Algorithm comparison
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
import numpy as np
from loguru import logger

from src.community.community_detector import CommunityDetector, CommunityAlgorithm
from src.community.graph_analyzer import GraphAnalyzer
from src.community.community_summarizer import CommunitySummarizer
from src.community.document_clusterer import DocumentClusterer
from src.community.cross_document_analyzer import CrossDocumentAnalyzer
from src.community.global_query_handler import GlobalQueryHandler
from src.community.community_pipeline import CommunityPipeline
from src.community.community_visualizer import CommunityVisualizer


def create_sample_knowledge_graph():
    """Create a sample knowledge graph with multiple communities."""
    logger.info("Creating sample knowledge graph...")

    G = nx.Graph()

    # Community 1: Machine Learning Research
    ml_entities = [
        "Neural Networks", "Deep Learning", "Gradient Descent",
        "Backpropagation", "CNN", "RNN", "Transformer", "BERT"
    ]

    for i, entity1 in enumerate(ml_entities):
        G.add_node(entity1, type="concept", domain="ML")
        for entity2 in ml_entities[i+1:]:
            if np.random.rand() > 0.3:  # 70% connection probability
                G.add_edge(entity1, entity2, type="RELATED_TO", weight=np.random.rand())

    # Community 2: Natural Language Processing
    nlp_entities = [
        "Tokenization", "Named Entity Recognition", "POS Tagging",
        "Sentiment Analysis", "Text Classification", "Language Model"
    ]

    for i, entity1 in enumerate(nlp_entities):
        G.add_node(entity1, type="concept", domain="NLP")
        for entity2 in nlp_entities[i+1:]:
            if np.random.rand() > 0.3:
                G.add_edge(entity1, entity2, type="RELATED_TO", weight=np.random.rand())

    # Community 3: Computer Vision
    cv_entities = [
        "Image Classification", "Object Detection", "Segmentation",
        "Face Recognition", "Image Preprocessing", "Feature Extraction"
    ]

    for i, entity1 in enumerate(cv_entities):
        G.add_node(entity1, type="concept", domain="CV")
        for entity2 in cv_entities[i+1:]:
            if np.random.rand() > 0.3:
                G.add_edge(entity1, entity2, type="RELATED_TO", weight=np.random.rand())

    # Community 4: Data Science
    ds_entities = [
        "Data Cleaning", "Feature Engineering", "Model Evaluation",
        "Cross Validation", "Hyperparameter Tuning", "Data Visualization"
    ]

    for i, entity1 in enumerate(ds_entities):
        G.add_node(entity1, type="concept", domain="DS")
        for entity2 in ds_entities[i+1:]:
            if np.random.rand() > 0.3:
                G.add_edge(entity1, entity2, type="RELATED_TO", weight=np.random.rand())

    # Add cross-community edges
    cross_edges = [
        ("Neural Networks", "Image Classification"),
        ("Deep Learning", "NLP"),
        ("Transformer", "Language Model"),
        ("CNN", "Object Detection"),
        ("Feature Engineering", "Deep Learning"),
        ("Model Evaluation", "Neural Networks"),
        ("Text Classification", "Sentiment Analysis"),
        ("BERT", "Named Entity Recognition")
    ]

    for entity1, entity2 in cross_edges:
        if entity1 in G and entity2 in G:
            G.add_edge(entity1, entity2, type="APPLIES_TO", weight=0.8)

    logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    return G


def demo_basic_detection(graph):
    """Demonstrate basic community detection."""
    print("\n" + "="*80)
    print("DEMO 1: Basic Community Detection")
    print("="*80)

    detector = CommunityDetector(algorithm=CommunityAlgorithm.AUTO)

    result = detector.detect_communities(graph)

    print(f"\nDetected {result.num_communities} communities using {result.algorithm}")
    print(f"Modularity score: {result.modularity:.4f}")
    print(f"Execution time: {result.execution_time:.2f}s")

    print("\nCommunity sizes:")
    for comm_id, members in sorted(result.communities.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  Community {comm_id}: {len(members)} members")
        print(f"    Sample members: {list(members)[:5]}")

    return result


def demo_graph_analysis(graph, community_result):
    """Demonstrate graph analysis."""
    print("\n" + "="*80)
    print("DEMO 2: Graph Analysis")
    print("="*80)

    analyzer = GraphAnalyzer()

    # Overall graph metrics
    print("\n--- Graph-Level Metrics ---")
    graph_metrics = analyzer.analyze_graph(graph)

    print(f"Nodes: {graph_metrics.num_nodes}")
    print(f"Edges: {graph_metrics.num_edges}")
    print(f"Density: {graph_metrics.density:.4f}")
    print(f"Average degree: {graph_metrics.avg_degree:.2f}")
    print(f"Clustering coefficient: {graph_metrics.avg_clustering_coefficient:.4f}")
    print(f"Connected components: {graph_metrics.num_connected_components}")

    # Community quality metrics
    print("\n--- Community Quality Metrics ---")
    community_metrics = analyzer.analyze_communities(graph, community_result)

    for comm_id in sorted(community_metrics.keys())[:3]:
        metrics = community_metrics[comm_id]
        print(f"\nCommunity {comm_id}:")
        print(f"  Size: {metrics.size}")
        print(f"  Density: {metrics.density:.4f}")
        print(f"  Conductance: {metrics.conductance:.4f}")
        print(f"  Top nodes: {[node for node, score in metrics.top_nodes[:3]]}")

    # Hub nodes
    print("\n--- Top Hub Nodes (PageRank) ---")
    hubs = analyzer.get_hub_nodes(graph, top_k=5)
    for node, score in hubs:
        print(f"  {node}: {score:.4f}")

    return graph_metrics, community_metrics


def demo_summarization(graph, community_result):
    """Demonstrate community summarization."""
    print("\n" + "="*80)
    print("DEMO 3: Community Summarization")
    print("="*80)

    # Note: Without OpenAI API key, this will use rule-based summarization
    summarizer = CommunitySummarizer()

    print("\nGenerating community summaries...")
    community_summaries = summarizer.summarize_all_communities(
        graph,
        community_result,
        include_detailed=False
    )

    print(f"\nGenerated summaries for {len(community_summaries)} communities")

    for comm_id in sorted(community_summaries.keys())[:3]:
        summary = community_summaries[comm_id]
        print(f"\n--- {summary.title} ---")
        print(f"Size: {summary.size} entities")
        print(f"Summary: {summary.brief_summary}")
        print(f"Themes: {', '.join(summary.themes[:5])}")
        print(f"Top entities: {[entity for entity, score in summary.key_entities[:5]]}")

    # Global summary
    print("\n--- Global Summary ---")
    global_summary = summarizer.generate_global_summary(
        graph,
        community_result,
        community_summaries
    )

    print(f"Total communities: {global_summary.num_communities}")
    print(f"Main themes: {', '.join(global_summary.main_themes[:10])}")
    print(f"\nOverall structure:\n{global_summary.overall_structure}")

    if global_summary.key_insights:
        print(f"\nKey insights:")
        for insight in global_summary.key_insights:
            print(f"  - {insight}")

    return community_summaries, global_summary


def demo_document_clustering():
    """Demonstrate document clustering."""
    print("\n" + "="*80)
    print("DEMO 4: Document Clustering")
    print("="*80)

    # Create sample documents with entities
    doc_entities = {
        "paper_1": {"Neural Networks", "Deep Learning", "CNN"},
        "paper_2": {"Deep Learning", "RNN", "BERT"},
        "paper_3": {"Tokenization", "Named Entity Recognition", "NLP"},
        "paper_4": {"Sentiment Analysis", "Text Classification"},
        "paper_5": {"Image Classification", "Object Detection", "CNN"},
        "paper_6": {"Face Recognition", "Feature Extraction"},
        "paper_7": {"Data Cleaning", "Feature Engineering"},
        "paper_8": {"Model Evaluation", "Cross Validation"}
    }

    clusterer = DocumentClusterer()

    print("\n--- Entity-Based Clustering ---")
    result = clusterer.cluster_by_entities(doc_entities, similarity_threshold=0.2)

    print(f"Detected {result.num_clusters} document clusters")
    for cluster_id, cluster in result.clusters.items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Documents: {cluster.document_ids}")
        print(f"  Shared entities: {cluster.shared_entities}")

    return result


def demo_cross_document_analysis():
    """Demonstrate cross-document analysis."""
    print("\n" + "="*80)
    print("DEMO 5: Cross-Document Analysis")
    print("="*80)

    doc_entities = {
        "doc_1": {"Neural Networks", "Deep Learning", "Training"},
        "doc_2": {"Deep Learning", "CNN", "Training"},
        "doc_3": {"CNN", "Image Classification", "Training"},
        "doc_4": {"NLP", "Text Classification", "BERT"},
        "doc_5": {"BERT", "Transformer", "NLP"}
    }

    analyzer = CrossDocumentAnalyzer()

    print("\n--- Entity Co-occurrence Patterns ---")
    patterns = analyzer.analyze_entity_cooccurrence(doc_entities, top_k=10)

    print(f"Found {len(patterns)} co-occurrence patterns:")
    for pattern in patterns[:5]:
        print(f"  {pattern.entity_pair[0]} <-> {pattern.entity_pair[1]}: "
              f"{pattern.cooccurrence_count} times (confidence: {pattern.confidence:.2f})")

    print("\n--- Co-occurrence Graph ---")
    graph = analyzer.build_cooccurrence_graph(doc_entities, min_cooccurrence=2)
    print(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    return patterns, graph


def demo_global_queries(graph, community_result, community_summaries, global_summary):
    """Demonstrate global query handling."""
    print("\n" + "="*80)
    print("DEMO 6: Global Query Handling")
    print("="*80)

    handler = GlobalQueryHandler()

    queries = [
        "What are the main themes?",
        "How many communities exist?",
        "How is the graph structured?",
        "What topics are related?"
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")

        response = handler.answer_query(
            query,
            graph,
            community_result,
            community_summaries,
            global_summary
        )

        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Relevant communities: {response.relevant_communities[:5]}")


def demo_algorithm_comparison(graph):
    """Demonstrate algorithm comparison."""
    print("\n" + "="*80)
    print("DEMO 7: Algorithm Comparison")
    print("="*80)

    detector = CommunityDetector()

    algorithms = [
        CommunityAlgorithm.LOUVAIN,
        CommunityAlgorithm.LABEL_PROPAGATION,
        CommunityAlgorithm.SPECTRAL
    ]

    print("\nComparing community detection algorithms...")
    results = detector.compare_algorithms(graph, algorithms)

    print(f"\nComparison results:")
    print(f"{'Algorithm':<20} {'Communities':<12} {'Modularity':<12} {'Time (s)':<10}")
    print("-" * 60)

    for algo_name, result in results.items():
        print(f"{algo_name:<20} {result.num_communities:<12} "
              f"{result.modularity:<12.4f} {result.execution_time:<10.2f}")

    return results


def demo_visualization(graph, community_result):
    """Demonstrate visualization."""
    print("\n" + "="*80)
    print("DEMO 8: Visualization")
    print("="*80)

    visualizer = CommunityVisualizer()

    output_dir = Path(__file__).parent.parent / "data" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Static visualization
    print("\nCreating static visualization...")
    visualizer.visualize_communities(
        graph,
        community_result,
        str(output_dir / "communities_static.png"),
        layout="spring",
        show_labels=True
    )
    print(f"Saved to: {output_dir / 'communities_static.png'}")

    # Interactive visualization
    print("\nCreating interactive visualization...")
    try:
        visualizer.create_interactive_visualization(
            graph,
            community_result,
            str(output_dir / "communities_interactive.html")
        )
        print(f"Saved to: {output_dir / 'communities_interactive.html'}")
    except Exception as e:
        print(f"Interactive visualization failed: {e}")

    # Export formats
    print("\nExporting to various formats...")
    visualizer.export_to_graphml(
        graph,
        community_result,
        str(output_dir / "communities.graphml")
    )
    print(f"GraphML saved to: {output_dir / 'communities.graphml'}")

    visualizer.export_to_d3_json(
        graph,
        community_result,
        str(output_dir / "communities.json")
    )
    print(f"D3 JSON saved to: {output_dir / 'communities.json'}")

    # Size distribution
    print("\nCreating size distribution plot...")
    visualizer.create_community_size_distribution(
        community_result,
        str(output_dir / "size_distribution.png")
    )
    print(f"Saved to: {output_dir / 'size_distribution.png'}")


def demo_pipeline():
    """Demonstrate the complete pipeline."""
    print("\n" + "="*80)
    print("DEMO 9: Complete Pipeline")
    print("="*80)

    # Create sample graph
    graph = create_sample_knowledge_graph()

    # Initialize pipeline
    pipeline = CommunityPipeline(
        algorithm=CommunityAlgorithm.AUTO,
        enable_storage=False,  # Disable storage for demo
        enable_summaries=True
    )

    print("\nRunning complete pipeline...")
    result = pipeline.run(
        graph=graph,
        load_from_neo4j=False,
        store_results=False
    )

    print(f"\n--- Pipeline Results ---")
    print(f"Execution time: {result.execution_time:.2f}s")
    print(f"Communities detected: {result.community_result.num_communities}")
    print(f"Modularity: {result.community_result.modularity:.4f}")
    print(f"Algorithm used: {result.community_result.algorithm}")

    print(f"\nGraph metrics:")
    print(f"  Nodes: {result.graph_metrics.num_nodes}")
    print(f"  Edges: {result.graph_metrics.num_edges}")
    print(f"  Density: {result.graph_metrics.density:.4f}")

    if result.global_summary:
        print(f"\nMain themes: {', '.join(result.global_summary.main_themes[:5])}")

    # Test global query
    print("\n--- Testing Global Query ---")
    response = pipeline.answer_global_query(
        "What are the main themes in this knowledge graph?",
        graph=graph
    )
    print(f"Query: What are the main themes in this knowledge graph?")
    print(f"Answer: {response.answer}")

    return result


def main():
    """Run all demonstrations."""
    logger.info("Starting Community Detection System Demo")

    print("\n" + "="*80)
    print("Contextprime - Community Detection System Demo")
    print("="*80)

    # Create sample graph
    graph = create_sample_knowledge_graph()

    # Run demonstrations
    community_result = demo_basic_detection(graph)
    graph_metrics, community_metrics = demo_graph_analysis(graph, community_result)
    community_summaries, global_summary = demo_summarization(graph, community_result)
    doc_clustering_result = demo_document_clustering()
    cooccurrence_patterns, cooccurrence_graph = demo_cross_document_analysis()
    demo_global_queries(graph, community_result, community_summaries, global_summary)
    algorithm_results = demo_algorithm_comparison(graph)
    demo_visualization(graph, community_result)
    pipeline_result = demo_pipeline()

    print("\n" + "="*80)
    print("Demo completed successfully!")
    print("="*80)

    print("\nSummary:")
    print(f"  - Detected {community_result.num_communities} communities")
    print(f"  - Modularity: {community_result.modularity:.4f}")
    print(f"  - Generated {len(community_summaries)} community summaries")
    print(f"  - Found {len(cooccurrence_patterns)} co-occurrence patterns")
    print(f"  - Compared {len(algorithm_results)} algorithms")

    print("\nCheck the data/visualizations directory for generated visualizations!")


if __name__ == "__main__":
    main()
