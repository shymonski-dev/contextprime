"""
Comprehensive tests for the community detection module.

Tests cover:
- Community detection algorithms
- Graph analysis
- Community summarization
- Document clustering
- Cross-document analysis
- Global query handling
- Storage and retrieval
- Pipeline integration
"""

import pytest
import networkx as nx
import numpy as np
from pathlib import Path
import tempfile
import os

from src.community.community_detector import (
    CommunityDetector,
    CommunityAlgorithm,
    CommunityResult
)
from src.community.graph_analyzer import GraphAnalyzer, GraphMetrics
from src.community.community_summarizer import CommunitySummarizer
from src.community.document_clusterer import DocumentClusterer, ClusteringMethod
from src.community.cross_document_analyzer import CrossDocumentAnalyzer
from src.community.global_query_handler import GlobalQueryHandler
from src.community.community_visualizer import CommunityVisualizer


class TestCommunityDetector:
    """Test community detection algorithms."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph with clear community structure."""
        G = nx.Graph()

        # Community 1: nodes 0-4
        for i in range(5):
            for j in range(i+1, 5):
                G.add_edge(f"node_{i}", f"node_{j}")

        # Community 2: nodes 5-9
        for i in range(5, 10):
            for j in range(i+1, 10):
                G.add_edge(f"node_{i}", f"node_{j}")

        # Community 3: nodes 10-14
        for i in range(10, 15):
            for j in range(i+1, 15):
                G.add_edge(f"node_{i}", f"node_{j}")

        # Add some inter-community edges
        G.add_edge("node_4", "node_5")
        G.add_edge("node_9", "node_10")

        return G

    def test_louvain_detection(self, sample_graph):
        """Test Louvain community detection."""
        detector = CommunityDetector(algorithm=CommunityAlgorithm.LOUVAIN)

        try:
            result = detector.detect_communities(sample_graph)

            assert isinstance(result, CommunityResult)
            assert result.num_communities > 0
            assert result.num_communities <= sample_graph.number_of_nodes()
            assert result.modularity >= 0
            assert len(result.node_to_community) == sample_graph.number_of_nodes()

            print(f"Louvain detected {result.num_communities} communities")
            print(f"Modularity: {result.modularity:.3f}")

        except ImportError:
            pytest.skip("python-louvain not installed")

    def test_label_propagation(self, sample_graph):
        """Test label propagation algorithm."""
        detector = CommunityDetector(algorithm=CommunityAlgorithm.LABEL_PROPAGATION)

        result = detector.detect_communities(sample_graph)

        assert isinstance(result, CommunityResult)
        assert result.num_communities > 0
        assert len(result.communities) == result.num_communities

        print(f"Label propagation detected {result.num_communities} communities")

    def test_spectral_clustering(self, sample_graph):
        """Test spectral clustering."""
        detector = CommunityDetector(algorithm=CommunityAlgorithm.SPECTRAL)

        result = detector.detect_communities(sample_graph)

        assert isinstance(result, CommunityResult)
        assert result.num_communities > 0

        print(f"Spectral clustering detected {result.num_communities} communities")

    def test_auto_selection(self, sample_graph):
        """Test automatic algorithm selection."""
        detector = CommunityDetector(algorithm=CommunityAlgorithm.AUTO)

        result = detector.detect_communities(sample_graph)

        assert isinstance(result, CommunityResult)
        assert result.algorithm in ["louvain", "leiden", "label_propagation", "spectral"]

        print(f"Auto-selected algorithm: {result.algorithm}")

    def test_min_community_size(self, sample_graph):
        """Test filtering by minimum community size."""
        detector = CommunityDetector(min_community_size=3)

        result = detector.detect_communities(sample_graph)

        # All communities should have at least 3 members
        for comm_id, members in result.communities.items():
            assert len(members) >= 3

    def test_compare_algorithms(self, sample_graph):
        """Test algorithm comparison."""
        detector = CommunityDetector()

        results = detector.compare_algorithms(sample_graph)

        assert isinstance(results, dict)
        assert len(results) > 0

        for algo_name, result in results.items():
            print(f"{algo_name}: {result.num_communities} communities, "
                  f"modularity={result.modularity:.3f}")


class TestGraphAnalyzer:
    """Test graph analysis functionality."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph."""
        G = nx.karate_club_graph()
        return G

    def test_analyze_graph(self, sample_graph):
        """Test graph-level analysis."""
        analyzer = GraphAnalyzer()

        metrics = analyzer.analyze_graph(sample_graph)

        assert isinstance(metrics, GraphMetrics)
        assert metrics.num_nodes == sample_graph.number_of_nodes()
        assert metrics.num_edges == sample_graph.number_of_edges()
        assert 0 <= metrics.density <= 1
        assert metrics.avg_degree > 0
        assert 0 <= metrics.avg_clustering_coefficient <= 1

        print(f"Graph metrics:")
        print(f"  Nodes: {metrics.num_nodes}")
        print(f"  Edges: {metrics.num_edges}")
        print(f"  Density: {metrics.density:.3f}")
        print(f"  Avg degree: {metrics.avg_degree:.2f}")

    def test_compute_node_metrics(self, sample_graph):
        """Test node-level metrics computation."""
        analyzer = GraphAnalyzer()

        node_metrics = analyzer.compute_node_metrics(sample_graph)

        assert isinstance(node_metrics, dict)
        assert len(node_metrics) == sample_graph.number_of_nodes()

        # Check metrics for first node
        first_node = list(node_metrics.keys())[0]
        metrics = node_metrics[first_node]

        assert metrics.degree >= 0
        assert 0 <= metrics.degree_centrality <= 1
        assert 0 <= metrics.pagerank <= 1

        print(f"Sample node metrics for {first_node}:")
        print(f"  Degree: {metrics.degree}")
        print(f"  PageRank: {metrics.pagerank:.4f}")
        print(f"  Betweenness: {metrics.betweenness_centrality:.4f}")

    def test_analyze_communities(self, sample_graph):
        """Test community quality analysis."""
        analyzer = GraphAnalyzer()
        detector = CommunityDetector()

        community_result = detector.detect_communities(sample_graph)
        community_metrics = analyzer.analyze_communities(sample_graph, community_result)

        assert isinstance(community_metrics, dict)
        assert len(community_metrics) == community_result.num_communities

        # Check metrics for each community
        for comm_id, metrics in community_metrics.items():
            assert metrics.size > 0
            assert 0 <= metrics.density <= 1
            assert 0 <= metrics.conductance <= 1

        print(f"Community metrics for {len(community_metrics)} communities:")
        for comm_id, metrics in list(community_metrics.items())[:3]:
            print(f"  Community {comm_id}: size={metrics.size}, "
                  f"density={metrics.density:.3f}, conductance={metrics.conductance:.3f}")

    def test_identify_bridge_nodes(self, sample_graph):
        """Test bridge node identification."""
        analyzer = GraphAnalyzer()

        bridges = analyzer.identify_bridge_nodes(sample_graph)

        assert isinstance(bridges, set)
        print(f"Found {len(bridges)} bridge nodes")

    def test_hub_nodes(self, sample_graph):
        """Test hub node identification."""
        analyzer = GraphAnalyzer()

        hubs = analyzer.get_hub_nodes(sample_graph, top_k=5)

        assert isinstance(hubs, list)
        assert len(hubs) <= 5

        print("Top 5 hub nodes:")
        for node, score in hubs:
            print(f"  {node}: {score:.4f}")


class TestDocumentClusterer:
    """Test document clustering functionality."""

    @pytest.fixture
    def sample_doc_entities(self):
        """Create sample document-entity mappings."""
        return {
            "doc_1": {"entity_A", "entity_B", "entity_C"},
            "doc_2": {"entity_A", "entity_B", "entity_D"},
            "doc_3": {"entity_E", "entity_F", "entity_G"},
            "doc_4": {"entity_E", "entity_F", "entity_H"},
            "doc_5": {"entity_I", "entity_J", "entity_K"}
        }

    @pytest.fixture
    def sample_doc_embeddings(self):
        """Create sample document embeddings."""
        np.random.seed(42)
        embeddings = {}

        # Create 3 clusters of documents
        for i in range(5):
            embeddings[f"doc_{i}"] = np.random.randn(128) + np.array([1, 0] * 64)
        for i in range(5, 10):
            embeddings[f"doc_{i}"] = np.random.randn(128) + np.array([0, 1] * 64)
        for i in range(10, 15):
            embeddings[f"doc_{i}"] = np.random.randn(128) + np.array([-1, 0] * 64)

        # Normalize
        for doc_id in embeddings:
            embeddings[doc_id] = embeddings[doc_id] / np.linalg.norm(embeddings[doc_id])

        return embeddings

    def test_entity_based_clustering(self, sample_doc_entities):
        """Test entity-based document clustering."""
        clusterer = DocumentClusterer()

        result = clusterer.cluster_by_entities(sample_doc_entities)

        assert result.num_clusters > 0
        assert len(result.doc_to_cluster) == len(sample_doc_entities)

        print(f"Entity-based clustering: {result.num_clusters} clusters")
        for cluster_id, cluster in result.clusters.items():
            print(f"  Cluster {cluster_id}: {len(cluster.document_ids)} docs, "
                  f"{len(cluster.shared_entities)} shared entities")

    def test_semantic_clustering(self, sample_doc_embeddings):
        """Test semantic document clustering."""
        clusterer = DocumentClusterer()

        result = clusterer.cluster_by_semantics(sample_doc_embeddings, n_clusters=3)

        assert result.num_clusters == 3
        assert result.silhouette_score is not None

        print(f"Semantic clustering: {result.num_clusters} clusters")
        print(f"Silhouette score: {result.silhouette_score:.3f}")

    def test_hierarchical_clustering(self, sample_doc_embeddings):
        """Test hierarchical clustering."""
        clusterer = DocumentClusterer()

        result = clusterer.hierarchical_cluster(
            sample_doc_embeddings,
            linkage='average',
            distance_threshold=0.5
        )

        assert result.num_clusters > 0
        print(f"Hierarchical clustering: {result.num_clusters} clusters")

    def test_hybrid_clustering(self, sample_doc_embeddings, sample_doc_entities):
        """Test hybrid clustering."""
        clusterer = DocumentClusterer()

        # Use subset that overlaps
        embeddings_subset = {k: v for k, v in list(sample_doc_embeddings.items())[:5]}

        result = clusterer.hybrid_cluster(
            embeddings_subset,
            sample_doc_entities,
            entity_weight=0.5
        )

        assert result.num_clusters > 0
        print(f"Hybrid clustering: {result.num_clusters} clusters")


class TestCrossDocumentAnalyzer:
    """Test cross-document analysis."""

    @pytest.fixture
    def sample_data(self):
        """Create sample cross-document data."""
        doc_entities = {
            "doc_1": {"Alice", "Bob", "Project_X"},
            "doc_2": {"Bob", "Charlie", "Project_X"},
            "doc_3": {"Charlie", "Diana", "Project_Y"}
        }

        doc_embeddings = {
            f"doc_{i}": np.random.randn(128) / 10 + np.random.randn(128)
            for i in range(1, 4)
        }

        return doc_entities, doc_embeddings

    def test_cooccurrence_analysis(self, sample_data):
        """Test entity co-occurrence analysis."""
        doc_entities, _ = sample_data
        analyzer = CrossDocumentAnalyzer()

        patterns = analyzer.analyze_entity_cooccurrence(doc_entities, top_k=10)

        assert isinstance(patterns, list)
        assert len(patterns) > 0

        print(f"Found {len(patterns)} co-occurrence patterns:")
        for pattern in patterns[:5]:
            print(f"  {pattern.entity_pair}: {pattern.cooccurrence_count} times")

    def test_cooccurrence_graph(self, sample_data):
        """Test co-occurrence graph building."""
        doc_entities, _ = sample_data
        analyzer = CrossDocumentAnalyzer()

        graph = analyzer.build_cooccurrence_graph(doc_entities, min_cooccurrence=1)

        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

        print(f"Co-occurrence graph: {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges")

    def test_document_similarity(self, sample_data):
        """Test document similarity computation."""
        doc_entities, doc_embeddings = sample_data
        analyzer = CrossDocumentAnalyzer()

        similarity = analyzer.compute_document_similarity(
            "doc_1",
            "doc_2",
            doc_embeddings,
            doc_entities
        )

        assert 0 <= similarity.combined_similarity <= 1
        assert 0 <= similarity.entity_overlap <= 1

        print(f"Document similarity: {similarity.combined_similarity:.3f}")
        print(f"  Semantic: {similarity.semantic_similarity:.3f}")
        print(f"  Entity overlap: {similarity.entity_overlap:.3f}")


class TestGlobalQueryHandler:
    """Test global query handling."""

    @pytest.fixture
    def sample_setup(self):
        """Create sample setup for query testing."""
        graph = nx.karate_club_graph()
        detector = CommunityDetector()
        summarizer = CommunitySummarizer()

        community_result = detector.detect_communities(graph)
        community_summaries = summarizer.summarize_all_communities(
            graph,
            community_result,
            include_detailed=False
        )
        global_summary = summarizer.generate_global_summary(
            graph,
            community_result,
            community_summaries
        )

        return graph, community_result, community_summaries, global_summary

    def test_theme_query(self, sample_setup):
        """Test theme-based queries."""
        graph, community_result, community_summaries, global_summary = sample_setup
        handler = GlobalQueryHandler()

        response = handler.answer_query(
            "What are the main themes?",
            graph,
            community_result,
            community_summaries,
            global_summary
        )

        assert response.answer is not None
        assert len(response.answer) > 0
        assert response.confidence > 0

        print(f"Query: What are the main themes?")
        print(f"Answer: {response.answer[:200]}...")
        print(f"Confidence: {response.confidence:.2f}")

    def test_community_query(self, sample_setup):
        """Test community-based queries."""
        graph, community_result, community_summaries, global_summary = sample_setup
        handler = GlobalQueryHandler()

        response = handler.answer_query(
            "How many communities exist?",
            graph,
            community_result,
            community_summaries,
            global_summary
        )

        assert response.answer is not None
        assert str(community_result.num_communities) in response.answer

        print(f"Query: How many communities exist?")
        print(f"Answer: {response.answer[:200]}...")

    def test_structure_query(self, sample_setup):
        """Test structure-based queries."""
        graph, community_result, community_summaries, global_summary = sample_setup
        handler = GlobalQueryHandler()

        response = handler.answer_query(
            "How is the graph structured?",
            graph,
            community_result,
            community_summaries,
            global_summary
        )

        assert response.answer is not None
        assert response.confidence > 0.8  # Structure queries should be high confidence

        print(f"Query: How is the graph structured?")
        print(f"Answer: {response.answer[:200]}...")


class TestCommunityVisualizer:
    """Test visualization functionality."""

    @pytest.fixture
    def sample_graph(self):
        """Create a sample graph."""
        return nx.karate_club_graph()

    @pytest.fixture
    def sample_result(self, sample_graph):
        """Create sample community detection result."""
        detector = CommunityDetector()
        return detector.detect_communities(sample_graph)

    def test_static_visualization(self, sample_graph, sample_result):
        """Test static visualization creation."""
        visualizer = CommunityVisualizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "communities.png")

            visualizer.visualize_communities(
                sample_graph,
                sample_result,
                output_path,
                layout="spring"
            )

            assert os.path.exists(output_path)
            print(f"Static visualization created at {output_path}")

    def test_export_graphml(self, sample_graph, sample_result):
        """Test GraphML export."""
        visualizer = CommunityVisualizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "graph.graphml")

            visualizer.export_to_graphml(
                sample_graph,
                sample_result,
                output_path
            )

            assert os.path.exists(output_path)
            print(f"GraphML exported to {output_path}")

    def test_export_d3_json(self, sample_graph, sample_result):
        """Test D3 JSON export."""
        visualizer = CommunityVisualizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "graph.json")

            visualizer.export_to_d3_json(
                sample_graph,
                sample_result,
                output_path
            )

            assert os.path.exists(output_path)

            # Verify JSON structure
            import json
            with open(output_path) as f:
                data = json.load(f)

            assert "nodes" in data
            assert "links" in data
            assert "metadata" in data

            print(f"D3 JSON exported to {output_path}")

    def test_size_distribution(self, sample_result):
        """Test community size distribution plot."""
        visualizer = CommunityVisualizer()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "distribution.png")

            visualizer.create_community_size_distribution(
                sample_result,
                output_path
            )

            assert os.path.exists(output_path)
            print(f"Size distribution plot created at {output_path}")


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_end_to_end_workflow(self):
        """Test complete end-to-end community detection workflow."""
        # Create sample graph
        graph = nx.karate_club_graph()

        # Detect communities
        detector = CommunityDetector()
        community_result = detector.detect_communities(graph)

        # Analyze
        analyzer = GraphAnalyzer()
        graph_metrics = analyzer.analyze_graph(graph)
        community_metrics = analyzer.analyze_communities(graph, community_result)

        # Summarize
        summarizer = CommunitySummarizer()
        community_summaries = summarizer.summarize_all_communities(
            graph,
            community_result
        )
        global_summary = summarizer.generate_global_summary(
            graph,
            community_result,
            community_summaries
        )

        # Visualize
        visualizer = CommunityVisualizer()
        with tempfile.TemporaryDirectory() as tmpdir:
            visualizer.visualize_communities(
                graph,
                community_result,
                os.path.join(tmpdir, "communities.png")
            )

        # Verify all components worked
        assert community_result.num_communities > 0
        assert graph_metrics.num_nodes == graph.number_of_nodes()
        assert len(community_metrics) == community_result.num_communities
        assert len(community_summaries) == community_result.num_communities
        assert global_summary.num_communities == community_result.num_communities

        print("\nEnd-to-end workflow completed successfully!")
        print(f"Detected {community_result.num_communities} communities")
        print(f"Modularity: {community_result.modularity:.3f}")
        print(f"Main themes: {global_summary.main_themes[:5]}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
