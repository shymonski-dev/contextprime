"""
Comprehensive tests for RAPTOR summarization system.
"""

import pytest
import numpy as np
from typing import List, Dict, Any

from src.summarization.cluster_manager import (
    ClusterManager, ClusteringMethod, Cluster, ClusteringResult
)
from src.summarization.summary_generator import (
    SummaryGenerator, SummaryLevel, Summary
)
from src.summarization.tree_builder import TreeBuilder, TreeNode, TreeStats
from src.summarization.tree_visualizer import TreeVisualizer


# Test Fixtures

@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    np.random.seed(42)
    # Create 3 clusters of 10 points each
    embeddings = []

    # Cluster 1: around [0, 0]
    embeddings.append(np.random.randn(10, 128) * 0.1)

    # Cluster 2: around [5, 5]
    embeddings.append(np.random.randn(10, 128) * 0.1 + 5)

    # Cluster 3: around [-5, -5]
    embeddings.append(np.random.randn(10, 128) * 0.1 - 5)

    return np.vstack(embeddings)


@pytest.fixture
def sample_chunks():
    """Generate sample chunks for testing."""
    chunks = []
    for i in range(30):
        chunks.append({
            'content': f"This is sample chunk {i} with some content about topic {i // 10}. "
                      f"It contains information that is relevant to the document.",
            'chunk_id': f"chunk_{i:04d}",
            'metadata': {
                'index': i,
                'topic': i // 10
            }
        })
    return chunks


@pytest.fixture
def cluster_manager():
    """Create cluster manager instance."""
    return ClusterManager(
        method=ClusteringMethod.KMEANS,
        min_cluster_size=3,
        max_cluster_size=20
    )


@pytest.fixture
def summary_generator():
    """Create summary generator instance (no API key for tests)."""
    return SummaryGenerator(
        provider="openai",
        model="gpt-3.5-turbo",
        api_key=None  # Will use fallback extractive summarization
    )


@pytest.fixture
def tree_builder(cluster_manager, summary_generator):
    """Create tree builder instance."""
    return TreeBuilder(
        cluster_manager=cluster_manager,
        summary_generator=summary_generator,
        max_depth=3,
        target_branching_factor=5
    )


@pytest.fixture
def tree_visualizer():
    """Create tree visualizer instance."""
    return TreeVisualizer(max_content_length=50)


# Cluster Manager Tests

class TestClusterManager:
    """Tests for ClusterManager."""

    def test_initialization(self, cluster_manager):
        """Test cluster manager initialization."""
        assert cluster_manager.min_cluster_size == 3
        assert cluster_manager.max_cluster_size == 20
        assert cluster_manager.method == ClusteringMethod.KMEANS

    def test_kmeans_clustering(self, cluster_manager, sample_embeddings):
        """Test K-means clustering."""
        result = cluster_manager.cluster(
            embeddings=sample_embeddings,
            num_clusters=3
        )

        assert isinstance(result, ClusteringResult)
        assert result.method == ClusteringMethod.KMEANS
        assert result.num_clusters > 0
        assert len(result.clusters) > 0
        assert len(result.labels) == len(sample_embeddings)

        # Check clusters have proper structure
        for cluster in result.clusters:
            assert isinstance(cluster, Cluster)
            assert cluster.size >= cluster_manager.min_cluster_size
            assert cluster.centroid is not None
            assert len(cluster.item_indices) == cluster.size

    def test_hierarchical_clustering(self, sample_embeddings):
        """Test hierarchical clustering."""
        manager = ClusterManager(
            method=ClusteringMethod.HIERARCHICAL,
            min_cluster_size=3
        )

        result = manager.cluster(
            embeddings=sample_embeddings,
            num_clusters=3
        )

        assert result.method == ClusteringMethod.HIERARCHICAL
        assert result.num_clusters > 0
        assert len(result.clusters) > 0

    def test_semantic_clustering(self, sample_embeddings):
        """Test semantic similarity clustering."""
        manager = ClusterManager(
            method=ClusteringMethod.SEMANTIC,
            similarity_threshold=0.7
        )

        result = manager.cluster(embeddings=sample_embeddings)

        assert result.method == ClusteringMethod.SEMANTIC
        assert result.num_clusters >= 0
        # May have outliers
        assert len(result.outliers) >= 0

    def test_empty_embeddings(self, cluster_manager):
        """Test clustering with empty embeddings."""
        result = cluster_manager.cluster(
            embeddings=np.array([])
        )

        assert result.num_clusters == 0
        assert len(result.clusters) == 0

    def test_cluster_metrics(self, cluster_manager, sample_embeddings):
        """Test cluster quality metrics computation."""
        result = cluster_manager.cluster(
            embeddings=sample_embeddings,
            num_clusters=3
        )

        # Should have some metrics
        assert isinstance(result.metrics, dict)

    def test_merge_small_clusters(self, cluster_manager, sample_embeddings):
        """Test merging small clusters."""
        # Force many clusters
        result = cluster_manager.cluster(
            embeddings=sample_embeddings,
            num_clusters=10
        )

        original_count = result.num_clusters

        # Merge small clusters
        merged_result = cluster_manager.merge_small_clusters(
            result=result,
            embeddings=sample_embeddings,
            min_size=5
        )

        # Should have fewer clusters after merging
        assert merged_result.num_clusters <= original_count


# Summary Generator Tests

class TestSummaryGenerator:
    """Tests for SummaryGenerator."""

    def test_initialization(self, summary_generator):
        """Test summary generator initialization."""
        assert summary_generator.provider == "openai"
        assert summary_generator.model == "gpt-3.5-turbo"
        assert summary_generator.temperature == 0.1

    def test_generate_leaf_summary(self, summary_generator):
        """Test generating leaf-level summary."""
        texts = [
            "This is the first text about machine learning.",
            "This is the second text about neural networks.",
            "This is the third text about deep learning."
        ]

        summary = summary_generator.generate_summary(
            texts=texts,
            level=SummaryLevel.LEAF,
            source_ids=["chunk_1", "chunk_2", "chunk_3"]
        )

        assert isinstance(summary, Summary)
        assert summary.level == SummaryLevel.LEAF
        assert len(summary.content) > 0
        assert len(summary.source_ids) == 3
        assert summary.quality_score >= 0.0

    def test_generate_intermediate_summary(self, summary_generator):
        """Test generating intermediate-level summary."""
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers."
        ]

        summary = summary_generator.generate_summary(
            texts=texts,
            level=SummaryLevel.INTERMEDIATE,
            source_ids=["node_1", "node_2"]
        )

        assert summary.level == SummaryLevel.INTERMEDIATE
        assert len(summary.content) > 0

    def test_generate_root_summary(self, summary_generator):
        """Test generating root-level summary."""
        texts = [
            "This document discusses machine learning and AI.",
            "Neural networks are a key technology in modern AI."
        ]

        summary = summary_generator.generate_summary(
            texts=texts,
            level=SummaryLevel.ROOT,
            source_ids=["section_1", "section_2"]
        )

        assert summary.level == SummaryLevel.ROOT
        assert len(summary.content) > 0

    def test_empty_texts(self, summary_generator):
        """Test summary generation with empty texts."""
        summary = summary_generator.generate_summary(
            texts=[],
            level=SummaryLevel.LEAF,
            source_ids=[]
        )

        assert summary.content == ""
        assert len(summary.source_ids) == 0

    def test_batch_summaries(self, summary_generator):
        """Test batch summary generation."""
        text_groups = [
            ["Text 1A", "Text 1B"],
            ["Text 2A", "Text 2B"],
            ["Text 3A", "Text 3B"]
        ]
        source_id_groups = [
            ["chunk_1a", "chunk_1b"],
            ["chunk_2a", "chunk_2b"],
            ["chunk_3a", "chunk_3b"]
        ]

        summaries = summary_generator.generate_batch_summaries(
            text_groups=text_groups,
            level=SummaryLevel.LEAF,
            source_id_groups=source_id_groups
        )

        assert len(summaries) == 3
        for summary in summaries:
            assert isinstance(summary, Summary)
            assert len(summary.content) > 0

    def test_extract_key_facts(self, summary_generator):
        """Test key fact extraction."""
        summary_text = (
            "The study shows that 85% of participants improved. "
            "This represents a significant increase from the 2020 baseline. "
            "The results demonstrate clear effectiveness."
        )

        facts = summary_generator._extract_key_facts(summary_text)

        assert isinstance(facts, list)
        # Should find facts with numbers
        assert any("85%" in fact for fact in facts)

    def test_extract_entities(self, summary_generator):
        """Test entity extraction."""
        summary_text = (
            "John Smith and Mary Johnson conducted the study at Stanford University. "
            "The results were published in Nature."
        )

        entities = summary_generator._extract_entities(summary_text)

        assert isinstance(entities, list)
        assert len(entities) > 0

    def test_quality_assessment(self, summary_generator):
        """Test summary quality assessment."""
        summary_text = "This is a well-written summary with multiple sentences. It contains relevant information."
        source_texts = ["Source text 1", "Source text 2"]
        key_facts = ["Fact 1", "Fact 2"]

        score = summary_generator._assess_quality(
            summary=summary_text,
            source_texts=source_texts,
            key_facts=key_facts
        )

        assert 0.0 <= score <= 1.0


# Tree Builder Tests

class TestTreeBuilder:
    """Tests for TreeBuilder."""

    def test_initialization(self, tree_builder):
        """Test tree builder initialization."""
        assert tree_builder.max_depth == 3
        assert tree_builder.target_branching_factor == 5

    def test_build_tree(self, tree_builder, sample_chunks, sample_embeddings):
        """Test building complete tree."""
        root, all_nodes = tree_builder.build_tree(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="test_doc"
        )

        assert isinstance(root, TreeNode)
        assert isinstance(all_nodes, dict)
        assert len(all_nodes) > len(sample_chunks)  # Should have internal nodes
        assert root.level > 0  # Root should be above level 0

        # Verify all leaf nodes are present
        leaf_nodes = [n for n in all_nodes.values() if n.is_leaf]
        assert len(leaf_nodes) == len(sample_chunks)

    def test_tree_structure(self, tree_builder, sample_chunks, sample_embeddings):
        """Test tree structure integrity."""
        root, all_nodes = tree_builder.build_tree(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="test_doc"
        )

        # Check all nodes are connected
        for node in all_nodes.values():
            if node.node_id != root.node_id:
                # Non-root nodes should have parent
                assert node.parent_id is not None
                assert node.parent_id in all_nodes

            # Check children exist
            for child_id in node.children_ids:
                assert child_id in all_nodes

    def test_compute_stats(self, tree_builder, sample_chunks, sample_embeddings):
        """Test tree statistics computation."""
        root, all_nodes = tree_builder.build_tree(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="test_doc"
        )

        stats = tree_builder.compute_tree_stats(root, all_nodes)

        assert isinstance(stats, TreeStats)
        assert stats.total_nodes == len(all_nodes)
        assert stats.leaf_nodes == len(sample_chunks)
        assert stats.internal_nodes == stats.total_nodes - stats.leaf_nodes
        assert stats.max_depth == root.level
        assert stats.avg_branching_factor > 0

    def test_get_path_to_root(self, tree_builder, sample_chunks, sample_embeddings):
        """Test path to root retrieval."""
        root, all_nodes = tree_builder.build_tree(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="test_doc"
        )

        # Get a leaf node
        leaf = next(n for n in all_nodes.values() if n.is_leaf)

        path = tree_builder.get_node_path_to_root(leaf.node_id, all_nodes)

        assert len(path) > 0
        assert path[0].node_id == leaf.node_id
        assert path[-1].node_id == root.node_id

    def test_get_subtree(self, tree_builder, sample_chunks, sample_embeddings):
        """Test subtree retrieval."""
        root, all_nodes = tree_builder.build_tree(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="test_doc"
        )

        # Get subtree from root (should be entire tree)
        subtree = tree_builder.get_subtree_nodes(root.node_id, all_nodes)

        assert len(subtree) == len(all_nodes)

        # Get subtree from intermediate node
        intermediate = next(
            n for n in all_nodes.values()
            if not n.is_leaf and n.node_id != root.node_id
        )
        subtree = tree_builder.get_subtree_nodes(intermediate.node_id, all_nodes)

        assert len(subtree) > 0
        assert len(subtree) < len(all_nodes)

    def test_get_level_nodes(self, tree_builder, sample_chunks, sample_embeddings):
        """Test retrieving nodes by level."""
        root, all_nodes = tree_builder.build_tree(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="test_doc"
        )

        # Get leaf nodes (level 0)
        level_0_nodes = tree_builder.get_level_nodes(0, all_nodes)
        assert len(level_0_nodes) == len(sample_chunks)
        assert all(n.is_leaf for n in level_0_nodes)

        # Get root level
        root_level_nodes = tree_builder.get_level_nodes(root.level, all_nodes)
        assert len(root_level_nodes) >= 1
        assert root in root_level_nodes


# Tree Visualizer Tests

class TestTreeVisualizer:
    """Tests for TreeVisualizer."""

    def test_initialization(self, tree_visualizer):
        """Test visualizer initialization."""
        assert tree_visualizer.max_content_length == 50

    def test_ascii_visualization(
        self,
        tree_visualizer,
        tree_builder,
        sample_chunks,
        sample_embeddings
    ):
        """Test ASCII tree visualization."""
        root, all_nodes = tree_builder.build_tree(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="test_doc"
        )

        ascii_output = tree_visualizer.visualize_ascii(
            root=root,
            all_nodes=all_nodes,
            show_content=True,
            show_scores=True
        )

        assert isinstance(ascii_output, str)
        assert len(ascii_output) > 0
        assert "Tree:" in ascii_output
        assert "L0" in ascii_output  # Should show level 0

    def test_level_visualization(
        self,
        tree_visualizer,
        tree_builder,
        sample_chunks,
        sample_embeddings
    ):
        """Test level visualization."""
        root, all_nodes = tree_builder.build_tree(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="test_doc"
        )

        level_output = tree_visualizer.visualize_level(
            level=0,
            all_nodes=all_nodes,
            show_content=True
        )

        assert isinstance(level_output, str)
        assert "Level 0" in level_output
        assert len(level_output) > 0

    def test_path_visualization(
        self,
        tree_visualizer,
        tree_builder,
        sample_chunks,
        sample_embeddings
    ):
        """Test path visualization."""
        root, all_nodes = tree_builder.build_tree(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="test_doc"
        )

        # Get a leaf node
        leaf = next(n for n in all_nodes.values() if n.is_leaf)

        path_output = tree_visualizer.visualize_path(
            node_id=leaf.node_id,
            all_nodes=all_nodes,
            show_content=True
        )

        assert isinstance(path_output, str)
        assert "Path to root" in path_output
        assert leaf.node_id in path_output

    def test_stats_visualization(
        self,
        tree_visualizer,
        tree_builder,
        sample_chunks,
        sample_embeddings
    ):
        """Test statistics visualization."""
        root, all_nodes = tree_builder.build_tree(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="test_doc"
        )

        stats = tree_builder.compute_tree_stats(root, all_nodes)

        stats_output = tree_visualizer.visualize_stats(
            stats=stats,
            all_nodes=all_nodes
        )

        assert isinstance(stats_output, str)
        assert "Tree Statistics" in stats_output
        assert "Total Nodes" in stats_output

    def test_find_nodes(
        self,
        tree_visualizer,
        tree_builder,
        sample_chunks,
        sample_embeddings
    ):
        """Test finding nodes by content."""
        root, all_nodes = tree_builder.build_tree(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="test_doc"
        )

        # Search for content
        matching_nodes = tree_visualizer.find_nodes_by_content(
            all_nodes=all_nodes,
            search_term="topic 0",
            case_sensitive=False
        )

        assert isinstance(matching_nodes, list)
        assert len(matching_nodes) > 0
        assert all(isinstance(n, TreeNode) for n in matching_nodes)


# Integration Tests

class TestIntegration:
    """Integration tests for complete pipeline."""

    def test_end_to_end_tree_construction(
        self,
        cluster_manager,
        summary_generator,
        sample_chunks,
        sample_embeddings
    ):
        """Test complete tree construction flow."""
        # Build tree
        builder = TreeBuilder(
            cluster_manager=cluster_manager,
            summary_generator=summary_generator,
            max_depth=3,
            target_branching_factor=5
        )

        root, all_nodes = builder.build_tree(
            chunks=sample_chunks,
            embeddings=sample_embeddings,
            doc_id="integration_test"
        )

        # Verify tree
        assert root is not None
        assert len(all_nodes) > len(sample_chunks)

        # Verify all summaries were generated
        internal_nodes = [n for n in all_nodes.values() if not n.is_leaf]
        assert len(internal_nodes) > 0
        assert all(n.summary is not None for n in internal_nodes)

        # Compute and verify stats
        stats = builder.compute_tree_stats(root, all_nodes)
        assert stats.total_nodes == len(all_nodes)
        assert stats.max_depth == root.level

        # Visualize
        visualizer = TreeVisualizer()
        ascii_tree = visualizer.visualize_ascii(root, all_nodes)
        assert len(ascii_tree) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
