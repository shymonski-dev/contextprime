"""
Graph Analysis Module for computing metrics and analyzing graph structures.

Provides comprehensive graph analysis including:
- Node-level metrics (centrality, importance)
- Graph-level metrics (diameter, clustering)
- Community-level metrics (modularity, conductance)
- Entity importance scoring
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time

import networkx as nx
import numpy as np
from loguru import logger

from .community_detector import CommunityResult


@dataclass
class NodeMetrics:
    """Metrics for a single node."""
    node_id: str
    degree: int
    degree_centrality: float
    betweenness_centrality: float
    closeness_centrality: float
    eigenvector_centrality: float
    pagerank: float
    clustering_coefficient: float
    is_bridge: bool = False
    is_hub: bool = False
    community_id: Optional[int] = None


@dataclass
class GraphMetrics:
    """Graph-level metrics."""
    num_nodes: int
    num_edges: int
    density: float
    avg_degree: float
    avg_clustering_coefficient: float
    num_connected_components: int
    largest_component_size: int
    diameter: Optional[int] = None
    avg_shortest_path_length: Optional[float] = None
    transitivity: float = 0.0
    assortativity: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommunityMetrics:
    """Metrics for communities."""
    community_id: int
    size: int
    density: float
    internal_edges: int
    external_edges: int
    conductance: float
    modularity_contribution: float
    avg_degree: float
    top_nodes: List[Tuple[str, float]]  # (node_id, importance_score)


class GraphAnalyzer:
    """
    Analyzes graph structures and computes various metrics.

    Supports:
    - Centrality measures
    - Community quality metrics
    - Entity importance scoring
    - Graph statistics
    """

    def __init__(
        self,
        cache_metrics: bool = True,
        top_k_nodes: int = 10
    ):
        """
        Initialize graph analyzer.

        Args:
            cache_metrics: Whether to cache computed metrics
            top_k_nodes: Number of top nodes to identify per metric
        """
        self.cache_metrics = cache_metrics
        self.top_k_nodes = top_k_nodes
        self._metrics_cache: Dict[str, Any] = {}

    def analyze_graph(
        self,
        graph: nx.Graph,
        compute_diameter: bool = False
    ) -> GraphMetrics:
        """
        Compute comprehensive graph-level metrics.

        Args:
            graph: NetworkX graph
            compute_diameter: Whether to compute diameter (expensive for large graphs)

        Returns:
            GraphMetrics object
        """
        logger.info(f"Analyzing graph with {graph.number_of_nodes()} nodes")
        start_time = time.time()

        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        if num_nodes == 0:
            return GraphMetrics(
                num_nodes=0,
                num_edges=0,
                density=0.0,
                avg_degree=0.0,
                avg_clustering_coefficient=0.0,
                num_connected_components=0,
                largest_component_size=0
            )

        # Basic metrics
        density = nx.density(graph)
        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0

        # Clustering coefficient
        if num_nodes < 10000:
            avg_clustering = nx.average_clustering(graph)
        else:
            # Sample for large graphs
            sample_size = min(1000, num_nodes)
            sample_nodes = np.random.choice(list(graph.nodes()), sample_size, replace=False)
            avg_clustering = nx.average_clustering(graph.subgraph(sample_nodes))

        # Connected components
        if graph.is_directed():
            num_components = nx.number_weakly_connected_components(graph)
            components = list(nx.weakly_connected_components(graph))
        else:
            num_components = nx.number_connected_components(graph)
            components = list(nx.connected_components(graph))

        largest_component_size = max(len(c) for c in components) if components else 0

        # Diameter and average path length (expensive, only for small graphs or if requested)
        diameter = None
        avg_path_length = None

        if compute_diameter and num_nodes < 5000:
            try:
                if graph.is_directed():
                    if nx.is_weakly_connected(graph):
                        largest_cc = max(nx.weakly_connected_components(graph), key=len)
                        subgraph = graph.subgraph(largest_cc)
                        avg_path_length = nx.average_shortest_path_length(subgraph)
                else:
                    if nx.is_connected(graph):
                        diameter = nx.diameter(graph)
                        avg_path_length = nx.average_shortest_path_length(graph)
                    else:
                        largest_cc = max(nx.connected_components(graph), key=len)
                        subgraph = graph.subgraph(largest_cc)
                        diameter = nx.diameter(subgraph)
                        avg_path_length = nx.average_shortest_path_length(subgraph)
            except Exception as e:
                logger.warning(f"Failed to compute diameter/path length: {e}")

        # Transitivity (global clustering coefficient)
        transitivity = nx.transitivity(graph)

        # Assortativity (degree correlation)
        try:
            assortativity = nx.degree_assortativity_coefficient(graph)
        except Exception as e:
            logger.warning(f"Failed to compute assortativity: {e}")
            assortativity = None

        execution_time = time.time() - start_time

        metrics = GraphMetrics(
            num_nodes=num_nodes,
            num_edges=num_edges,
            density=density,
            avg_degree=avg_degree,
            avg_clustering_coefficient=avg_clustering,
            num_connected_components=num_components,
            largest_component_size=largest_component_size,
            diameter=diameter,
            avg_shortest_path_length=avg_path_length,
            transitivity=transitivity,
            assortativity=assortativity,
            metadata={"execution_time": execution_time}
        )

        logger.info(f"Graph analysis completed in {execution_time:.2f}s")
        return metrics

    def compute_node_metrics(
        self,
        graph: nx.Graph,
        nodes: Optional[List[str]] = None
    ) -> Dict[str, NodeMetrics]:
        """
        Compute metrics for nodes in the graph.

        Args:
            graph: NetworkX graph
            nodes: List of nodes to analyze (all nodes if None)

        Returns:
            Dictionary mapping node IDs to NodeMetrics
        """
        if nodes is None:
            nodes = list(graph.nodes())

        logger.info(f"Computing metrics for {len(nodes)} nodes")
        start_time = time.time()

        # Compute centrality measures
        degree_centrality = nx.degree_centrality(graph)

        # For large graphs, use approximations
        if graph.number_of_nodes() > 5000:
            # Approximate betweenness for large graphs
            k = min(1000, graph.number_of_nodes())
            betweenness = nx.betweenness_centrality(graph, k=k)
            closeness = {}  # Skip closeness for large graphs
        else:
            betweenness = nx.betweenness_centrality(graph)
            if not graph.is_directed():
                closeness = nx.closeness_centrality(graph)
            else:
                closeness = {}

        # Eigenvector centrality (may fail for disconnected graphs)
        try:
            eigenvector = nx.eigenvector_centrality(graph, max_iter=1000)
        except Exception as e:
            logger.warning(f"Eigenvector centrality failed: {e}")
            eigenvector = {node: 0.0 for node in graph.nodes()}

        # PageRank
        pagerank = nx.pagerank(graph)

        # Clustering coefficient
        clustering = nx.clustering(graph)

        # Build node metrics
        node_metrics = {}
        for node in nodes:
            if node not in graph:
                continue

            metrics = NodeMetrics(
                node_id=node,
                degree=graph.degree(node),
                degree_centrality=degree_centrality.get(node, 0.0),
                betweenness_centrality=betweenness.get(node, 0.0),
                closeness_centrality=closeness.get(node, 0.0),
                eigenvector_centrality=eigenvector.get(node, 0.0),
                pagerank=pagerank.get(node, 0.0),
                clustering_coefficient=clustering.get(node, 0.0)
            )
            node_metrics[node] = metrics

        # Identify bridges
        bridges = self._identify_bridges(graph)
        for node in bridges:
            if node in node_metrics:
                node_metrics[node].is_bridge = True

        # Identify hubs (high degree nodes)
        degrees = [m.degree for m in node_metrics.values()]
        if degrees:
            hub_threshold = np.percentile(degrees, 90)
            for node, metrics in node_metrics.items():
                if metrics.degree >= hub_threshold:
                    metrics.is_hub = True

        execution_time = time.time() - start_time
        logger.info(f"Node metrics computed in {execution_time:.2f}s")

        return node_metrics

    def analyze_communities(
        self,
        graph: nx.Graph,
        community_result: CommunityResult
    ) -> Dict[int, CommunityMetrics]:
        """
        Analyze quality metrics for detected communities.

        Args:
            graph: NetworkX graph
            community_result: Community detection result

        Returns:
            Dictionary mapping community IDs to metrics
        """
        logger.info(f"Analyzing {community_result.num_communities} communities")

        community_metrics = {}

        # Compute node importance for ranking
        pagerank = nx.pagerank(graph)

        for comm_id, members in community_result.communities.items():
            # Get subgraph
            subgraph = graph.subgraph(members)

            # Basic metrics
            size = len(members)
            density = nx.density(subgraph)
            internal_edges = subgraph.number_of_edges()

            # Count external edges
            external_edges = 0
            for node in members:
                for neighbor in graph.neighbors(node):
                    if neighbor not in members:
                        external_edges += 1

            # Conductance: ratio of external edges to all edges
            total_edges = internal_edges + external_edges
            conductance = external_edges / total_edges if total_edges > 0 else 0.0

            # Average degree within community
            avg_degree = 2 * internal_edges / size if size > 0 else 0.0

            # Top nodes by PageRank
            top_nodes = sorted(
                [(node, pagerank.get(node, 0.0)) for node in members],
                key=lambda x: x[1],
                reverse=True
            )[:self.top_k_nodes]

            # Modularity contribution (approximate)
            modularity_contrib = self._compute_modularity_contribution(
                graph, members, internal_edges
            )

            metrics = CommunityMetrics(
                community_id=comm_id,
                size=size,
                density=density,
                internal_edges=internal_edges,
                external_edges=external_edges,
                conductance=conductance,
                modularity_contribution=modularity_contrib,
                avg_degree=avg_degree,
                top_nodes=top_nodes
            )

            community_metrics[comm_id] = metrics

        return community_metrics

    def identify_bridge_nodes(self, graph: nx.Graph) -> Set[str]:
        """
        Identify bridge nodes (articulation points).

        Args:
            graph: NetworkX graph

        Returns:
            Set of bridge node IDs
        """
        return self._identify_bridges(graph)

    def _identify_bridges(self, graph: nx.Graph) -> Set[str]:
        """Identify articulation points/bridges."""
        if graph.is_directed():
            # For directed graphs, use strongly connected components
            return set()

        try:
            articulation_points = list(nx.articulation_points(graph))
            return set(articulation_points)
        except Exception as e:
            logger.warning(f"Failed to identify bridges: {e}")
            return set()

    def compute_influence_scores(
        self,
        graph: nx.Graph,
        seed_nodes: Optional[Set[str]] = None,
        iterations: int = 10
    ) -> Dict[str, float]:
        """
        Compute influence propagation scores using iterative spreading.

        Args:
            graph: NetworkX graph
            seed_nodes: Initial influential nodes (uses PageRank if None)
            iterations: Number of propagation iterations

        Returns:
            Dictionary mapping node IDs to influence scores
        """
        if seed_nodes is None:
            # Use top PageRank nodes as seeds
            pagerank = nx.pagerank(graph)
            threshold = np.percentile(list(pagerank.values()), 90)
            seed_nodes = {node for node, score in pagerank.items() if score >= threshold}

        # Initialize influence scores
        influence = {node: 1.0 if node in seed_nodes else 0.0 for node in graph.nodes()}

        # Propagate influence
        for _ in range(iterations):
            new_influence = influence.copy()
            for node in graph.nodes():
                if node not in seed_nodes:
                    # Aggregate influence from neighbors
                    neighbor_influence = sum(
                        influence[neighbor] for neighbor in graph.neighbors(node)
                    )
                    new_influence[node] = 0.5 * influence[node] + 0.5 * neighbor_influence / max(1, graph.degree(node))

            influence = new_influence

        return influence

    def _compute_modularity_contribution(
        self,
        graph: nx.Graph,
        community: Set[str],
        internal_edges: int
    ) -> float:
        """Compute a single community's contribution to overall modularity."""
        m = graph.number_of_edges()
        if m == 0:
            return 0.0

        # Sum of degrees in community
        degree_sum = sum(graph.degree(node) for node in community)

        # Modularity contribution formula
        lc = internal_edges
        dc = degree_sum

        return (lc / m) - ((dc / (2 * m)) ** 2)

    def get_hub_nodes(
        self,
        graph: nx.Graph,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Get hub nodes ranked by PageRank.

        Args:
            graph: NetworkX graph
            top_k: Number of top hubs to return (all if None)

        Returns:
            List of (node_id, pagerank_score) tuples
        """
        pagerank = nx.pagerank(graph)
        ranked = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)

        if top_k:
            return ranked[:top_k]
        return ranked

    def get_authority_nodes(
        self,
        graph: nx.Graph,
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Get authority nodes using HITS algorithm.

        Args:
            graph: NetworkX graph
            top_k: Number of top authorities to return

        Returns:
            List of (node_id, authority_score) tuples
        """
        try:
            hubs, authorities = nx.hits(graph, max_iter=1000)
            ranked = sorted(authorities.items(), key=lambda x: x[1], reverse=True)

            if top_k:
                return ranked[:top_k]
            return ranked
        except Exception as e:
            logger.warning(f"HITS algorithm failed: {e}")
            return []

    def compute_degree_distribution(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Compute degree distribution statistics.

        Args:
            graph: NetworkX graph

        Returns:
            Dictionary with distribution statistics
        """
        degrees = [d for n, d in graph.degree()]

        if not degrees:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0,
                "max": 0,
                "percentiles": {}
            }

        return {
            "mean": np.mean(degrees),
            "median": np.median(degrees),
            "std": np.std(degrees),
            "min": np.min(degrees),
            "max": np.max(degrees),
            "percentiles": {
                "25": np.percentile(degrees, 25),
                "50": np.percentile(degrees, 50),
                "75": np.percentile(degrees, 75),
                "90": np.percentile(degrees, 90),
                "95": np.percentile(degrees, 95),
                "99": np.percentile(degrees, 99)
            },
            "histogram": np.histogram(degrees, bins=20)
        }
