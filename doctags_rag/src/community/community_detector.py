"""
Community Detection Module using multiple algorithms.

Implements various community detection algorithms:
- Louvain Method: Fast modularity optimization with hierarchical structure
- Leiden Algorithm: Improved Louvain with better quality guarantees
- Label Propagation: Fast, memory-efficient approach
- Spectral Clustering: For well-separated communities
- Auto-selection based on graph properties
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from collections import defaultdict

import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering
from loguru import logger

# Import community detection libraries
try:
    import community as community_louvain
except ImportError:
    community_louvain = None
    logger.warning("python-louvain not installed. Louvain algorithm unavailable.")

try:
    import leidenalg
    import igraph as ig
except ImportError:
    leidenalg = None
    ig = None
    logger.warning("leidenalg not installed. Leiden algorithm unavailable.")


class CommunityAlgorithm(Enum):
    """Community detection algorithms."""
    LOUVAIN = "louvain"
    LEIDEN = "leiden"
    LABEL_PROPAGATION = "label_propagation"
    SPECTRAL = "spectral"
    AUTO = "auto"


@dataclass
class CommunityResult:
    """Results from community detection."""
    communities: Dict[int, Set[str]]  # community_id -> set of node_ids
    node_to_community: Dict[str, int]  # node_id -> community_id
    algorithm: str
    modularity: float
    num_communities: int
    execution_time: float
    hierarchical_levels: Optional[Dict[int, Dict[int, Set[str]]]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GraphProperties:
    """Properties of a graph for algorithm selection."""
    num_nodes: int
    num_edges: int
    avg_degree: float
    density: float
    is_connected: bool
    num_components: int
    avg_clustering_coefficient: float


class CommunityDetector:
    """
    Detects communities in graphs using multiple algorithms.

    Supports Louvain, Leiden, Label Propagation, and Spectral Clustering.
    Can auto-select the best algorithm based on graph properties.
    """

    def __init__(
        self,
        algorithm: CommunityAlgorithm = CommunityAlgorithm.AUTO,
        resolution: float = 1.0,
        min_community_size: int = 3,
        random_seed: int = 42
    ):
        """
        Initialize community detector.

        Args:
            algorithm: Algorithm to use (or AUTO for automatic selection)
            resolution: Resolution parameter for modularity-based methods
            min_community_size: Minimum size for valid communities
            random_seed: Random seed for reproducibility
        """
        self.algorithm = algorithm
        self.resolution = resolution
        self.min_community_size = min_community_size
        self.random_seed = random_seed

    def detect_communities(
        self,
        graph: nx.Graph,
        algorithm: Optional[CommunityAlgorithm] = None
    ) -> CommunityResult:
        """
        Detect communities in the graph.

        Args:
            graph: NetworkX graph
            algorithm: Override default algorithm

        Returns:
            CommunityResult with detected communities
        """
        start_time = time.time()

        algo = algorithm or self.algorithm

        # Auto-select algorithm if needed
        if algo == CommunityAlgorithm.AUTO:
            algo = self._select_algorithm(graph)
            logger.info(f"Auto-selected algorithm: {algo.value}")

        # Run appropriate algorithm
        if algo == CommunityAlgorithm.LOUVAIN:
            result = self._detect_louvain(graph)
        elif algo == CommunityAlgorithm.LEIDEN:
            result = self._detect_leiden(graph)
        elif algo == CommunityAlgorithm.LABEL_PROPAGATION:
            result = self._detect_label_propagation(graph)
        elif algo == CommunityAlgorithm.SPECTRAL:
            result = self._detect_spectral(graph)
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        # Filter small communities
        result = self._filter_small_communities(result)

        # Calculate modularity
        result.modularity = self._calculate_modularity(graph, result.node_to_community)

        result.execution_time = time.time() - start_time

        logger.info(
            f"Detected {result.num_communities} communities using {result.algorithm} "
            f"(modularity: {result.modularity:.4f}, time: {result.execution_time:.2f}s)"
        )

        return result

    def _detect_louvain(self, graph: nx.Graph) -> CommunityResult:
        """Detect communities using Louvain method."""
        if community_louvain is None:
            raise ImportError("python-louvain not installed. Install with: pip install python-louvain")

        # Convert to undirected if needed
        if graph.is_directed():
            graph = graph.to_undirected()

        # Run Louvain
        partition = community_louvain.best_partition(
            graph,
            resolution=self.resolution,
            random_state=self.random_seed
        )

        # Build community structure
        communities = defaultdict(set)
        for node, comm_id in partition.items():
            communities[comm_id].add(node)

        # Get hierarchical structure
        dendogram = community_louvain.generate_dendrogram(
            graph,
            resolution=self.resolution,
            random_state=self.random_seed
        )

        hierarchical_levels = {}
        for level in range(len(dendogram)):
            level_partition = community_louvain.partition_at_level(dendogram, level)
            level_communities = defaultdict(set)
            for node, comm_id in level_partition.items():
                level_communities[comm_id].add(node)
            hierarchical_levels[level] = dict(level_communities)

        return CommunityResult(
            communities=dict(communities),
            node_to_community=partition,
            algorithm="louvain",
            modularity=0.0,  # Will be calculated later
            num_communities=len(communities),
            execution_time=0.0,  # Will be set later
            hierarchical_levels=hierarchical_levels
        )

    def _detect_leiden(self, graph: nx.Graph) -> CommunityResult:
        """Detect communities using Leiden algorithm."""
        if leidenalg is None or ig is None:
            raise ImportError(
                "leidenalg not installed. Install with: pip install leidenalg python-igraph"
            )

        # Convert NetworkX to igraph
        ig_graph = ig.Graph.from_networkx(graph)

        # Run Leiden
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.resolution,
            seed=self.random_seed
        )

        # Convert back to NetworkX node IDs
        node_list = list(graph.nodes())
        node_to_community = {}
        communities = defaultdict(set)

        for comm_id, members in enumerate(partition):
            for member_idx in members:
                node_id = node_list[member_idx]
                node_to_community[node_id] = comm_id
                communities[comm_id].add(node_id)

        return CommunityResult(
            communities=dict(communities),
            node_to_community=node_to_community,
            algorithm="leiden",
            modularity=0.0,
            num_communities=len(communities),
            execution_time=0.0
        )

    def _detect_label_propagation(self, graph: nx.Graph) -> CommunityResult:
        """Detect communities using label propagation."""
        # Convert to undirected if needed
        if graph.is_directed():
            graph = graph.to_undirected()

        # Run label propagation
        communities_generator = nx.community.label_propagation_communities(graph)
        communities_list = list(communities_generator)

        # Build structures
        communities = {}
        node_to_community = {}

        for comm_id, members in enumerate(communities_list):
            communities[comm_id] = set(members)
            for node in members:
                node_to_community[node] = comm_id

        return CommunityResult(
            communities=communities,
            node_to_community=node_to_community,
            algorithm="label_propagation",
            modularity=0.0,
            num_communities=len(communities),
            execution_time=0.0
        )

    def _detect_spectral(self, graph: nx.Graph) -> CommunityResult:
        """Detect communities using spectral clustering."""
        # Estimate number of clusters based on graph properties
        n_clusters = max(2, min(50, int(np.sqrt(graph.number_of_nodes()))))

        # Get adjacency matrix
        adj_matrix = nx.to_scipy_sparse_array(graph)

        # Convert indices to int32 for sklearn 1.6+ compatibility
        adj_matrix.indices = adj_matrix.indices.astype(np.int32)
        adj_matrix.indptr = adj_matrix.indptr.astype(np.int32)

        # Run spectral clustering
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=self.random_seed,
            n_init=10
        )

        labels = clustering.fit_predict(adj_matrix)

        # Build structures
        node_list = list(graph.nodes())
        communities = defaultdict(set)
        node_to_community = {}

        for node_idx, label in enumerate(labels):
            node_id = node_list[node_idx]
            communities[label].add(node_id)
            node_to_community[node_id] = label

        return CommunityResult(
            communities=dict(communities),
            node_to_community=node_to_community,
            algorithm="spectral",
            modularity=0.0,
            num_communities=len(communities),
            execution_time=0.0
        )

    def _select_algorithm(self, graph: nx.Graph) -> CommunityAlgorithm:
        """
        Auto-select best algorithm based on graph properties.

        Args:
            graph: NetworkX graph

        Returns:
            Selected algorithm
        """
        props = self._analyze_graph_properties(graph)

        # Decision logic based on graph properties
        if props.num_nodes < 100:
            # Small graphs: use spectral clustering
            return CommunityAlgorithm.SPECTRAL

        elif props.num_nodes < 10000:
            # Medium graphs: use Leiden if available, else Louvain
            if leidenalg is not None:
                return CommunityAlgorithm.LEIDEN
            elif community_louvain is not None:
                return CommunityAlgorithm.LOUVAIN
            else:
                return CommunityAlgorithm.LABEL_PROPAGATION

        else:
            # Large graphs: use label propagation (fastest)
            return CommunityAlgorithm.LABEL_PROPAGATION

    def _analyze_graph_properties(self, graph: nx.Graph) -> GraphProperties:
        """Analyze graph properties for algorithm selection."""
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()

        if num_nodes == 0:
            return GraphProperties(0, 0, 0.0, 0.0, False, 0, 0.0)

        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
        max_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_edges if max_edges > 0 else 0

        # Check connectivity
        if graph.is_directed():
            is_connected = nx.is_weakly_connected(graph)
            num_components = nx.number_weakly_connected_components(graph)
        else:
            is_connected = nx.is_connected(graph)
            num_components = nx.number_connected_components(graph)

        # Clustering coefficient (sample for large graphs)
        if num_nodes > 1000:
            sample_nodes = np.random.choice(
                list(graph.nodes()),
                size=min(1000, num_nodes),
                replace=False
            )
            avg_clustering = nx.average_clustering(graph.subgraph(sample_nodes))
        else:
            avg_clustering = nx.average_clustering(graph)

        return GraphProperties(
            num_nodes=num_nodes,
            num_edges=num_edges,
            avg_degree=avg_degree,
            density=density,
            is_connected=is_connected,
            num_components=num_components,
            avg_clustering_coefficient=avg_clustering
        )

    def _filter_small_communities(self, result: CommunityResult) -> CommunityResult:
        """Filter out communities smaller than min_community_size."""
        if self.min_community_size <= 1:
            return result

        # Find communities to keep
        valid_communities = {
            comm_id: members
            for comm_id, members in result.communities.items()
            if len(members) >= self.min_community_size
        }

        # Reassign node mappings
        new_node_to_community = {
            node: comm_id
            for comm_id, members in valid_communities.items()
            for node in members
        }

        # Renumber communities
        old_to_new = {old_id: new_id for new_id, old_id in enumerate(valid_communities.keys())}

        renumbered_communities = {
            old_to_new[old_id]: members
            for old_id, members in valid_communities.items()
        }

        renumbered_node_to_community = {
            node: old_to_new[old_comm]
            for node, old_comm in new_node_to_community.items()
        }

        return CommunityResult(
            communities=renumbered_communities,
            node_to_community=renumbered_node_to_community,
            algorithm=result.algorithm,
            modularity=result.modularity,
            num_communities=len(renumbered_communities),
            execution_time=result.execution_time,
            hierarchical_levels=result.hierarchical_levels,
            metadata=result.metadata
        )

    def _calculate_modularity(
        self,
        graph: nx.Graph,
        partition: Dict[str, int]
    ) -> float:
        """
        Calculate modularity of a partition.

        Args:
            graph: NetworkX graph
            partition: Node to community mapping

        Returns:
            Modularity score
        """
        # Convert to undirected for modularity calculation
        if graph.is_directed():
            graph = graph.to_undirected()

        # Convert partition to set of sets format for NetworkX
        communities = defaultdict(set)
        for node, comm_id in partition.items():
            communities[comm_id].add(node)

        community_list = list(communities.values())

        try:
            return nx.community.modularity(graph, community_list)
        except Exception as e:
            logger.warning(f"Failed to calculate modularity: {e}")
            return 0.0

    def compare_algorithms(
        self,
        graph: nx.Graph,
        algorithms: Optional[List[CommunityAlgorithm]] = None
    ) -> Dict[str, CommunityResult]:
        """
        Compare multiple algorithms on the same graph.

        Args:
            graph: NetworkX graph
            algorithms: List of algorithms to compare (all available if None)

        Returns:
            Dictionary mapping algorithm names to results
        """
        if algorithms is None:
            algorithms = [
                CommunityAlgorithm.LOUVAIN,
                CommunityAlgorithm.LEIDEN,
                CommunityAlgorithm.LABEL_PROPAGATION,
                CommunityAlgorithm.SPECTRAL
            ]

        results = {}

        for algo in algorithms:
            try:
                result = self.detect_communities(graph, algorithm=algo)
                results[algo.value] = result
            except (ImportError, Exception) as e:
                logger.warning(f"Algorithm {algo.value} failed: {e}")

        return results

    def get_community_subgraph(
        self,
        graph: nx.Graph,
        result: CommunityResult,
        community_id: int
    ) -> nx.Graph:
        """
        Extract subgraph for a specific community.

        Args:
            graph: Original graph
            result: Community detection result
            community_id: Community to extract

        Returns:
            Subgraph containing only community nodes
        """
        if community_id not in result.communities:
            raise ValueError(f"Community {community_id} not found")

        nodes = result.communities[community_id]
        return graph.subgraph(nodes).copy()
