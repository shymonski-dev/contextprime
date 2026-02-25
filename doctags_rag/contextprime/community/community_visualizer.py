"""
Community Visualizer for creating visual representations of community structures.

Supports:
- Force-directed layouts
- Community-based coloring
- Interactive HTML visualizations
- Export to various formats
"""

from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from loguru import logger

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False
    logger.warning("pyvis not installed. Interactive visualizations unavailable.")

from .community_detector import CommunityResult


class CommunityVisualizer:
    """
    Creates visualizations of community structures in graphs.

    Supports static (matplotlib) and interactive (pyvis) visualizations.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 10),
        dpi: int = 300
    ):
        """
        Initialize community visualizer.

        Args:
            figsize: Figure size for static plots
            dpi: DPI for static plots
        """
        self.figsize = figsize
        self.dpi = dpi

    def visualize_communities(
        self,
        graph: nx.Graph,
        community_result: CommunityResult,
        output_path: str,
        layout: str = "spring",
        node_size_scale: float = 100.0,
        show_labels: bool = True,
        max_label_length: int = 20
    ) -> None:
        """
        Create a static visualization of communities.

        Args:
            graph: NetworkX graph
            community_result: Community detection result
            output_path: Output file path
            layout: Layout algorithm (spring, kamada_kawai, circular)
            node_size_scale: Scale factor for node sizes
            show_labels: Whether to show node labels
            max_label_length: Maximum label length
        """
        logger.info(f"Creating community visualization with {layout} layout")

        # Create figure
        plt.figure(figsize=self.figsize, dpi=self.dpi)

        # Compute layout
        if layout == "spring":
            pos = nx.spring_layout(graph, k=1, iterations=50)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(graph)
        elif layout == "circular":
            pos = nx.circular_layout(graph)
        else:
            pos = nx.spring_layout(graph)

        # Assign colors to communities
        num_communities = community_result.num_communities
        cmap = cm.get_cmap('tab20' if num_communities <= 20 else 'hsv')

        node_colors = []
        for node in graph.nodes():
            comm_id = community_result.node_to_community.get(node, -1)
            if comm_id >= 0:
                color = cmap(comm_id / max(num_communities, 1))
            else:
                color = (0.5, 0.5, 0.5, 1.0)  # Gray for unclustered
            node_colors.append(color)

        # Compute node sizes based on degree
        degrees = dict(graph.degree())
        node_sizes = [degrees.get(node, 1) * node_size_scale for node in graph.nodes()]

        # Draw graph
        nx.draw_networkx_edges(
            graph,
            pos,
            alpha=0.2,
            width=0.5,
            edge_color='gray'
        )

        nx.draw_networkx_nodes(
            graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8
        )

        # Draw labels if requested
        if show_labels:
            labels = {}
            for node in graph.nodes():
                label = str(node)
                if len(label) > max_label_length:
                    label = label[:max_label_length-3] + "..."
                labels[node] = label

            nx.draw_networkx_labels(
                graph,
                pos,
                labels,
                font_size=6,
                font_color='black'
            )

        plt.title(
            f"Community Structure ({num_communities} communities, "
            f"modularity={community_result.modularity:.3f})",
            fontsize=14
        )
        plt.axis('off')
        plt.tight_layout()

        # Save figure
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Visualization saved to {output_path}")

    def create_interactive_visualization(
        self,
        graph: nx.Graph,
        community_result: CommunityResult,
        output_path: str,
        height: str = "800px",
        width: str = "100%"
    ) -> None:
        """
        Create an interactive HTML visualization using pyvis.

        Args:
            graph: NetworkX graph
            community_result: Community detection result
            output_path: Output HTML file path
            height: Visualization height
            width: Visualization width
        """
        if not PYVIS_AVAILABLE:
            logger.error("pyvis not installed. Cannot create interactive visualization.")
            return

        logger.info("Creating interactive visualization")

        # Create pyvis network
        net = Network(height=height, width=width, notebook=False)

        # Assign colors to communities
        num_communities = community_result.num_communities
        cmap = cm.get_cmap('tab20' if num_communities <= 20 else 'hsv')

        # Add nodes
        for node in graph.nodes():
            comm_id = community_result.node_to_community.get(node, -1)

            if comm_id >= 0:
                color = self._rgba_to_hex(cmap(comm_id / max(num_communities, 1)))
            else:
                color = "#808080"  # Gray

            # Node size based on degree
            size = graph.degree(node) * 5 + 10

            net.add_node(
                str(node),
                label=str(node),
                color=color,
                size=size,
                title=f"Community: {comm_id}<br>Degree: {graph.degree(node)}"
            )

        # Add edges
        for source, target in graph.edges():
            net.add_edge(str(source), str(target))

        # Configure physics
        net.toggle_physics(True)
        net.show_buttons(filter_=['physics'])

        # Save
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(output_path))

        logger.info(f"Interactive visualization saved to {output_path}")

    def visualize_community_hierarchy(
        self,
        community_result: CommunityResult,
        output_path: str
    ) -> None:
        """
        Visualize hierarchical community structure if available.

        Args:
            community_result: Community detection result with hierarchical levels
            output_path: Output file path
        """
        if not community_result.hierarchical_levels:
            logger.warning("No hierarchical structure available")
            return

        logger.info("Creating hierarchical community visualization")

        num_levels = len(community_result.hierarchical_levels)
        fig, axes = plt.subplots(1, num_levels, figsize=(5*num_levels, 5))

        if num_levels == 1:
            axes = [axes]

        for level, (level_id, communities) in enumerate(community_result.hierarchical_levels.items()):
            ax = axes[level]

            # Create simple bar chart of community sizes
            sizes = [len(members) for members in communities.values()]
            comm_ids = list(communities.keys())

            ax.bar(range(len(sizes)), sizes)
            ax.set_xlabel("Community ID")
            ax.set_ylabel("Size")
            ax.set_title(f"Level {level_id}")

        plt.tight_layout()

        # Save
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Hierarchical visualization saved to {output_path}")

    def export_to_graphml(
        self,
        graph: nx.Graph,
        community_result: CommunityResult,
        output_path: str
    ) -> None:
        """
        Export graph with community information to GraphML format.

        Args:
            graph: NetworkX graph
            community_result: Community detection result
            output_path: Output file path
        """
        logger.info("Exporting to GraphML format")

        # Create a copy of the graph
        export_graph = graph.copy()

        # Add community information as node attributes
        for node in export_graph.nodes():
            comm_id = community_result.node_to_community.get(node, -1)
            export_graph.nodes[node]['community'] = comm_id

        # Export
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(export_graph, output_path)

        logger.info(f"GraphML export saved to {output_path}")

    def export_to_gexf(
        self,
        graph: nx.Graph,
        community_result: CommunityResult,
        output_path: str
    ) -> None:
        """
        Export graph with community information to GEXF format (for Gephi).

        Args:
            graph: NetworkX graph
            community_result: Community detection result
            output_path: Output file path
        """
        logger.info("Exporting to GEXF format")

        # Create a copy of the graph
        export_graph = graph.copy()

        # Add community information as node attributes
        for node in export_graph.nodes():
            comm_id = community_result.node_to_community.get(node, -1)
            export_graph.nodes[node]['community'] = comm_id

        # Export
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        nx.write_gexf(export_graph, output_path)

        logger.info(f"GEXF export saved to {output_path}")

    def export_to_d3_json(
        self,
        graph: nx.Graph,
        community_result: CommunityResult,
        output_path: str
    ) -> None:
        """
        Export graph to D3.js-compatible JSON format.

        Args:
            graph: NetworkX graph
            community_result: Community detection result
            output_path: Output file path
        """
        logger.info("Exporting to D3 JSON format")

        # Build nodes list
        nodes = []
        node_to_idx = {}

        for idx, node in enumerate(graph.nodes()):
            comm_id = community_result.node_to_community.get(node, -1)
            node_to_idx[node] = idx

            nodes.append({
                "id": str(node),
                "name": str(node),
                "community": comm_id,
                "degree": graph.degree(node)
            })

        # Build links list
        links = []
        for source, target in graph.edges():
            links.append({
                "source": node_to_idx[source],
                "target": node_to_idx[target]
            })

        # Create JSON structure
        data = {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "num_communities": community_result.num_communities,
                "modularity": community_result.modularity
            }
        }

        # Export
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"D3 JSON export saved to {output_path}")

    def create_community_size_distribution(
        self,
        community_result: CommunityResult,
        output_path: str
    ) -> None:
        """
        Create a visualization of community size distribution.

        Args:
            community_result: Community detection result
            output_path: Output file path
        """
        logger.info("Creating community size distribution plot")

        # Get community sizes
        sizes = [len(members) for members in community_result.communities.values()]

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        ax1.hist(sizes, bins=20, edgecolor='black')
        ax1.set_xlabel('Community Size')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Community Size Distribution')
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(sizes, vert=True)
        ax2.set_ylabel('Community Size')
        ax2.set_title('Community Size Statistics')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Size distribution plot saved to {output_path}")

    def _rgba_to_hex(self, rgba: Tuple[float, float, float, float]) -> str:
        """Convert RGBA tuple to hex color string."""
        r = int(rgba[0] * 255)
        g = int(rgba[1] * 255)
        b = int(rgba[2] * 255)
        return f"#{r:02x}{g:02x}{b:02x}"

    def visualize_modularity_comparison(
        self,
        results: Dict[str, CommunityResult],
        output_path: str
    ) -> None:
        """
        Compare modularity scores across different algorithms.

        Args:
            results: Dictionary mapping algorithm names to results
            output_path: Output file path
        """
        logger.info("Creating modularity comparison plot")

        algorithms = list(results.keys())
        modularities = [results[algo].modularity for algo in algorithms]
        num_communities = [results[algo].num_communities for algo in algorithms]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Modularity comparison
        ax1.bar(algorithms, modularities)
        ax1.set_ylabel('Modularity Score')
        ax1.set_title('Modularity Comparison')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Number of communities
        ax2.bar(algorithms, num_communities)
        ax2.set_ylabel('Number of Communities')
        ax2.set_title('Community Count Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Modularity comparison saved to {output_path}")
