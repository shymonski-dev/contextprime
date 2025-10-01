"""
Hierarchical Tree Builder for RAPTOR System.

Implements bottom-up tree construction with:
- Recursive clustering and summarization
- Balanced tree structure
- Parent-child relationships
- Cross-references between nodes
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from uuid import uuid4
import numpy as np
from loguru import logger

from .cluster_manager import ClusterManager, ClusteringResult, ClusteringMethod
from .summary_generator import SummaryGenerator, SummaryLevel, Summary


@dataclass
class TreeNode:
    """
    Represents a node in the RAPTOR tree.

    Can be a leaf node (original chunk) or internal node (summary).
    """
    node_id: str
    content: str
    level: int  # 0 = leaf, increases towards root
    is_leaf: bool
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    sibling_ids: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[Summary] = None

    def add_child(self, child_id: str):
        """Add child node."""
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def add_sibling(self, sibling_id: str):
        """Add sibling node."""
        if sibling_id not in self.sibling_ids and sibling_id != self.node_id:
            self.sibling_ids.append(sibling_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'node_id': self.node_id,
            'content': self.content,
            'level': self.level,
            'is_leaf': self.is_leaf,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'sibling_ids': self.sibling_ids,
            'metadata': self.metadata,
            'summary': self.summary.to_dict() if self.summary else None,
            'has_embedding': self.embedding is not None
        }


@dataclass
class TreeStats:
    """Statistics about the tree structure."""
    total_nodes: int
    leaf_nodes: int
    internal_nodes: int
    max_depth: int
    avg_branching_factor: float
    num_levels: int
    nodes_per_level: Dict[int, int]

    def __repr__(self) -> str:
        return (
            f"TreeStats(total={self.total_nodes}, depth={self.max_depth}, "
            f"branching={self.avg_branching_factor:.2f})"
        )


class TreeBuilder:
    """
    Builds hierarchical trees using bottom-up construction.

    Process:
    1. Start with document chunks as leaves
    2. Cluster similar chunks
    3. Generate summaries for each cluster
    4. Recursively cluster and summarize
    5. Continue until reaching root
    """

    def __init__(
        self,
        cluster_manager: ClusterManager,
        summary_generator: SummaryGenerator,
        max_depth: int = 5,
        min_nodes_per_level: int = 2,
        target_branching_factor: int = 5
    ):
        """
        Initialize tree builder.

        Args:
            cluster_manager: Cluster manager instance
            summary_generator: Summary generator instance
            max_depth: Maximum tree depth
            min_nodes_per_level: Minimum nodes to continue building
            target_branching_factor: Target children per parent
        """
        self.cluster_manager = cluster_manager
        self.summary_generator = summary_generator
        self.max_depth = max_depth
        self.min_nodes_per_level = min_nodes_per_level
        self.target_branching_factor = target_branching_factor

        logger.info(
            f"TreeBuilder initialized: max_depth={max_depth}, "
            f"branching={target_branching_factor}"
        )

    def build_tree(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: np.ndarray,
        doc_id: str
    ) -> Tuple[TreeNode, Dict[str, TreeNode]]:
        """
        Build hierarchical tree from document chunks.

        Args:
            chunks: List of chunk dictionaries
            embeddings: Chunk embeddings
            doc_id: Document ID

        Returns:
            Tuple of (root_node, all_nodes_dict)
        """
        logger.info(f"Building tree for document {doc_id} with {len(chunks)} chunks")

        # Create leaf nodes
        all_nodes = {}
        current_level_nodes = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            node = TreeNode(
                node_id=f"{doc_id}_leaf_{i:04d}",
                content=chunk.get('content', ''),
                level=0,
                is_leaf=True,
                embedding=embedding,
                metadata={
                    'chunk_id': chunk.get('chunk_id', f"chunk_{i}"),
                    'doc_id': doc_id,
                    **chunk.get('metadata', {})
                }
            )
            all_nodes[node.node_id] = node
            current_level_nodes.append(node)

        logger.info(f"Created {len(current_level_nodes)} leaf nodes")

        # Build tree bottom-up
        level = 1
        while len(current_level_nodes) > 1 and level <= self.max_depth:
            logger.info(f"Building level {level} from {len(current_level_nodes)} nodes")

            # Build next level
            next_level_nodes = self._build_level(
                current_level_nodes,
                level,
                doc_id,
                all_nodes
            )

            if not next_level_nodes:
                logger.warning(f"No nodes created at level {level}, stopping")
                break

            current_level_nodes = next_level_nodes
            level += 1

        # Handle root node
        if len(current_level_nodes) == 1:
            root = current_level_nodes[0]
        else:
            # Create artificial root if multiple nodes remain
            root = self._create_root_node(
                current_level_nodes,
                level,
                doc_id,
                all_nodes
            )

        logger.info(f"Tree built successfully: {len(all_nodes)} total nodes, depth={root.level}")

        return root, all_nodes

    def _build_level(
        self,
        child_nodes: List[TreeNode],
        level: int,
        doc_id: str,
        all_nodes: Dict[str, TreeNode]
    ) -> List[TreeNode]:
        """
        Build a single level of the tree.

        Args:
            child_nodes: Nodes from previous level
            level: Current level
            doc_id: Document ID
            all_nodes: Dictionary of all nodes

        Returns:
            List of parent nodes
        """
        if len(child_nodes) < self.min_nodes_per_level:
            return child_nodes

        # Extract embeddings
        embeddings = np.array([node.embedding for node in child_nodes])

        # Estimate number of clusters
        num_clusters = max(
            2,
            len(child_nodes) // self.target_branching_factor
        )

        # Cluster nodes
        clustering_result = self.cluster_manager.cluster(
            embeddings,
            num_clusters=num_clusters
        )

        logger.info(
            f"Level {level}: created {clustering_result.num_clusters} clusters "
            f"from {len(child_nodes)} nodes"
        )

        # Create parent nodes for each cluster
        parent_nodes = []

        for cluster in clustering_result.clusters:
            # Get cluster children
            cluster_children = [child_nodes[i] for i in cluster.item_indices]

            # Create parent node
            parent = self._create_parent_node(
                cluster_children,
                level,
                doc_id,
                cluster.cluster_id,
                cluster.centroid
            )

            # Update relationships
            for child in cluster_children:
                child.parent_id = parent.node_id
                parent.add_child(child.node_id)

                # Add siblings
                for other_child in cluster_children:
                    if other_child.node_id != child.node_id:
                        child.add_sibling(other_child.node_id)

            all_nodes[parent.node_id] = parent
            parent_nodes.append(parent)

        # Handle outliers by assigning to nearest cluster
        if clustering_result.outliers:
            self._handle_outliers(
                clustering_result.outliers,
                child_nodes,
                parent_nodes,
                embeddings
            )

        return parent_nodes

    def _create_parent_node(
        self,
        children: List[TreeNode],
        level: int,
        doc_id: str,
        cluster_id: int,
        centroid: Optional[np.ndarray]
    ) -> TreeNode:
        """
        Create parent node from cluster of children.

        Args:
            children: Child nodes
            level: Parent level
            doc_id: Document ID
            cluster_id: Cluster ID
            centroid: Cluster centroid embedding

        Returns:
            Parent node
        """
        # Collect child contents
        child_contents = [child.content for child in children]
        child_ids = [child.node_id for child in children]

        # Determine summary level
        if level == 1:
            summary_level = SummaryLevel.LEAF
        elif level < self.max_depth - 1:
            summary_level = SummaryLevel.INTERMEDIATE
        else:
            summary_level = SummaryLevel.ROOT

        # Generate summary
        summary = self.summary_generator.generate_summary(
            texts=child_contents,
            level=summary_level,
            source_ids=child_ids,
            context={
                'doc_id': doc_id,
                'level': level,
                'num_children': len(children)
            }
        )

        # Create node
        node = TreeNode(
            node_id=f"{doc_id}_L{level}_C{cluster_id:04d}",
            content=summary.content,
            level=level,
            is_leaf=False,
            embedding=centroid,
            metadata={
                'doc_id': doc_id,
                'cluster_id': cluster_id,
                'num_children': len(children),
                'summary_length': len(summary.content),
                'quality_score': summary.quality_score
            },
            summary=summary
        )

        return node

    def _create_root_node(
        self,
        top_level_nodes: List[TreeNode],
        level: int,
        doc_id: str,
        all_nodes: Dict[str, TreeNode]
    ) -> TreeNode:
        """
        Create artificial root node from top-level nodes.

        Args:
            top_level_nodes: Nodes at top level
            level: Root level
            doc_id: Document ID
            all_nodes: Dictionary of all nodes

        Returns:
            Root node
        """
        logger.info(f"Creating root node from {len(top_level_nodes)} top-level nodes")

        # Collect contents
        contents = [node.content for node in top_level_nodes]
        node_ids = [node.node_id for node in top_level_nodes]

        # Generate root summary
        summary = self.summary_generator.generate_summary(
            texts=contents,
            level=SummaryLevel.ROOT,
            source_ids=node_ids,
            context={
                'doc_id': doc_id,
                'level': level,
                'is_root': True
            }
        )

        # Compute root embedding (average of children)
        embeddings = np.array([node.embedding for node in top_level_nodes])
        root_embedding = np.mean(embeddings, axis=0)

        # Create root node
        root = TreeNode(
            node_id=f"{doc_id}_root",
            content=summary.content,
            level=level,
            is_leaf=False,
            embedding=root_embedding,
            metadata={
                'doc_id': doc_id,
                'is_root': True,
                'num_children': len(top_level_nodes),
                'quality_score': summary.quality_score
            },
            summary=summary
        )

        # Update relationships
        for child in top_level_nodes:
            child.parent_id = root.node_id
            root.add_child(child.node_id)

            # Add siblings
            for other_child in top_level_nodes:
                if other_child.node_id != child.node_id:
                    child.add_sibling(other_child.node_id)

        all_nodes[root.node_id] = root

        return root

    def _handle_outliers(
        self,
        outlier_indices: List[int],
        all_children: List[TreeNode],
        parent_nodes: List[TreeNode],
        embeddings: np.ndarray
    ):
        """
        Assign outlier nodes to nearest cluster.

        Args:
            outlier_indices: Indices of outlier nodes
            all_children: All child nodes
            parent_nodes: All parent nodes
            embeddings: Node embeddings
        """
        if not outlier_indices or not parent_nodes:
            return

        logger.info(f"Handling {len(outlier_indices)} outlier nodes")

        # Get parent centroids
        parent_embeddings = np.array([p.embedding for p in parent_nodes])

        for idx in outlier_indices:
            child = all_children[idx]
            child_embedding = embeddings[idx]

            # Find nearest parent
            similarities = np.dot(parent_embeddings, child_embedding)
            nearest_idx = np.argmax(similarities)
            nearest_parent = parent_nodes[nearest_idx]

            # Assign to nearest parent
            child.parent_id = nearest_parent.node_id
            nearest_parent.add_child(child.node_id)

            # Update siblings
            for other_child_id in nearest_parent.children_ids:
                if other_child_id != child.node_id:
                    child.add_sibling(other_child_id)

    def compute_tree_stats(
        self,
        root: TreeNode,
        all_nodes: Dict[str, TreeNode]
    ) -> TreeStats:
        """
        Compute statistics about the tree.

        Args:
            root: Root node
            all_nodes: All nodes

        Returns:
            TreeStats
        """
        leaf_nodes = sum(1 for n in all_nodes.values() if n.is_leaf)
        internal_nodes = sum(1 for n in all_nodes.values() if not n.is_leaf)

        # Count nodes per level
        nodes_per_level = {}
        for node in all_nodes.values():
            nodes_per_level[node.level] = nodes_per_level.get(node.level, 0) + 1

        # Compute average branching factor
        parent_nodes = [n for n in all_nodes.values() if not n.is_leaf]
        if parent_nodes:
            avg_branching = sum(len(n.children_ids) for n in parent_nodes) / len(parent_nodes)
        else:
            avg_branching = 0

        return TreeStats(
            total_nodes=len(all_nodes),
            leaf_nodes=leaf_nodes,
            internal_nodes=internal_nodes,
            max_depth=root.level,
            avg_branching_factor=avg_branching,
            num_levels=len(nodes_per_level),
            nodes_per_level=nodes_per_level
        )

    def get_node_path_to_root(
        self,
        node_id: str,
        all_nodes: Dict[str, TreeNode]
    ) -> List[TreeNode]:
        """
        Get path from node to root.

        Args:
            node_id: Starting node ID
            all_nodes: All nodes

        Returns:
            List of nodes from node to root
        """
        path = []
        current_id = node_id

        while current_id and current_id in all_nodes:
            node = all_nodes[current_id]
            path.append(node)
            current_id = node.parent_id

        return path

    def get_subtree_nodes(
        self,
        node_id: str,
        all_nodes: Dict[str, TreeNode]
    ) -> List[TreeNode]:
        """
        Get all nodes in subtree rooted at node.

        Args:
            node_id: Root node ID of subtree
            all_nodes: All nodes

        Returns:
            List of nodes in subtree
        """
        if node_id not in all_nodes:
            return []

        subtree = []
        queue = [node_id]
        visited = set()

        while queue:
            current_id = queue.pop(0)

            if current_id in visited:
                continue

            visited.add(current_id)
            node = all_nodes[current_id]
            subtree.append(node)

            # Add children to queue
            queue.extend(node.children_ids)

        return subtree

    def get_level_nodes(
        self,
        level: int,
        all_nodes: Dict[str, TreeNode]
    ) -> List[TreeNode]:
        """
        Get all nodes at a specific level.

        Args:
            level: Tree level
            all_nodes: All nodes

        Returns:
            List of nodes at level
        """
        return [node for node in all_nodes.values() if node.level == level]

    def balance_tree(
        self,
        root: TreeNode,
        all_nodes: Dict[str, TreeNode]
    ) -> Tuple[TreeNode, Dict[str, TreeNode]]:
        """
        Balance tree by redistributing nodes.

        Args:
            root: Root node
            all_nodes: All nodes

        Returns:
            Tuple of (new_root, new_all_nodes)
        """
        # Check if tree needs balancing
        stats = self.compute_tree_stats(root, all_nodes)

        # If tree is relatively balanced, return as is
        if stats.avg_branching_factor <= self.target_branching_factor * 1.5:
            return root, all_nodes

        logger.info("Tree is imbalanced, rebuilding...")

        # Extract leaf nodes
        leaf_nodes = [n for n in all_nodes.values() if n.is_leaf]
        leaf_embeddings = np.array([n.embedding for n in leaf_nodes])

        # Rebuild tree
        chunks = [
            {
                'content': n.content,
                'chunk_id': n.node_id,
                'metadata': n.metadata
            }
            for n in leaf_nodes
        ]

        doc_id = root.metadata.get('doc_id', 'doc')

        return self.build_tree(chunks, leaf_embeddings, doc_id)
