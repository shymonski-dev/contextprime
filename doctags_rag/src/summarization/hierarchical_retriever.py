"""
Hierarchical Retriever for RAPTOR System.

Implements multi-level retrieval strategies:
- Top-down traversal from root
- Bottom-up aggregation from leaves
- Mid-level balanced retrieval
- Lateral exploration of siblings
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from loguru import logger

from .tree_builder import TreeNode
from .tree_storage import TreeStorage


class RetrievalStrategy(Enum):
    """Retrieval strategy for hierarchical search."""
    TOP_DOWN = "top_down"           # Start from root, traverse down
    BOTTOM_UP = "bottom_up"         # Start from leaves, aggregate up
    MID_LEVEL = "mid_level"         # Start from middle levels
    ADAPTIVE = "adaptive"           # Choose strategy based on query
    FULL_TREE = "full_tree"         # Search all levels


@dataclass
class RetrievalResult:
    """Result from hierarchical retrieval."""
    node: TreeNode
    score: float
    level: int
    path_to_root: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'node_id': self.node.node_id,
            'content': self.node.content,
            'score': self.score,
            'level': self.level,
            'is_leaf': self.node.is_leaf,
            'path_to_root': self.path_to_root,
            'context': self.context,
            'metadata': self.node.metadata
        }


class HierarchicalRetriever:
    """
    Retrieves relevant information from RAPTOR trees.

    Supports multiple traversal strategies and context building.
    """

    def __init__(
        self,
        tree_storage: TreeStorage,
        strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE,
        top_k: int = 10,
        score_threshold: float = 0.5,
        include_context: bool = True
    ):
        """
        Initialize hierarchical retriever.

        Args:
            tree_storage: Tree storage instance
            strategy: Retrieval strategy
            top_k: Number of results to return
            score_threshold: Minimum similarity score
            include_context: Include contextual nodes
        """
        self.tree_storage = tree_storage
        self.strategy = strategy
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.include_context = include_context

        logger.info(
            f"HierarchicalRetriever initialized: strategy={strategy.value}, "
            f"top_k={top_k}"
        )

    def retrieve(
        self,
        query_embedding: np.ndarray,
        tree_id: str,
        query_text: Optional[str] = None,
        strategy: Optional[RetrievalStrategy] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant nodes from tree.

        Args:
            query_embedding: Query embedding vector
            tree_id: Tree identifier
            query_text: Optional query text for strategy selection
            strategy: Override default strategy

        Returns:
            List of retrieval results
        """
        # Load tree
        tree_data = self.tree_storage.load_tree(tree_id)
        if not tree_data:
            logger.error(f"Tree {tree_id} not found")
            return []

        root, all_nodes = tree_data

        # Select strategy
        if strategy is None:
            strategy = self._select_strategy(query_text, root, all_nodes)

        logger.info(f"Retrieving from tree {tree_id} using {strategy.value} strategy")

        # Execute retrieval
        if strategy == RetrievalStrategy.TOP_DOWN:
            results = self._retrieve_top_down(
                query_embedding, root, all_nodes
            )
        elif strategy == RetrievalStrategy.BOTTOM_UP:
            results = self._retrieve_bottom_up(
                query_embedding, root, all_nodes
            )
        elif strategy == RetrievalStrategy.MID_LEVEL:
            results = self._retrieve_mid_level(
                query_embedding, root, all_nodes
            )
        elif strategy == RetrievalStrategy.FULL_TREE:
            results = self._retrieve_full_tree(
                query_embedding, root, all_nodes
            )
        else:  # ADAPTIVE
            results = self._retrieve_adaptive(
                query_embedding, root, all_nodes, query_text
            )

        # Add context if requested
        if self.include_context:
            results = self._add_context(results, all_nodes)

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:self.top_k]

        logger.info(f"Retrieved {len(results)} results")

        return results

    def _select_strategy(
        self,
        query_text: Optional[str],
        root: TreeNode,
        all_nodes: Dict[str, TreeNode]
    ) -> RetrievalStrategy:
        """
        Select retrieval strategy based on query characteristics.

        Args:
            query_text: Query text
            root: Root node
            all_nodes: All nodes

        Returns:
            Selected strategy
        """
        if self.strategy != RetrievalStrategy.ADAPTIVE:
            return self.strategy

        # Analyze query if available
        if query_text:
            # Simple heuristics
            words = query_text.lower().split()

            # Broad queries -> top-down
            if any(w in words for w in ['overview', 'summary', 'general', 'main']):
                return RetrievalStrategy.TOP_DOWN

            # Specific queries -> bottom-up
            if any(w in words for w in ['specific', 'detail', 'exact', 'particular']):
                return RetrievalStrategy.BOTTOM_UP

        # Default to mid-level for balanced retrieval
        return RetrievalStrategy.MID_LEVEL

    def _retrieve_top_down(
        self,
        query_embedding: np.ndarray,
        root: TreeNode,
        all_nodes: Dict[str, TreeNode]
    ) -> List[RetrievalResult]:
        """
        Top-down retrieval: start from root and traverse down.

        Args:
            query_embedding: Query embedding
            root: Root node
            all_nodes: All nodes

        Returns:
            Retrieval results
        """
        results = []
        visited = set()

        # BFS from root
        queue = [(root, 1.0)]  # (node, parent_score)

        while queue:
            node, parent_score = queue.pop(0)

            if node.node_id in visited:
                continue

            visited.add(node.node_id)

            # Compute similarity
            if node.embedding is not None:
                similarity = self._compute_similarity(
                    query_embedding,
                    node.embedding
                )

                # Weight by parent score
                score = similarity * parent_score

                if score >= self.score_threshold:
                    result = RetrievalResult(
                        node=node,
                        score=score,
                        level=node.level
                    )
                    results.append(result)

                    # Add children to queue
                    for child_id in node.children_ids:
                        if child_id in all_nodes:
                            child = all_nodes[child_id]
                            # Propagate score down
                            queue.append((child, score * 0.9))

        return results

    def _retrieve_bottom_up(
        self,
        query_embedding: np.ndarray,
        root: TreeNode,
        all_nodes: Dict[str, TreeNode]
    ) -> List[RetrievalResult]:
        """
        Bottom-up retrieval: start from leaves and aggregate up.

        Args:
            query_embedding: Query embedding
            root: Root node
            all_nodes: All nodes

        Returns:
            Retrieval results
        """
        results = []
        visited = set()

        # Get leaf nodes
        leaf_nodes = [n for n in all_nodes.values() if n.is_leaf]

        # Score all leaves
        leaf_scores = []
        for node in leaf_nodes:
            if node.embedding is not None:
                similarity = self._compute_similarity(
                    query_embedding,
                    node.embedding
                )

                if similarity >= self.score_threshold:
                    leaf_scores.append((node, similarity))

        # Sort leaves by score
        leaf_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top leaves and their ancestors
        for node, score in leaf_scores[:self.top_k * 2]:
            if node.node_id in visited:
                continue

            visited.add(node.node_id)

            # Add leaf result
            result = RetrievalResult(
                node=node,
                score=score,
                level=node.level
            )
            results.append(result)

            # Traverse up to root
            current_node = node
            current_score = score * 0.8  # Decay score as we go up

            while current_node.parent_id and current_node.parent_id in all_nodes:
                parent = all_nodes[current_node.parent_id]

                if parent.node_id not in visited:
                    visited.add(parent.node_id)

                    # Add parent result
                    result = RetrievalResult(
                        node=parent,
                        score=current_score,
                        level=parent.level
                    )
                    results.append(result)

                current_node = parent
                current_score *= 0.8

        return results

    def _retrieve_mid_level(
        self,
        query_embedding: np.ndarray,
        root: TreeNode,
        all_nodes: Dict[str, TreeNode]
    ) -> List[RetrievalResult]:
        """
        Mid-level retrieval: focus on intermediate levels.

        Args:
            query_embedding: Query embedding
            root: Root node
            all_nodes: All nodes

        Returns:
            Retrieval results
        """
        results = []

        # Determine middle level
        max_level = root.level
        mid_level = max_level // 2

        # Get nodes at mid-level
        mid_nodes = [n for n in all_nodes.values() if n.level == mid_level]

        # Score mid-level nodes
        for node in mid_nodes:
            if node.embedding is not None:
                similarity = self._compute_similarity(
                    query_embedding,
                    node.embedding
                )

                if similarity >= self.score_threshold:
                    result = RetrievalResult(
                        node=node,
                        score=similarity,
                        level=node.level
                    )
                    results.append(result)

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)

        # For top results, add their children and parents
        top_results = results[:self.top_k // 2]
        expanded_results = list(results)

        for result in top_results:
            node = result.node

            # Add children
            for child_id in node.children_ids:
                if child_id in all_nodes:
                    child = all_nodes[child_id]

                    if child.embedding is not None:
                        child_similarity = self._compute_similarity(
                            query_embedding,
                            child.embedding
                        )

                        if child_similarity >= self.score_threshold:
                            child_result = RetrievalResult(
                                node=child,
                                score=child_similarity,
                                level=child.level
                            )
                            expanded_results.append(child_result)

            # Add parent
            if node.parent_id and node.parent_id in all_nodes:
                parent = all_nodes[node.parent_id]

                if parent.embedding is not None:
                    parent_similarity = self._compute_similarity(
                        query_embedding,
                        parent.embedding
                    )

                    if parent_similarity >= self.score_threshold:
                        parent_result = RetrievalResult(
                            node=parent,
                            score=parent_similarity,
                            level=parent.level
                        )
                        expanded_results.append(parent_result)

        return expanded_results

    def _retrieve_full_tree(
        self,
        query_embedding: np.ndarray,
        root: TreeNode,
        all_nodes: Dict[str, TreeNode]
    ) -> List[RetrievalResult]:
        """
        Full tree retrieval: search all levels.

        Args:
            query_embedding: Query embedding
            root: Root node
            all_nodes: All nodes

        Returns:
            Retrieval results
        """
        results = []

        # Score all nodes
        for node in all_nodes.values():
            if node.embedding is not None:
                similarity = self._compute_similarity(
                    query_embedding,
                    node.embedding
                )

                if similarity >= self.score_threshold:
                    result = RetrievalResult(
                        node=node,
                        score=similarity,
                        level=node.level
                    )
                    results.append(result)

        return results

    def _retrieve_adaptive(
        self,
        query_embedding: np.ndarray,
        root: TreeNode,
        all_nodes: Dict[str, TreeNode],
        query_text: Optional[str]
    ) -> List[RetrievalResult]:
        """
        Adaptive retrieval: combine multiple strategies.

        Args:
            query_embedding: Query embedding
            root: Root node
            all_nodes: All nodes
            query_text: Query text

        Returns:
            Retrieval results
        """
        # Try multiple strategies and combine results
        strategies = [
            RetrievalStrategy.TOP_DOWN,
            RetrievalStrategy.BOTTOM_UP,
            RetrievalStrategy.MID_LEVEL
        ]

        all_results = []
        seen_nodes = {}

        for strategy in strategies:
            if strategy == RetrievalStrategy.TOP_DOWN:
                strategy_results = self._retrieve_top_down(
                    query_embedding, root, all_nodes
                )
            elif strategy == RetrievalStrategy.BOTTOM_UP:
                strategy_results = self._retrieve_bottom_up(
                    query_embedding, root, all_nodes
                )
            else:  # MID_LEVEL
                strategy_results = self._retrieve_mid_level(
                    query_embedding, root, all_nodes
                )

            # Merge results, keeping best score for each node
            for result in strategy_results:
                node_id = result.node.node_id

                if node_id not in seen_nodes or result.score > seen_nodes[node_id].score:
                    seen_nodes[node_id] = result

        return list(seen_nodes.values())

    def _compute_similarity(
        self,
        query_embedding: np.ndarray,
        node_embedding: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between query and node.

        Args:
            query_embedding: Query embedding
            node_embedding: Node embedding

        Returns:
            Similarity score
        """
        # Normalize embeddings
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        node_norm = node_embedding / np.linalg.norm(node_embedding)

        # Compute cosine similarity
        similarity = np.dot(query_norm, node_norm)

        return float(similarity)

    def _add_context(
        self,
        results: List[RetrievalResult],
        all_nodes: Dict[str, TreeNode]
    ) -> List[RetrievalResult]:
        """
        Add contextual information to results.

        Args:
            results: Retrieval results
            all_nodes: All nodes

        Returns:
            Results with context
        """
        for result in results:
            node = result.node

            # Build path to root
            path = []
            current_id = node.node_id

            while current_id and current_id in all_nodes:
                current_node = all_nodes[current_id]
                path.append(current_id)
                current_id = current_node.parent_id

            result.path_to_root = path

            # Add context information
            result.context = {
                'num_children': len(node.children_ids),
                'num_siblings': len(node.sibling_ids),
                'depth_from_root': len(path) - 1
            }

            # Add parent summary if available
            if node.parent_id and node.parent_id in all_nodes:
                parent = all_nodes[node.parent_id]
                result.context['parent_content'] = parent.content[:200]

            # Add representative children if available
            if node.children_ids:
                child_contents = []
                for child_id in node.children_ids[:3]:  # Top 3 children
                    if child_id in all_nodes:
                        child = all_nodes[child_id]
                        child_contents.append(child.content[:100])

                result.context['children_preview'] = child_contents

        return results

    def get_contextual_passage(
        self,
        result: RetrievalResult,
        all_nodes: Dict[str, TreeNode],
        include_siblings: bool = True,
        include_children: bool = False
    ) -> str:
        """
        Build contextual passage from result.

        Args:
            result: Retrieval result
            all_nodes: All nodes
            include_siblings: Include sibling nodes
            include_children: Include child nodes

        Returns:
            Contextual passage
        """
        parts = []

        # Add path to root for context
        if result.path_to_root:
            for node_id in reversed(result.path_to_root[1:]):  # Skip self
                if node_id in all_nodes:
                    node = all_nodes[node_id]
                    if not node.is_leaf:
                        parts.append(f"[Level {node.level}] {node.content[:200]}")

        # Add main content
        parts.append(f"\n[Main Content]\n{result.node.content}")

        # Add siblings
        if include_siblings and result.node.sibling_ids:
            parts.append("\n[Related Content]")
            for sibling_id in result.node.sibling_ids[:3]:
                if sibling_id in all_nodes:
                    sibling = all_nodes[sibling_id]
                    parts.append(f"- {sibling.content[:150]}")

        # Add children
        if include_children and result.node.children_ids:
            parts.append("\n[Details]")
            for child_id in result.node.children_ids[:5]:
                if child_id in all_nodes:
                    child = all_nodes[child_id]
                    parts.append(f"- {child.content[:150]}")

        return "\n\n".join(parts)

    def retrieve_by_level(
        self,
        query_embedding: np.ndarray,
        tree_id: str,
        target_level: int
    ) -> List[RetrievalResult]:
        """
        Retrieve nodes from specific tree level.

        Args:
            query_embedding: Query embedding
            tree_id: Tree identifier
            target_level: Target level

        Returns:
            Retrieval results from that level
        """
        # Load tree
        tree_data = self.tree_storage.load_tree(tree_id)
        if not tree_data:
            logger.error(f"Tree {tree_id} not found")
            return []

        root, all_nodes = tree_data

        # Get nodes at target level
        level_nodes = [n for n in all_nodes.values() if n.level == target_level]

        results = []

        for node in level_nodes:
            if node.embedding is not None:
                similarity = self._compute_similarity(
                    query_embedding,
                    node.embedding
                )

                if similarity >= self.score_threshold:
                    result = RetrievalResult(
                        node=node,
                        score=similarity,
                        level=node.level
                    )
                    results.append(result)

        # Sort and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:self.top_k]
