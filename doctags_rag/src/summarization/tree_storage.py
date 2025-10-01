"""
Tree Storage Manager for RAPTOR System.

Handles persistence of hierarchical trees in:
- Neo4j for tree structure and relationships
- Qdrant for vector embeddings per level
- Metadata management and versioning
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import numpy as np
from loguru import logger

from ..knowledge_graph.neo4j_manager import Neo4jManager
from ..retrieval.qdrant_manager import QdrantManager
from .tree_builder import TreeNode, TreeStats


class TreeStorage:
    """
    Manages storage and retrieval of RAPTOR trees.

    Stores:
    - Tree structure in Neo4j
    - Node embeddings in Qdrant (separate collections per level)
    - Tree metadata and statistics
    """

    def __init__(
        self,
        neo4j_manager: Neo4jManager,
        qdrant_manager: QdrantManager,
        collection_prefix: str = "raptor"
    ):
        """
        Initialize tree storage.

        Args:
            neo4j_manager: Neo4j manager instance
            qdrant_manager: Qdrant manager instance
            collection_prefix: Prefix for Qdrant collections
        """
        self.neo4j = neo4j_manager
        self.qdrant = qdrant_manager
        self.collection_prefix = collection_prefix

        # Track created collections
        self.level_collections = {}

        logger.info("TreeStorage initialized")

    def save_tree(
        self,
        root: TreeNode,
        all_nodes: Dict[str, TreeNode],
        tree_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save complete tree to storage.

        Args:
            root: Root node
            all_nodes: All nodes in tree
            tree_id: Unique tree identifier
            metadata: Additional tree metadata

        Returns:
            Success status
        """
        logger.info(f"Saving tree {tree_id} with {len(all_nodes)} nodes")

        try:
            # Save tree metadata
            self._save_tree_metadata(tree_id, root, all_nodes, metadata)

            # Save nodes to Neo4j
            self._save_nodes_to_neo4j(tree_id, all_nodes)

            # Save relationships to Neo4j
            self._save_relationships_to_neo4j(tree_id, all_nodes)

            # Save embeddings to Qdrant
            self._save_embeddings_to_qdrant(tree_id, all_nodes)

            logger.info(f"Tree {tree_id} saved successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to save tree {tree_id}: {e}")
            return False

    def load_tree(
        self,
        tree_id: str
    ) -> Optional[Tuple[TreeNode, Dict[str, TreeNode]]]:
        """
        Load complete tree from storage.

        Args:
            tree_id: Tree identifier

        Returns:
            Tuple of (root, all_nodes) or None if not found
        """
        logger.info(f"Loading tree {tree_id}")

        try:
            # Load nodes from Neo4j
            all_nodes = self._load_nodes_from_neo4j(tree_id)

            if not all_nodes:
                logger.warning(f"No nodes found for tree {tree_id}")
                return None

            # Load embeddings from Qdrant
            self._load_embeddings_from_qdrant(tree_id, all_nodes)

            # Find root node
            root = None
            for node in all_nodes.values():
                if node.parent_id is None or node.metadata.get('is_root', False):
                    root = node
                    break

            if not root:
                logger.error(f"No root node found for tree {tree_id}")
                return None

            logger.info(f"Tree {tree_id} loaded successfully: {len(all_nodes)} nodes")
            return root, all_nodes

        except Exception as e:
            logger.error(f"Failed to load tree {tree_id}: {e}")
            return None

    def delete_tree(self, tree_id: str) -> bool:
        """
        Delete tree from storage.

        Args:
            tree_id: Tree identifier

        Returns:
            Success status
        """
        logger.info(f"Deleting tree {tree_id}")

        try:
            # Delete from Neo4j
            query = """
            MATCH (n:RAPTORNode {tree_id: $tree_id})
            DETACH DELETE n
            """
            self.neo4j.execute_query(query, {'tree_id': tree_id})

            # Delete metadata
            query = """
            MATCH (m:RAPTORTree {tree_id: $tree_id})
            DELETE m
            """
            self.neo4j.execute_query(query, {'tree_id': tree_id})

            # Delete from Qdrant
            # Would need to track which collections were used
            # For now, just log
            logger.warning("Qdrant cleanup not fully implemented")

            logger.info(f"Tree {tree_id} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to delete tree {tree_id}: {e}")
            return False

    def _save_tree_metadata(
        self,
        tree_id: str,
        root: TreeNode,
        all_nodes: Dict[str, TreeNode],
        metadata: Optional[Dict[str, Any]]
    ):
        """Save tree metadata to Neo4j."""
        from ..summarization.tree_builder import TreeBuilder

        # Compute stats (would need TreeBuilder instance for this)
        stats = {
            'total_nodes': len(all_nodes),
            'max_depth': root.level,
            'leaf_nodes': sum(1 for n in all_nodes.values() if n.is_leaf)
        }

        tree_metadata = {
            'tree_id': tree_id,
            'root_node_id': root.node_id,
            'created_at': datetime.now().isoformat(),
            'stats': json.dumps(stats),
            **(metadata or {})
        }

        query = """
        MERGE (t:RAPTORTree {tree_id: $tree_id})
        SET t += $metadata
        RETURN t
        """

        self.neo4j.execute_query(
            query,
            {
                'tree_id': tree_id,
                'metadata': tree_metadata
            }
        )

    def _save_nodes_to_neo4j(
        self,
        tree_id: str,
        all_nodes: Dict[str, TreeNode]
    ):
        """Save all nodes to Neo4j."""
        logger.info(f"Saving {len(all_nodes)} nodes to Neo4j")

        # Batch insert nodes
        batch_size = 100
        node_list = list(all_nodes.values())

        for i in range(0, len(node_list), batch_size):
            batch = node_list[i:i + batch_size]

            # Prepare node data
            nodes_data = []
            for node in batch:
                node_data = {
                    'node_id': node.node_id,
                    'tree_id': tree_id,
                    'content': node.content,
                    'level': node.level,
                    'is_leaf': node.is_leaf,
                    'parent_id': node.parent_id,
                    'metadata': json.dumps(node.metadata)
                }

                if node.summary:
                    node_data['summary_quality'] = node.summary.quality_score
                    node_data['key_facts'] = json.dumps(node.summary.key_facts)
                    node_data['entities'] = json.dumps(node.summary.entities)

                nodes_data.append(node_data)

            # Insert batch
            query = """
            UNWIND $nodes AS node
            CREATE (n:RAPTORNode)
            SET n = node
            """

            self.neo4j.execute_query(query, {'nodes': nodes_data})

        logger.info(f"Saved {len(all_nodes)} nodes to Neo4j")

    def _save_relationships_to_neo4j(
        self,
        tree_id: str,
        all_nodes: Dict[str, TreeNode]
    ):
        """Save node relationships to Neo4j."""
        logger.info("Saving node relationships to Neo4j")

        # Collect relationships
        relationships = []

        for node in all_nodes.values():
            # Parent-child relationships
            if node.parent_id:
                relationships.append({
                    'from_id': node.node_id,
                    'to_id': node.parent_id,
                    'type': 'HAS_PARENT'
                })

            # Sibling relationships
            for sibling_id in node.sibling_ids:
                # Only create one direction to avoid duplicates
                if node.node_id < sibling_id:
                    relationships.append({
                        'from_id': node.node_id,
                        'to_id': sibling_id,
                        'type': 'SIBLING_OF'
                    })

        # Batch insert relationships
        batch_size = 500
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i + batch_size]

            query = """
            UNWIND $rels AS rel
            MATCH (from:RAPTORNode {node_id: rel.from_id, tree_id: $tree_id})
            MATCH (to:RAPTORNode {node_id: rel.to_id, tree_id: $tree_id})
            CREATE (from)-[r:TREE_RELATIONSHIP {type: rel.type}]->(to)
            """

            self.neo4j.execute_query(
                query,
                {'rels': batch, 'tree_id': tree_id}
            )

        logger.info(f"Saved {len(relationships)} relationships to Neo4j")

    def _save_embeddings_to_qdrant(
        self,
        tree_id: str,
        all_nodes: Dict[str, TreeNode]
    ):
        """Save node embeddings to Qdrant."""
        logger.info("Saving embeddings to Qdrant")

        # Group nodes by level
        nodes_by_level = {}
        for node in all_nodes.values():
            if node.embedding is not None:
                level = node.level
                if level not in nodes_by_level:
                    nodes_by_level[level] = []
                nodes_by_level[level].append(node)

        # Save each level to separate collection
        for level, nodes in nodes_by_level.items():
            collection_name = f"{self.collection_prefix}_L{level}"

            # Ensure collection exists
            self._ensure_collection_exists(
                collection_name,
                len(nodes[0].embedding)
            )

            # Prepare points
            points = []
            for node in nodes:
                point = {
                    'id': hash(node.node_id) & 0x7FFFFFFFFFFFFFFF,  # Positive int
                    'vector': node.embedding.tolist(),
                    'payload': {
                        'node_id': node.node_id,
                        'tree_id': tree_id,
                        'level': level,
                        'is_leaf': node.is_leaf,
                        'content_preview': node.content[:200]
                    }
                }
                points.append(point)

            # Upsert points
            self.qdrant.upsert_vectors(
                collection_name=collection_name,
                points=points
            )

            logger.info(f"Saved {len(points)} embeddings for level {level}")

    def _load_nodes_from_neo4j(
        self,
        tree_id: str
    ) -> Dict[str, TreeNode]:
        """Load all nodes from Neo4j."""
        query = """
        MATCH (n:RAPTORNode {tree_id: $tree_id})
        RETURN n
        """

        result = self.neo4j.execute_query(query, {'tree_id': tree_id})

        all_nodes = {}

        for record in result:
            node_data = dict(record['n'])

            # Parse JSON fields
            metadata = json.loads(node_data.get('metadata', '{}'))

            # Create node
            node = TreeNode(
                node_id=node_data['node_id'],
                content=node_data['content'],
                level=node_data['level'],
                is_leaf=node_data['is_leaf'],
                parent_id=node_data.get('parent_id'),
                metadata=metadata
            )

            all_nodes[node.node_id] = node

        # Load relationships
        self._load_relationships_from_neo4j(tree_id, all_nodes)

        return all_nodes

    def _load_relationships_from_neo4j(
        self,
        tree_id: str,
        all_nodes: Dict[str, TreeNode]
    ):
        """Load node relationships from Neo4j."""
        query = """
        MATCH (from:RAPTORNode {tree_id: $tree_id})-[r:TREE_RELATIONSHIP]->(to:RAPTORNode)
        RETURN from.node_id AS from_id, to.node_id AS to_id, r.type AS rel_type
        """

        result = self.neo4j.execute_query(query, {'tree_id': tree_id})

        for record in result:
            from_id = record['from_id']
            to_id = record['to_id']
            rel_type = record['rel_type']

            if from_id in all_nodes and to_id in all_nodes:
                from_node = all_nodes[from_id]

                if rel_type == 'HAS_PARENT':
                    # Already set in node data
                    pass
                elif rel_type == 'SIBLING_OF':
                    from_node.add_sibling(to_id)
                    # Add reverse relationship
                    all_nodes[to_id].add_sibling(from_id)

        # Build children lists
        for node in all_nodes.values():
            if node.parent_id and node.parent_id in all_nodes:
                parent = all_nodes[node.parent_id]
                parent.add_child(node.node_id)

    def _load_embeddings_from_qdrant(
        self,
        tree_id: str,
        all_nodes: Dict[str, TreeNode]
    ):
        """Load node embeddings from Qdrant."""
        # Group nodes by level
        nodes_by_level = {}
        for node in all_nodes.values():
            level = node.level
            if level not in nodes_by_level:
                nodes_by_level[level] = []
            nodes_by_level[level].append(node)

        # Load embeddings for each level
        for level, nodes in nodes_by_level.items():
            collection_name = f"{self.collection_prefix}_L{level}"

            # Check if collection exists
            try:
                collections = self.qdrant.client.get_collections()
                collection_exists = any(
                    c.name == collection_name
                    for c in collections.collections
                )

                if not collection_exists:
                    logger.warning(f"Collection {collection_name} does not exist")
                    continue

                # Retrieve embeddings for all nodes
                for node in nodes:
                    point_id = hash(node.node_id) & 0x7FFFFFFFFFFFFFFF

                    try:
                        point = self.qdrant.client.retrieve(
                            collection_name=collection_name,
                            ids=[point_id]
                        )

                        if point:
                            node.embedding = np.array(point[0].vector)

                    except Exception as e:
                        logger.warning(f"Could not load embedding for {node.node_id}: {e}")

            except Exception as e:
                logger.error(f"Error loading embeddings from {collection_name}: {e}")

    def _ensure_collection_exists(
        self,
        collection_name: str,
        vector_size: int
    ):
        """Ensure Qdrant collection exists."""
        if collection_name in self.level_collections:
            return

        try:
            collections = self.qdrant.client.get_collections()
            collection_exists = any(
                c.name == collection_name
                for c in collections.collections
            )

            if not collection_exists:
                from qdrant_client.models import Distance, VectorParams

                self.qdrant.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection {collection_name}")

            self.level_collections[collection_name] = True

        except Exception as e:
            logger.error(f"Failed to ensure collection {collection_name}: {e}")
            raise

    def list_trees(self) -> List[Dict[str, Any]]:
        """
        List all stored trees.

        Returns:
            List of tree metadata
        """
        query = """
        MATCH (t:RAPTORTree)
        RETURN t
        ORDER BY t.created_at DESC
        """

        result = self.neo4j.execute_query(query, {})

        trees = []
        for record in result:
            tree_data = dict(record['t'])
            # Parse JSON stats
            if 'stats' in tree_data:
                tree_data['stats'] = json.loads(tree_data['stats'])
            trees.append(tree_data)

        return trees

    def get_tree_metadata(self, tree_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific tree.

        Args:
            tree_id: Tree identifier

        Returns:
            Tree metadata or None
        """
        query = """
        MATCH (t:RAPTORTree {tree_id: $tree_id})
        RETURN t
        """

        result = self.neo4j.execute_query(query, {'tree_id': tree_id})

        if not result:
            return None

        tree_data = dict(result[0]['t'])
        if 'stats' in tree_data:
            tree_data['stats'] = json.loads(tree_data['stats'])

        return tree_data

    def update_node(
        self,
        tree_id: str,
        node_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Update a specific node.

        Args:
            tree_id: Tree identifier
            node_id: Node identifier
            updates: Fields to update

        Returns:
            Success status
        """
        try:
            # Update Neo4j
            query = """
            MATCH (n:RAPTORNode {tree_id: $tree_id, node_id: $node_id})
            SET n += $updates
            RETURN n
            """

            result = self.neo4j.execute_query(
                query,
                {
                    'tree_id': tree_id,
                    'node_id': node_id,
                    'updates': updates
                }
            )

            return len(result) > 0

        except Exception as e:
            logger.error(f"Failed to update node {node_id}: {e}")
            return False
