"""
Neo4j Graph Database Manager for DocTags RAG System.

Provides comprehensive graph database operations including:
- Connection pool management
- Node and relationship CRUD operations
- Vector similarity search using HNSW indexes
- Graph traversal and pattern matching
- Batch operations for efficiency
- Schema management and validation
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from contextlib import contextmanager
import time
from functools import wraps

from neo4j import GraphDatabase, Driver, Session, Result
from neo4j.exceptions import (
    ServiceUnavailable,
    TransientError,
    Neo4jError,
    AuthError,
    DriverError
)
from loguru import logger

from ..core.config import Neo4jConfig, get_settings


@dataclass
class GraphNode:
    """Represents a node in the graph."""
    id: Optional[str]
    labels: List[str]
    properties: Dict[str, Any]


@dataclass
class GraphRelationship:
    """Represents a relationship in the graph."""
    type: str
    start_node: str
    end_node: str
    properties: Dict[str, Any]


@dataclass
class SearchResult:
    """Represents a search result from the graph."""
    node_id: str
    score: float
    labels: List[str]
    properties: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


def retry_on_transient_error(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry operations on transient errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except TransientError as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        sleep_time = delay * (2 ** attempt)
                        logger.warning(
                            f"Transient error in {func.__name__}, "
                            f"retrying in {sleep_time}s (attempt {attempt + 1}/{max_retries}): {e}"
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.error(f"Max retries reached for {func.__name__}")
                except Exception as e:
                    logger.error(f"Non-transient error in {func.__name__}: {e}")
                    raise
            raise last_exception
        return wrapper
    return decorator


class Neo4jManager:
    """
    Manages Neo4j graph database operations with connection pooling and error handling.

    Supports:
    - CRUD operations for nodes and relationships
    - Vector similarity search with HNSW indexes
    - Graph traversal and pattern matching
    - Batch operations
    - Transaction management
    """

    def __init__(self, config: Optional[Neo4jConfig] = None):
        """
        Initialize Neo4j manager.

        Args:
            config: Neo4j configuration. If None, loads from global settings.
        """
        if config is None:
            settings = get_settings()
            config = settings.neo4j

        self.config = config
        self.driver: Optional[Driver] = None
        self._connected = False

        # Initialize connection
        self._initialize_driver()

    def _initialize_driver(self) -> None:
        """Initialize the Neo4j driver with connection pool."""
        try:
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_timeout=self.config.connection_timeout,
                max_connection_lifetime=3600,  # 1 hour
                keep_alive=True
            )
            # Verify connectivity
            self.driver.verify_connectivity()
            self._connected = True
            logger.info(f"Neo4j driver initialized: {self.config.uri}")
        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {e}")
            raise
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {e}")
            raise

    @contextmanager
    def get_session(self, database: Optional[str] = None) -> Session:
        """
        Context manager for Neo4j sessions.

        Args:
            database: Database name. If None, uses config default.

        Yields:
            Neo4j session
        """
        if not self._connected or self.driver is None:
            raise RuntimeError("Neo4j driver not connected")

        db = database or self.config.database
        session = self.driver.session(database=db)
        try:
            yield session
        finally:
            session.close()

    def health_check(self) -> bool:
        """
        Check if Neo4j connection is healthy.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            if not self.driver:
                return False
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.warning(f"Neo4j health check failed: {e}")
            return False

    def close(self) -> None:
        """Close the Neo4j driver and cleanup resources."""
        if self.driver:
            self.driver.close()
            self._connected = False
            logger.info("Neo4j driver closed")

    @retry_on_transient_error(max_retries=3)
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query safely with parameterization.

        Args:
            query: Cypher query string
            parameters: Query parameters for safe parameterization
            database: Database name (optional)

        Returns:
            List of result records as dictionaries
        """
        with self.get_session(database) as session:
            try:
                result = session.run(query, parameters or {})
                return [dict(record) for record in result]
            except Neo4jError as e:
                logger.error(f"Query execution failed: {e}\nQuery: {query}")
                raise

    @retry_on_transient_error(max_retries=3)
    def execute_write_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a write query in a transaction.

        Args:
            query: Cypher query string
            parameters: Query parameters
            database: Database name (optional)

        Returns:
            List of result records
        """
        with self.get_session(database) as session:
            try:
                result = session.execute_write(
                    lambda tx: list(tx.run(query, parameters or {}))
                )
                return [dict(record) for record in result]
            except Neo4jError as e:
                logger.error(f"Write query failed: {e}\nQuery: {query}")
                raise

    def create_node(
        self,
        labels: List[str],
        properties: Dict[str, Any],
        return_node: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Create a node with labels and properties.

        Args:
            labels: List of node labels
            properties: Node properties
            return_node: Whether to return the created node

        Returns:
            Created node data if return_node is True
        """
        labels_str = ":".join(labels)
        query = f"""
        CREATE (n:{labels_str})
        SET n = $properties
        {'RETURN n' if return_node else ''}
        """

        result = self.execute_write_query(query, {"properties": properties})
        return result[0] if result and return_node else None

    def create_nodes_batch(
        self,
        nodes: List[GraphNode],
        batch_size: int = 1000
    ) -> int:
        """
        Create multiple nodes in batches for efficiency.

        Args:
            nodes: List of GraphNode objects
            batch_size: Number of nodes per batch

        Returns:
            Number of nodes created
        """
        total_created = 0

        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]

            # Group by labels for efficient batch creation
            nodes_by_label = {}
            for node in batch:
                label_key = ":".join(sorted(node.labels))
                if label_key not in nodes_by_label:
                    nodes_by_label[label_key] = []
                nodes_by_label[label_key].append(node.properties)

            for labels, props_list in nodes_by_label.items():
                query = f"""
                UNWIND $props_list AS props
                CREATE (n:{labels})
                SET n = props
                """
                self.execute_write_query(query, {"props_list": props_list})
                total_created += len(props_list)

        logger.info(f"Created {total_created} nodes in batches")
        return total_created

    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a node by its ID.

        Args:
            node_id: Node ID

        Returns:
            Node data or None if not found
        """
        query = """
        MATCH (n)
        WHERE elementId(n) = $node_id
        RETURN n, labels(n) as labels
        """
        result = self.execute_query(query, {"node_id": node_id})
        return result[0] if result else None

    def update_node(
        self,
        node_id: str,
        properties: Dict[str, Any],
        merge: bool = True
    ) -> bool:
        """
        Update a node's properties.

        Args:
            node_id: Node ID
            properties: Properties to update
            merge: If True, merge with existing properties. If False, replace.

        Returns:
            True if node was updated
        """
        if merge:
            query = """
            MATCH (n)
            WHERE elementId(n) = $node_id
            SET n += $properties
            RETURN n
            """
        else:
            query = """
            MATCH (n)
            WHERE elementId(n) = $node_id
            SET n = $properties
            RETURN n
            """

        result = self.execute_write_query(query, {
            "node_id": node_id,
            "properties": properties
        })
        return len(result) > 0

    def delete_node(self, node_id: str, detach: bool = True) -> bool:
        """
        Delete a node.

        Args:
            node_id: Node ID
            detach: If True, also delete relationships

        Returns:
            True if node was deleted
        """
        query = f"""
        MATCH (n)
        WHERE elementId(n) = $node_id
        {'DETACH ' if detach else ''}DELETE n
        """

        try:
            self.execute_write_query(query, {"node_id": node_id})
            return True
        except Neo4jError:
            return False

    def create_relationship(
        self,
        start_node_id: str,
        end_node_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create a relationship between two nodes.

        Args:
            start_node_id: Start node ID
            end_node_id: End node ID
            rel_type: Relationship type
            properties: Relationship properties

        Returns:
            Created relationship data
        """
        query = f"""
        MATCH (a), (b)
        WHERE elementId(a) = $start_id AND elementId(b) = $end_id
        CREATE (a)-[r:{rel_type}]->(b)
        SET r = $properties
        RETURN r, elementId(r) as rel_id
        """

        result = self.execute_write_query(query, {
            "start_id": start_node_id,
            "end_id": end_node_id,
            "properties": properties or {}
        })
        return result[0] if result else None

    def create_relationships_batch(
        self,
        relationships: List[GraphRelationship],
        batch_size: int = 1000
    ) -> int:
        """
        Create multiple relationships in batches.

        Args:
            relationships: List of GraphRelationship objects
            batch_size: Number of relationships per batch

        Returns:
            Number of relationships created
        """
        total_created = 0

        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i + batch_size]

            # Group by type for efficiency
            rels_by_type = {}
            for rel in batch:
                if rel.type not in rels_by_type:
                    rels_by_type[rel.type] = []
                rels_by_type[rel.type].append({
                    "start_id": rel.start_node,
                    "end_id": rel.end_node,
                    "properties": rel.properties
                })

            for rel_type, rels_data in rels_by_type.items():
                query = f"""
                UNWIND $rels AS rel
                MATCH (a), (b)
                WHERE elementId(a) = rel.start_id AND elementId(b) = rel.end_id
                CREATE (a)-[r:{rel_type}]->(b)
                SET r = rel.properties
                """
                self.execute_write_query(query, {"rels": rels_data})
                total_created += len(rels_data)

        logger.info(f"Created {total_created} relationships in batches")
        return total_created

    def initialize_vector_index(
        self,
        index_name: str,
        label: str,
        property_name: str,
        dimensions: int,
        similarity_function: str = "cosine"
    ) -> bool:
        """
        Create an HNSW vector index for similarity search.

        Args:
            index_name: Name of the index
            label: Node label to index
            property_name: Property containing the vector
            dimensions: Vector dimensions
            similarity_function: Similarity function (cosine, euclidean)

        Returns:
            True if index was created
        """
        # Check if index exists
        check_query = """
        SHOW INDEXES
        YIELD name
        WHERE name = $index_name
        RETURN count(*) as count
        """
        result = self.execute_query(check_query, {"index_name": index_name})

        if result and result[0]["count"] > 0:
            logger.info(f"Vector index '{index_name}' already exists")
            return True

        # Create vector index
        query = f"""
        CREATE VECTOR INDEX {index_name}
        FOR (n:{label})
        ON n.{property_name}
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {dimensions},
                `vector.similarity_function`: '{similarity_function}'
            }}
        }}
        """

        try:
            self.execute_query(query)
            logger.info(f"Created vector index '{index_name}'")
            return True
        except Neo4jError as e:
            logger.error(f"Failed to create vector index: {e}")
            return False

    def vector_similarity_search(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform vector similarity search using HNSW index.

        Args:
            index_name: Name of the vector index
            query_vector: Query vector
            top_k: Number of results to return
            filters: Optional property filters

        Returns:
            List of search results
        """
        # Build filter clause
        filter_clause = ""
        if filters:
            conditions = [f"n.{key} = ${key}" for key in filters.keys()]
            filter_clause = "WHERE " + " AND ".join(conditions)

        query = f"""
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        {filter_clause}
        RETURN elementId(node) as node_id, score, labels(node) as labels, node
        ORDER BY score DESC
        """

        params = {
            "index_name": index_name,
            "top_k": top_k,
            "query_vector": query_vector
        }
        if filters:
            params.update(filters)

        results = self.execute_query(query, params)

        return [
            SearchResult(
                node_id=r["node_id"],
                score=r["score"],
                labels=r["labels"],
                properties=dict(r["node"])
            )
            for r in results
        ]

    def traverse_graph(
        self,
        start_node_id: str,
        relationship_types: Optional[List[str]] = None,
        direction: str = "both",
        max_depth: int = 3,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Traverse the graph from a starting node.

        Args:
            start_node_id: Starting node ID
            relationship_types: List of relationship types to follow
            direction: Direction to traverse (outgoing, incoming, both)
            max_depth: Maximum traversal depth
            limit: Maximum number of results

        Returns:
            List of nodes and relationships in the path
        """
        # Build relationship pattern
        rel_types = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)

        if direction == "outgoing":
            pattern = f"-[r:{rel_types}*1..{max_depth}]->"
        elif direction == "incoming":
            pattern = f"<-[r:{rel_types}*1..{max_depth}]-"
        else:
            pattern = f"-[r:{rel_types}*1..{max_depth}]-"

        query = f"""
        MATCH path = (start){pattern}(end)
        WHERE elementId(start) = $start_node_id
        RETURN path
        LIMIT $limit
        """

        results = self.execute_query(query, {
            "start_node_id": start_node_id,
            "limit": limit
        })

        return results

    def find_shortest_path(
        self,
        start_node_id: str,
        end_node_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Find shortest path between two nodes.

        Args:
            start_node_id: Start node ID
            end_node_id: End node ID
            relationship_types: Relationship types to traverse
            max_depth: Maximum path length

        Returns:
            Shortest path or None if no path exists
        """
        rel_types = ""
        if relationship_types:
            rel_types = ":" + "|".join(relationship_types)

        query = f"""
        MATCH path = shortestPath(
            (start)-[{rel_types}*1..{max_depth}]-(end)
        )
        WHERE elementId(start) = $start_id AND elementId(end) = $end_id
        RETURN path, length(path) as path_length
        """

        result = self.execute_query(query, {
            "start_id": start_node_id,
            "end_id": end_node_id
        })

        return result[0] if result else None

    def pattern_match(
        self,
        pattern: str,
        parameters: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Execute a pattern matching query.

        Args:
            pattern: Cypher pattern (e.g., "(a:Person)-[:KNOWS]->(b:Person)")
            parameters: Query parameters
            limit: Maximum results

        Returns:
            List of matching patterns
        """
        query = f"""
        MATCH {pattern}
        RETURN *
        LIMIT $limit
        """

        params = parameters or {}
        params["limit"] = limit

        return self.execute_query(query, params)

    def create_schema_constraints(self) -> None:
        """Create schema constraints for data integrity."""
        constraints = [
            # Unique document IDs
            """
            CREATE CONSTRAINT document_id_unique IF NOT EXISTS
            FOR (d:Document)
            REQUIRE d.doc_id IS UNIQUE
            """,
            # Unique section IDs
            """
            CREATE CONSTRAINT section_id_unique IF NOT EXISTS
            FOR (s:Section)
            REQUIRE s.section_id IS UNIQUE
            """,
            # Unique entity names per type
            """
            CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
            FOR (e:Entity)
            REQUIRE e.name IS UNIQUE
            """,
        ]

        for constraint in constraints:
            try:
                self.execute_query(constraint)
                logger.info(f"Created constraint")
            except Neo4jError as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Constraint creation failed: {e}")

    def create_indexes(self) -> None:
        """Create indexes for common query patterns."""
        indexes = [
            # Index on document properties
            """
            CREATE INDEX document_title IF NOT EXISTS
            FOR (d:Document)
            ON (d.title)
            """,
            # Index on entity types
            """
            CREATE INDEX entity_type IF NOT EXISTS
            FOR (e:Entity)
            ON (e.type)
            """,
            # Index on timestamps
            """
            CREATE INDEX document_created IF NOT EXISTS
            FOR (d:Document)
            ON (d.created_at)
            """,
        ]

        for index in indexes:
            try:
                self.execute_query(index)
                logger.info(f"Created index")
            except Neo4jError as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Index creation failed: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with node counts, relationship counts, etc.
        """
        stats = {}

        # Node count by label
        query = """
        MATCH (n)
        RETURN labels(n) as labels, count(*) as count
        """
        results = self.execute_query(query)
        stats["nodes_by_label"] = {
            ":".join(r["labels"]): r["count"] for r in results
        }

        # Relationship count by type
        query = """
        MATCH ()-[r]->()
        RETURN type(r) as type, count(*) as count
        """
        results = self.execute_query(query)
        stats["relationships_by_type"] = {
            r["type"]: r["count"] for r in results
        }

        # Total counts
        query = """
        MATCH (n)
        RETURN count(n) as node_count
        """
        result = self.execute_query(query)
        stats["total_nodes"] = result[0]["node_count"] if result else 0

        query = """
        MATCH ()-[r]->()
        RETURN count(r) as rel_count
        """
        result = self.execute_query(query)
        stats["total_relationships"] = result[0]["rel_count"] if result else 0

        return stats

    def clear_database(self, confirm: bool = False) -> bool:
        """
        Clear all data from the database.

        Args:
            confirm: Must be True to execute

        Returns:
            True if database was cleared
        """
        if not confirm:
            logger.warning("Database clear requires confirmation")
            return False

        query = """
        MATCH (n)
        DETACH DELETE n
        """

        try:
            self.execute_write_query(query)
            logger.warning("Database cleared")
            return True
        except Neo4jError as e:
            logger.error(f"Failed to clear database: {e}")
            return False
