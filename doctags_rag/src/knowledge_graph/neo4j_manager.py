"""
Neo4j Graph Database Manager for Contextprime.

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
import re
import json
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

SAFE_PROPERTY_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
SAFE_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_]{1,64}$")
SAFE_RELATIONSHIP_TYPE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")


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
        labels_str = ":".join(self._sanitize_labels(labels))
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
        identity_keys = ("doc_id", "chunk_id", "entity_id", "canonical_name", "name")

        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]

            # Group by labels for efficient batch creation
            nodes_by_label = {}
            for node in batch:
                sanitized_labels = sorted(self._sanitize_labels(node.labels))
                label_key = ":".join(sanitized_labels)
                if label_key not in nodes_by_label:
                    nodes_by_label[label_key] = []
                nodes_by_label[label_key].append(node.properties)

            for labels, props_list in nodes_by_label.items():
                identity_key = None
                for candidate in identity_keys:
                    if all(props.get(candidate) is not None for props in props_list):
                        identity_key = candidate
                        break

                if identity_key:
                    query = f"""
                    UNWIND $props_list AS props
                    MERGE (n:{labels} {{{identity_key}: props.{identity_key}}})
                    SET n += props
                    """
                else:
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
        filter_clause, filter_params = self._build_safe_property_filters(filters)

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
        params.update(filter_params)

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

    def _build_safe_property_filters(
        self,
        filters: Optional[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        """Build a safe Cypher WHERE clause for node property filters."""
        if not filters:
            return "", {}

        clauses: List[str] = []
        params: Dict[str, Any] = {}

        for idx, (raw_key, raw_value) in enumerate(filters.items()):
            key = self._sanitize_property_key(raw_key)
            property_ref = f"n.`{key}`"

            if raw_value is None:
                clauses.append(f"{property_ref} IS NULL")
                continue

            param_name = f"filter_value_{idx}"
            if isinstance(raw_value, (list, tuple, set)):
                values = list(raw_value)
                if not values:
                    continue
                clauses.append(f"{property_ref} IN ${param_name}")
                params[param_name] = values
                continue

            clauses.append(f"{property_ref} = ${param_name}")
            params[param_name] = raw_value

        if not clauses:
            return "", {}
        return "WHERE " + " AND ".join(clauses), params

    def _sanitize_property_key(self, raw_key: Any) -> str:
        key = str(raw_key).strip()
        if not SAFE_PROPERTY_KEY_PATTERN.fullmatch(key):
            raise ValueError(f"Invalid filter key: {raw_key}")
        return key

    def _sanitize_labels(self, labels: List[Any]) -> List[str]:
        """Validate and normalize labels before interpolation into Cypher."""
        if not labels:
            raise ValueError("At least one label is required")

        sanitized: List[str] = []
        seen = set()
        for raw_label in labels:
            label = str(raw_label).strip()
            if not SAFE_LABEL_PATTERN.fullmatch(label):
                raise ValueError(f"Invalid label: {raw_label}")
            if label in seen:
                continue
            seen.add(label)
            sanitized.append(label)

        if not sanitized:
            raise ValueError("At least one label is required")
        return sanitized

    def _sanitize_relationship_type(self, raw_type: Any) -> str:
        rel_type = str(raw_type).strip()
        if not SAFE_RELATIONSHIP_TYPE_PATTERN.fullmatch(rel_type):
            raise ValueError(f"Invalid relationship type: {raw_type}")
        return rel_type

    def store_cross_references(
        self,
        refs: List[Any],
        batch_size: int = 500,
    ) -> int:
        """Store cross-reference edges between Chunk nodes.

        Creates (:Chunk {chunk_id: source})-[:REFERENCES {ref_type, target_label}]->
        (:Chunk {chunk_id: target}) edges using MERGE to remain idempotent.

        When the target chunk does not yet exist in Neo4j (e.g. it is referenced
        but not yet ingested) the relationship is silently skipped so the ingestion
        pipeline never fails on unresolvable references.

        Args:
            refs:       List of CrossRef dataclass instances
                        (duck-typed: needs .source_chunk_id, .target_label,
                         .ref_type, .doc_id attributes).
            batch_size: Number of edges to write per transaction.

        Returns:
            Number of edges created or merged.
        """
        if not refs:
            return 0

        # Build parameter list
        params_list = [
            {
                "source_chunk_id": ref.source_chunk_id,
                "target_label": ref.target_label,
                "ref_type": ref.ref_type,
                "doc_id": ref.doc_id,
            }
            for ref in refs
        ]

        total_merged = 0
        for i in range(0, len(params_list), batch_size):
            batch = params_list[i : i + batch_size]
            query = """
            UNWIND $batch AS ref
            MATCH (src:Chunk {chunk_id: ref.source_chunk_id})
            MATCH (tgt:Chunk)
            WHERE tgt.doc_id = ref.doc_id
              AND (
                tgt.content STARTS WITH ref.target_label
                OR tgt.chunk_id CONTAINS ref.target_label
                OR tgt.metadata IS NOT NULL AND tgt.metadata CONTAINS ref.target_label
              )
            MERGE (src)-[r:REFERENCES {ref_type: ref.ref_type, target_label: ref.target_label}]->(tgt)
            RETURN count(r) AS merged
            """
            try:
                result = self.execute_write_query(query, {"batch": batch})
                if result:
                    total_merged += int(result[0].get("merged", 0))
            except Exception as exc:
                logger.warning(
                    "store_cross_references batch %d failed (skipping): %s",
                    i // batch_size,
                    exc,
                )

        logger.info("Stored %d cross-reference edges", total_merged)
        return total_merged

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
            rel_types = "|".join(
                self._sanitize_relationship_type(raw_type)
                for raw_type in relationship_types
            )

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

    def expand_from_seed_nodes(
        self,
        seed_scores: Dict[str, float],
        max_depth: int = 2,
        limit: int = 100,
    ) -> List[SearchResult]:
        """
        Expand graph neighborhood from seed nodes and return scored neighbors.

        Args:
            seed_scores: Mapping of seed node id to seed score
            max_depth: Maximum traversal depth from each seed
            limit: Maximum neighbor records to return before scoring

        Returns:
            Scored neighbor search results
        """
        if not seed_scores:
            return []

        query = """
        UNWIND $seed_ids AS seed_id
        MATCH (seed)
        WHERE elementId(seed) = seed_id
        MATCH path = (seed)-[*1..$max_depth]-(neighbor)
        WHERE elementId(neighbor) <> seed_id
        WITH seed_id, neighbor, min(length(path)) AS depth
        RETURN
            seed_id,
            elementId(neighbor) AS node_id,
            labels(neighbor) AS labels,
            neighbor,
            depth
        LIMIT $limit
        """

        rows = self.execute_query(
            query,
            {
                "seed_ids": list(seed_scores.keys()),
                "max_depth": max_depth,
                "limit": limit,
            },
        )

        if not rows:
            return []

        grouped: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            node_id = row.get("node_id")
            if not node_id:
                continue
            seed_id = row.get("seed_id")
            depth = max(1, int(row.get("depth", 1)))
            seed_score = float(seed_scores.get(seed_id, 0.5))
            neighbor_score = seed_score / (1.0 + depth)

            existing = grouped.get(node_id)
            if existing is None or neighbor_score > existing["score"]:
                grouped[node_id] = {
                    "score": neighbor_score,
                    "labels": row.get("labels", []),
                    "node": dict(row.get("neighbor") or {}),
                    "seed_id": seed_id,
                    "depth": depth,
                }

        expanded: List[SearchResult] = []
        for node_id, item in grouped.items():
            metadata = {
                "seed_node_id": item["seed_id"],
                "graph_depth": item["depth"],
                "graph_signal": "local_expansion",
            }
            expanded.append(
                SearchResult(
                    node_id=node_id,
                    score=float(item["score"]),
                    labels=item["labels"],
                    properties=item["node"],
                    metadata=metadata,
                )
            )

        expanded.sort(key=lambda result: result.score, reverse=True)
        return expanded

    def keyword_search_nodes(
        self,
        query_text: str,
        top_k: int = 20,
        scan_limit: int = 1500,
        max_terms: int = 8,
    ) -> List[SearchResult]:
        """
        Perform keyword search over graph node payload fields.

        Uses the Neo4j full-text index when available, falling back to the
        O(n) Python scan when the index does not yet exist.

        Args:
            query_text: Query text
            top_k: Number of results to return
            scan_limit: Maximum nodes to scan (Python fallback only)
            max_terms: Maximum query terms to use (Python fallback only)

        Returns:
            Keyword-scored graph results
        """
        # Attempt fast path via full-text index
        try:
            lucene_query = " ".join(
                f'"{term}"' for term in query_text.strip().split()[:max_terms] if term
            ) or query_text.strip()
            ft_query = """
            CALL db.index.fulltext.queryNodes('node_text_search', $query_text)
            YIELD node, score
            RETURN elementId(node) AS node_id, labels(node) AS labels, node AS n, score
            LIMIT $top_k
            """
            rows = self.execute_query(ft_query, {"query_text": lucene_query, "top_k": int(top_k)})
            if rows is not None:
                results: List[SearchResult] = []
                raw_scores = [float(row.get("score", 0.0)) for row in rows]
                max_score = max(raw_scores) if raw_scores else 1.0
                for row, raw_score in zip(rows, raw_scores):
                    properties = dict(row.get("n") or {})
                    normalized = raw_score / max(max_score, 1e-9)
                    results.append(
                        SearchResult(
                            node_id=row["node_id"],
                            score=normalized,
                            labels=row.get("labels", []),
                            properties=properties,
                            metadata={"graph_signal": "fulltext_index"},
                        )
                    )
                return results
        except Exception:
            pass  # Index not yet available; fall through to Python scan

        # Python fallback: O(n) scan
        terms = self._tokenize_keyword_query(query_text, max_terms=max_terms)
        if not terms:
            return []

        query = """
        MATCH (n)
        WHERE n.text IS NOT NULL OR n.content IS NOT NULL OR n.title IS NOT NULL OR n.name IS NOT NULL
        RETURN elementId(n) AS node_id, labels(n) AS labels, n
        LIMIT $scan_limit
        """
        rows = self.execute_query(query, {"scan_limit": int(scan_limit)})

        scored: List[SearchResult] = []
        for row in rows:
            properties = dict(row.get("n") or {})
            haystack = " ".join(
                str(properties.get(key, ""))
                for key in ("text", "content", "title", "name")
            ).lower()
            if not haystack:
                continue

            score = 0.0
            for term in terms:
                if term in haystack:
                    score += 1.0
            if score <= 0:
                continue

            normalized = score / max(1.0, float(len(terms)))
            scored.append(
                SearchResult(
                    node_id=row["node_id"],
                    score=normalized,
                    labels=row.get("labels", []),
                    properties=properties,
                    metadata={
                        "matched_terms": int(score),
                        "query_terms": len(terms),
                        "graph_signal": "global_keyword",
                    },
                )
            )

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]

    def community_summary_search(
        self,
        query_text: str,
        top_k: int = 20,
        version: Optional[str] = None,
        scan_limit: int = 500,
        max_terms: int = 8,
    ) -> List[SearchResult]:
        """
        Search stored community summaries and return community-level candidates.

        Args:
            query_text: Query text
            top_k: Number of results to return
            version: Optional community version (latest versions when None)
            scan_limit: Maximum community nodes to scan
            max_terms: Maximum query terms to use

        Returns:
            Community summary search results
        """
        terms = self._tokenize_keyword_query(query_text, max_terms=max_terms)
        if not terms:
            return []

        query = """
        MATCH (c:Community)
        WHERE $version IS NULL OR c.version = $version
        RETURN elementId(c) AS node_id, labels(c) AS labels, c
        ORDER BY c.created_at DESC
        LIMIT $scan_limit
        """
        rows = self.execute_query(
            query,
            {
                "version": version,
                "scan_limit": int(scan_limit),
            },
        )

        scored: List[SearchResult] = []
        for row in rows:
            properties = dict(row.get("c") or {})
            haystack = self._build_community_lexical_text(properties)
            if not haystack:
                continue

            matched_terms = 0
            term_score = 0.0
            for term in terms:
                occurrences = haystack.count(term)
                if occurrences > 0:
                    matched_terms += 1
                    term_score += min(2.0, 1.0 + (0.35 * float(occurrences - 1)))

            if term_score <= 0:
                continue

            coverage = matched_terms / max(1.0, float(len(terms)))
            normalized = term_score / max(1.0, float(len(terms)))
            size_boost = min(0.2, float(properties.get("size", 0)) / 250.0)
            final_score = min(1.0, (0.55 * coverage) + (0.35 * normalized) + size_boost)

            title = str(properties.get("title", "") or "").strip()
            brief_summary = str(properties.get("brief_summary", "") or "").strip()
            summary_text = ". ".join(part for part in [title, brief_summary] if part).strip()

            enriched_properties = dict(properties)
            enriched_properties["text"] = summary_text or str(properties.get("community_id", ""))
            enriched_properties["community_score"] = float(final_score)

            scored.append(
                SearchResult(
                    node_id=str(row.get("node_id")),
                    score=float(final_score),
                    labels=row.get("labels", ["Community"]),
                    properties=enriched_properties,
                    metadata={
                        "matched_terms": matched_terms,
                        "query_terms": len(terms),
                        "graph_signal": "community_summary",
                        "community_id": properties.get("community_id"),
                        "community_version": properties.get("version"),
                    },
                )
            )

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def community_member_search(
        self,
        community_scores: Dict[str, float],
        top_k: int = 20,
        members_per_community: int = 6,
    ) -> List[SearchResult]:
        """
        Expand community candidates to member evidence nodes.

        Args:
            community_scores: Mapping of community_id to community relevance score
            top_k: Number of evidence nodes to return
            members_per_community: Maximum member nodes per community

        Returns:
            Community member evidence search results
        """
        if not community_scores:
            return []

        query = """
        UNWIND $community_ids AS community_id
        MATCH (c:Community {community_id: community_id})
        CALL {
            WITH c
            MATCH (e:Entity)-[:BELONGS_TO]->(c)
            OPTIONAL MATCH (chunk:Chunk)-[:MENTIONS]->(e)
            WITH e, chunk
            RETURN e, chunk
            LIMIT $members_per_community
        }
        RETURN
            c.community_id AS community_id,
            c.version AS community_version,
            elementId(coalesce(chunk, e)) AS node_id,
            labels(coalesce(chunk, e)) AS labels,
            coalesce(chunk, e) AS node,
            CASE WHEN chunk IS NOT NULL THEN 'chunk' ELSE 'entity' END AS evidence_type,
            coalesce(e.name, c.title, c.community_id) AS evidence_name
        """
        rows = self.execute_query(
            query,
            {
                "community_ids": list(community_scores.keys()),
                "members_per_community": int(max(1, members_per_community)),
            },
        )

        if not rows:
            return []

        expanded: List[SearchResult] = []
        for row in rows:
            node_id = row.get("node_id")
            if not node_id:
                continue

            community_id = str(row.get("community_id", ""))
            base_score = float(community_scores.get(community_id, 0.5))
            evidence_type = str(row.get("evidence_type", "entity"))
            type_boost = 1.0 if evidence_type == "chunk" else 0.85
            final_score = min(1.0, base_score * type_boost)

            properties = dict(row.get("node") or {})
            if not str(properties.get("text", "")).strip():
                evidence_name = str(row.get("evidence_name", "")).strip()
                properties["text"] = (
                    f"Community evidence from {community_id}: {evidence_name}"
                    if evidence_name
                    else f"Community evidence from {community_id}"
                )

            expanded.append(
                SearchResult(
                    node_id=str(node_id),
                    score=float(final_score),
                    labels=row.get("labels", []),
                    properties=properties,
                    metadata={
                        "community_id": community_id,
                        "community_version": row.get("community_version"),
                        "graph_signal": "community_membership",
                        "evidence_type": evidence_type,
                    },
                )
            )

        expanded.sort(key=lambda item: item.score, reverse=True)
        return expanded[:top_k]

    def _build_community_lexical_text(
        self,
        properties: Dict[str, Any],
    ) -> str:
        """Build a normalized lexical text from community properties."""
        theme_blob = properties.get("themes", [])
        topic_blob = properties.get("topics", [])
        if isinstance(theme_blob, str):
            try:
                parsed = json.loads(theme_blob)
                theme_blob = parsed if isinstance(parsed, list) else [theme_blob]
            except Exception:
                theme_blob = [theme_blob]
        if isinstance(topic_blob, str):
            try:
                parsed = json.loads(topic_blob)
                topic_blob = parsed if isinstance(parsed, list) else [topic_blob]
            except Exception:
                topic_blob = [topic_blob]

        parts = [
            str(properties.get("title", "") or ""),
            str(properties.get("brief_summary", "") or ""),
            str(properties.get("detailed_summary", "") or ""),
            " ".join(str(item) for item in theme_blob if item),
            " ".join(str(item) for item in topic_blob if item),
        ]
        return " ".join(part for part in parts if part).lower()

    def _tokenize_keyword_query(
        self,
        query_text: str,
        max_terms: int = 8,
    ) -> List[str]:
        """Tokenize query text for keyword search."""
        stopwords = {
            "the", "a", "an", "and", "or", "but", "if", "then", "else", "in",
            "on", "at", "to", "for", "of", "with", "by", "from", "as", "is",
            "are", "was", "were", "be", "been", "being", "do", "does", "did",
            "what", "who", "where", "when", "why", "how", "which", "that",
            "this", "these", "those", "it", "its", "their", "there", "can",
            "could", "would", "should", "may", "might", "will", "about",
        }
        tokens = re.findall(r"\b[a-zA-Z0-9]{2,}\b", query_text.lower())
        deduped: List[str] = []
        seen = set()
        for token in tokens:
            if token in stopwords or token in seen:
                continue
            seen.add(token)
            deduped.append(token)
            if len(deduped) >= max_terms:
                break
        return deduped

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

    def initialize_fulltext_index(self) -> None:
        """Create the full-text search index used by keyword_search_nodes()."""
        try:
            self.execute_query(
                """
                CREATE FULLTEXT INDEX node_text_search IF NOT EXISTS
                FOR (n:DocumentChunk|Entity|Summary|Community)
                ON EACH [n.text, n.content, n.title, n.name]
                """
            )
            logger.info("Full-text index 'node_text_search' is ready")
        except Exception as e:
            logger.warning("Could not create full-text index 'node_text_search': {}", e)

    def create_indexes(self) -> None:
        """Create indexes for common query patterns."""
        self.initialize_fulltext_index()
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
