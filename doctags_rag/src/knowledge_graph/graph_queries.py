"""
Graph Query Interface for Knowledge Graph.

Provides high-level query interface for common graph patterns:
- Entity queries
- Relationship queries
- Path finding
- Document queries
- Analytics queries
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from .neo4j_manager import Neo4jManager


class QueryType(str, Enum):
    """Types of graph queries."""
    ENTITY_SEARCH = "entity_search"
    RELATIONSHIP_QUERY = "relationship_query"
    PATH_FINDING = "path_finding"
    DOCUMENT_QUERY = "document_query"
    ANALYTICS = "analytics"


@dataclass
class QueryResult:
    """Result of a graph query."""
    query_type: QueryType
    results: List[Dict[str, Any]]
    count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphQueryInterface:
    """
    High-level interface for querying the knowledge graph.

    Provides commonly used query patterns:
    - Entity search and retrieval
    - Relationship queries
    - Path finding between entities
    - Document-entity queries
    - Graph analytics
    """

    def __init__(self, neo4j_manager: Optional[Neo4jManager] = None):
        """
        Initialize query interface.

        Args:
            neo4j_manager: Neo4j manager instance
        """
        self.neo4j_manager = neo4j_manager or Neo4jManager()

    # ========== Entity Queries ==========

    def find_entity(
        self,
        name: str,
        entity_type: Optional[str] = None,
        fuzzy: bool = False
    ) -> QueryResult:
        """
        Find entity by name.

        Args:
            name: Entity name to search for
            entity_type: Optional entity type filter
            fuzzy: Whether to use fuzzy matching

        Returns:
            QueryResult with matching entities
        """
        if fuzzy:
            query = """
            MATCH (e:Entity)
            WHERE e.canonical_name CONTAINS toLower($name)
            """
        else:
            query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) = toLower($name)
            """

        if entity_type:
            query += " AND e.type = $entity_type"

        query += """
        RETURN e.name as name,
               e.type as type,
               e.confidence as confidence,
               e.occurrence_count as occurrences,
               elementId(e) as entity_id
        LIMIT 50
        """

        params = {"name": name}
        if entity_type:
            params["entity_type"] = entity_type

        results = self.neo4j_manager.execute_query(query, params)

        return QueryResult(
            query_type=QueryType.ENTITY_SEARCH,
            results=results,
            count=len(results)
        )

    def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get entity by Neo4j ID."""
        query = """
        MATCH (e:Entity)
        WHERE elementId(e) = $entity_id
        RETURN e as entity, labels(e) as labels
        """

        result = self.neo4j_manager.execute_query(query, {"entity_id": entity_id})

        if result:
            return dict(result[0]["entity"])
        return None

    def search_entities(
        self,
        entity_type: Optional[str] = None,
        min_confidence: float = 0.0,
        min_occurrences: int = 1,
        limit: int = 100
    ) -> QueryResult:
        """
        Search entities with filters.

        Args:
            entity_type: Optional entity type filter
            min_confidence: Minimum confidence score
            min_occurrences: Minimum occurrence count
            limit: Maximum results

        Returns:
            QueryResult with matching entities
        """
        query = """
        MATCH (e:Entity)
        WHERE e.confidence >= $min_confidence
        """

        params = {
            "min_confidence": min_confidence,
            "limit": limit
        }

        if entity_type:
            query += " AND e.type = $entity_type"
            params["entity_type"] = entity_type

        if min_occurrences > 1:
            query += " AND e.occurrence_count >= $min_occurrences"
            params["min_occurrences"] = min_occurrences

        query += """
        RETURN e.name as name,
               e.type as type,
               e.confidence as confidence,
               e.occurrence_count as occurrences,
               elementId(e) as entity_id
        ORDER BY e.occurrence_count DESC
        LIMIT $limit
        """

        results = self.neo4j_manager.execute_query(query, params)

        return QueryResult(
            query_type=QueryType.ENTITY_SEARCH,
            results=results,
            count=len(results)
        )

    def get_entity_neighbors(
        self,
        entity_id: str,
        relationship_types: Optional[List[str]] = None,
        direction: str = "both",
        limit: int = 50
    ) -> QueryResult:
        """
        Get neighboring entities.

        Args:
            entity_id: Entity Neo4j ID
            relationship_types: Optional relationship type filter
            direction: Direction (outgoing, incoming, both)
            limit: Maximum results

        Returns:
            QueryResult with neighbor entities
        """
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_filter = f":{rel_types}"

        if direction == "outgoing":
            pattern = f"-[r{rel_filter}]->"
        elif direction == "incoming":
            pattern = f"<-[r{rel_filter}]-"
        else:
            pattern = f"-[r{rel_filter}]-"

        query = f"""
        MATCH (e:Entity){pattern}(neighbor:Entity)
        WHERE elementId(e) = $entity_id
        RETURN neighbor.name as name,
               neighbor.type as type,
               type(r) as relationship_type,
               r.confidence as confidence,
               elementId(neighbor) as entity_id
        LIMIT $limit
        """

        results = self.neo4j_manager.execute_query(
            query,
            {"entity_id": entity_id, "limit": limit}
        )

        return QueryResult(
            query_type=QueryType.ENTITY_SEARCH,
            results=results,
            count=len(results)
        )

    # ========== Relationship Queries ==========

    def find_relationships(
        self,
        source_entity: str,
        target_entity: Optional[str] = None,
        relationship_type: Optional[str] = None
    ) -> QueryResult:
        """
        Find relationships between entities.

        Args:
            source_entity: Source entity name
            target_entity: Optional target entity name
            relationship_type: Optional relationship type

        Returns:
            QueryResult with relationships
        """
        query = """
        MATCH (source:Entity)-[r]->(target:Entity)
        WHERE toLower(source.name) = toLower($source)
        """

        params = {"source": source_entity}

        if target_entity:
            query += " AND toLower(target.name) = toLower($target)"
            params["target"] = target_entity

        if relationship_type:
            query += " AND type(r) = $rel_type"
            params["rel_type"] = relationship_type

        query += """
        RETURN source.name as source,
               target.name as target,
               type(r) as relationship,
               r.confidence as confidence,
               r.evidence as evidence
        LIMIT 100
        """

        results = self.neo4j_manager.execute_query(query, params)

        return QueryResult(
            query_type=QueryType.RELATIONSHIP_QUERY,
            results=results,
            count=len(results)
        )

    def get_relationship_patterns(
        self,
        pattern: str,
        limit: int = 50
    ) -> QueryResult:
        """
        Find relationship patterns (e.g., "Person-WORKS_FOR-Organization").

        Args:
            pattern: Pattern string like "PERSON-WORKS_FOR-ORGANIZATION"
            limit: Maximum results

        Returns:
            QueryResult with matching patterns
        """
        # Parse pattern
        parts = pattern.split("-")
        if len(parts) != 3:
            raise ValueError("Pattern must be in format: SOURCE_TYPE-REL_TYPE-TARGET_TYPE")

        source_type, rel_type, target_type = parts

        query = """
        MATCH (source:Entity {type: $source_type})-[r]->(target:Entity {type: $target_type})
        WHERE type(r) = $rel_type
        RETURN source.name as source,
               target.name as target,
               type(r) as relationship,
               r.confidence as confidence
        LIMIT $limit
        """

        results = self.neo4j_manager.execute_query(
            query,
            {
                "source_type": source_type,
                "target_type": target_type,
                "rel_type": rel_type,
                "limit": limit
            }
        )

        return QueryResult(
            query_type=QueryType.RELATIONSHIP_QUERY,
            results=results,
            count=len(results)
        )

    # ========== Path Finding ==========

    def find_shortest_path(
        self,
        source_entity: str,
        target_entity: str,
        max_depth: int = 5
    ) -> QueryResult:
        """
        Find shortest path between two entities.

        Args:
            source_entity: Source entity name
            target_entity: Target entity name
            max_depth: Maximum path length

        Returns:
            QueryResult with path information
        """
        query = f"""
        MATCH (source:Entity {{canonical_name: toLower($source)}}),
              (target:Entity {{canonical_name: toLower($target)}}),
              path = shortestPath((source)-[*1..{max_depth}]-(target))
        WITH path, [node in nodes(path) | node.name] as entity_names,
             [rel in relationships(path) | type(rel)] as relationship_types
        RETURN entity_names,
               relationship_types,
               length(path) as path_length
        LIMIT 1
        """

        results = self.neo4j_manager.execute_query(
            query,
            {"source": source_entity, "target": target_entity}
        )

        return QueryResult(
            query_type=QueryType.PATH_FINDING,
            results=results,
            count=len(results)
        )

    def find_all_paths(
        self,
        source_entity: str,
        target_entity: str,
        max_depth: int = 3,
        limit: int = 10
    ) -> QueryResult:
        """
        Find all paths between two entities.

        Args:
            source_entity: Source entity name
            target_entity: Target entity name
            max_depth: Maximum path length
            limit: Maximum number of paths

        Returns:
            QueryResult with paths
        """
        query = f"""
        MATCH (source:Entity {{canonical_name: toLower($source)}}),
              (target:Entity {{canonical_name: toLower($target)}}),
              path = (source)-[*1..{max_depth}]-(target)
        WITH path, [node in nodes(path) | node.name] as entity_names,
             [rel in relationships(path) | type(rel)] as relationship_types
        RETURN entity_names,
               relationship_types,
               length(path) as path_length
        ORDER BY path_length ASC
        LIMIT $limit
        """

        results = self.neo4j_manager.execute_query(
            query,
            {"source": source_entity, "target": target_entity, "limit": limit}
        )

        return QueryResult(
            query_type=QueryType.PATH_FINDING,
            results=results,
            count=len(results)
        )

    def find_common_neighbors(
        self,
        entity1: str,
        entity2: str
    ) -> QueryResult:
        """
        Find common neighbors between two entities.

        Args:
            entity1: First entity name
            entity2: Second entity name

        Returns:
            QueryResult with common neighbors
        """
        query = """
        MATCH (e1:Entity {canonical_name: toLower($entity1)})-[]-(common:Entity)-[]-(e2:Entity {canonical_name: toLower($entity2)})
        RETURN common.name as name,
               common.type as type,
               elementId(common) as entity_id
        """

        results = self.neo4j_manager.execute_query(
            query,
            {"entity1": entity1, "entity2": entity2}
        )

        return QueryResult(
            query_type=QueryType.PATH_FINDING,
            results=results,
            count=len(results)
        )

    # ========== Document Queries ==========

    def get_document_entities(
        self,
        doc_id: str,
        entity_type: Optional[str] = None
    ) -> QueryResult:
        """
        Get all entities in a document.

        Args:
            doc_id: Document identifier
            entity_type: Optional entity type filter

        Returns:
            QueryResult with document entities
        """
        query = """
        MATCH (d:Document {doc_id: $doc_id})-[:CONTAINS]->(e:Entity)
        """

        if entity_type:
            query += " WHERE e.type = $entity_type"

        query += """
        RETURN e.name as name,
               e.type as type,
               e.confidence as confidence,
               elementId(e) as entity_id
        ORDER BY e.confidence DESC
        """

        params = {"doc_id": doc_id}
        if entity_type:
            params["entity_type"] = entity_type

        results = self.neo4j_manager.execute_query(query, params)

        return QueryResult(
            query_type=QueryType.DOCUMENT_QUERY,
            results=results,
            count=len(results)
        )

    def find_documents_with_entity(
        self,
        entity_name: str,
        limit: int = 50
    ) -> QueryResult:
        """
        Find documents containing an entity.

        Args:
            entity_name: Entity name
            limit: Maximum results

        Returns:
            QueryResult with documents
        """
        query = """
        MATCH (d:Document)-[:CONTAINS]->(e:Entity)
        WHERE toLower(e.name) = toLower($entity_name)
        RETURN d.doc_id as doc_id,
               d.title as title,
               d.source as source,
               elementId(d) as document_id
        LIMIT $limit
        """

        results = self.neo4j_manager.execute_query(
            query,
            {"entity_name": entity_name, "limit": limit}
        )

        return QueryResult(
            query_type=QueryType.DOCUMENT_QUERY,
            results=results,
            count=len(results)
        )

    def find_similar_documents(
        self,
        doc_id: str,
        min_shared_entities: int = 3,
        limit: int = 10
    ) -> QueryResult:
        """
        Find documents similar based on shared entities.

        Args:
            doc_id: Document identifier
            min_shared_entities: Minimum shared entities
            limit: Maximum results

        Returns:
            QueryResult with similar documents
        """
        query = """
        MATCH (d1:Document {doc_id: $doc_id})-[:CONTAINS]->(e:Entity)<-[:CONTAINS]-(d2:Document)
        WHERE d1 <> d2
        WITH d2, count(e) as shared_entities
        WHERE shared_entities >= $min_shared
        RETURN d2.doc_id as doc_id,
               d2.title as title,
               shared_entities
        ORDER BY shared_entities DESC
        LIMIT $limit
        """

        results = self.neo4j_manager.execute_query(
            query,
            {
                "doc_id": doc_id,
                "min_shared": min_shared_entities,
                "limit": limit
            }
        )

        return QueryResult(
            query_type=QueryType.DOCUMENT_QUERY,
            results=results,
            count=len(results)
        )

    # ========== Analytics Queries ==========

    def get_most_connected_entities(
        self,
        entity_type: Optional[str] = None,
        limit: int = 20
    ) -> QueryResult:
        """
        Get most connected entities by degree.

        Args:
            entity_type: Optional entity type filter
            limit: Maximum results

        Returns:
            QueryResult with top entities
        """
        query = """
        MATCH (e:Entity)
        """

        if entity_type:
            query += " WHERE e.type = $entity_type"

        query += """
        WITH e
        OPTIONAL MATCH (e)-[r]-()
        WITH e, count(r) as degree
        RETURN e.name as name,
               e.type as type,
               degree,
               elementId(e) as entity_id
        ORDER BY degree DESC
        LIMIT $limit
        """

        params = {"limit": limit}
        if entity_type:
            params["entity_type"] = entity_type

        results = self.neo4j_manager.execute_query(query, params)

        return QueryResult(
            query_type=QueryType.ANALYTICS,
            results=results,
            count=len(results),
            metadata={"metric": "degree_centrality"}
        )

    def calculate_pagerank(
        self,
        limit: int = 20
    ) -> QueryResult:
        """
        Calculate PageRank for entities.

        Args:
            limit: Maximum results

        Returns:
            QueryResult with PageRank scores
        """
        # Note: This requires Graph Data Science library
        # For now, use degree as approximation
        return self.get_most_connected_entities(limit=limit)

    def detect_communities(
        self,
        algorithm: str = "louvain"
    ) -> QueryResult:
        """
        Detect communities in the graph.

        Args:
            algorithm: Community detection algorithm

        Returns:
            QueryResult with communities
        """
        # This is a simplified version
        # Full implementation would use Neo4j GDS library
        query = """
        MATCH (e:Entity)-[r]-(other:Entity)
        WITH e, collect(DISTINCT other) as neighbors
        RETURN e.name as entity,
               e.type as type,
               size(neighbors) as neighbor_count
        ORDER BY neighbor_count DESC
        LIMIT 100
        """

        results = self.neo4j_manager.execute_query(query)

        return QueryResult(
            query_type=QueryType.ANALYTICS,
            results=results,
            count=len(results),
            metadata={"algorithm": algorithm}
        )

    def get_entity_statistics(self) -> Dict[str, Any]:
        """Get entity statistics."""
        query = """
        MATCH (e:Entity)
        WITH e.type as type, count(*) as count
        RETURN type, count
        ORDER BY count DESC
        """

        results = self.neo4j_manager.execute_query(query)

        stats = {
            "by_type": {r["type"]: r["count"] for r in results},
            "total": sum(r["count"] for r in results)
        }

        return stats

    def get_relationship_statistics(self) -> Dict[str, Any]:
        """Get relationship statistics."""
        query = """
        MATCH ()-[r]->()
        WITH type(r) as rel_type, count(*) as count
        RETURN rel_type, count
        ORDER BY count DESC
        """

        results = self.neo4j_manager.execute_query(query)

        stats = {
            "by_type": {r["rel_type"]: r["count"] for r in results},
            "total": sum(r["count"] for r in results)
        }

        return stats

    # ========== Custom Queries ==========

    def execute_custom_query(
        self,
        cypher_query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> QueryResult:
        """
        Execute a custom Cypher query.

        Args:
            cypher_query: Cypher query string
            parameters: Query parameters

        Returns:
            QueryResult
        """
        results = self.neo4j_manager.execute_query(cypher_query, parameters)

        return QueryResult(
            query_type=QueryType.ANALYTICS,
            results=results,
            count=len(results),
            metadata={"custom_query": True}
        )
