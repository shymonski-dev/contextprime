"""
Community Storage for persisting community detection results in Neo4j.

Stores:
- Community nodes
- Membership relationships
- Community properties and metadata
- Version history
"""

from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import json

from neo4j import GraphDatabase
from loguru import logger

from ..knowledge_graph.neo4j_manager import Neo4jManager
from .community_detector import CommunityResult
from .community_summarizer import CommunitySummary, GlobalSummary


class CommunityStorage:
    """
    Stores and retrieves community detection results in Neo4j.

    Supports versioning and incremental updates.
    """

    def __init__(self, neo4j_manager: Optional[Neo4jManager] = None):
        """
        Initialize community storage.

        Args:
            neo4j_manager: Neo4j manager instance
        """
        self.neo4j = neo4j_manager or Neo4jManager()
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Initialize Neo4j schema for communities."""
        # Create constraints and indexes
        queries = [
            """
            CREATE CONSTRAINT community_id_unique IF NOT EXISTS
            FOR (c:Community)
            REQUIRE c.community_id IS UNIQUE
            """,
            """
            CREATE INDEX community_version IF NOT EXISTS
            FOR (c:Community)
            ON (c.version)
            """,
            """
            CREATE INDEX community_created IF NOT EXISTS
            FOR (c:Community)
            ON (c.created_at)
            """
        ]

        for query in queries:
            try:
                self.neo4j.execute_query(query)
            except Exception as e:
                logger.warning(f"Schema initialization query failed: {e}")

    def store_communities(
        self,
        community_result: CommunityResult,
        community_summaries: Optional[Dict[int, CommunitySummary]] = None,
        version: Optional[str] = None
    ) -> str:
        """
        Store community detection results in Neo4j.

        Args:
            community_result: Community detection result
            community_summaries: Optional summaries for each community
            version: Version identifier (timestamp if None)

        Returns:
            Version identifier
        """
        if version is None:
            version = datetime.now().isoformat()

        logger.info(f"Storing {community_result.num_communities} communities (version: {version})")

        # Create community nodes
        for comm_id, members in community_result.communities.items():
            summary = community_summaries.get(comm_id) if community_summaries else None

            properties = {
                "community_id": f"{version}_{comm_id}",
                "internal_id": comm_id,
                "version": version,
                "size": len(members),
                "algorithm": community_result.algorithm,
                "created_at": datetime.now().isoformat()
            }

            # Add summary information if available
            if summary:
                properties.update({
                    "title": summary.title,
                    "brief_summary": summary.brief_summary,
                    "detailed_summary": summary.detailed_summary[:1000] if summary.detailed_summary else "",
                    "themes": json.dumps(summary.themes),
                    "topics": json.dumps(summary.topics)
                })

            # Create community node
            self.neo4j.create_node(
                labels=["Community"],
                properties=properties,
                return_node=False
            )

            # Create membership relationships
            self._create_memberships(version, comm_id, members)

        # Store metadata
        self._store_metadata(version, community_result, community_summaries)

        logger.info(f"Communities stored successfully (version: {version})")
        return version

    def _create_memberships(
        self,
        version: str,
        comm_id: int,
        members: Set[str]
    ) -> None:
        """Create BELONGS_TO relationships for community members."""
        query = """
        MATCH (c:Community {community_id: $community_id})
        MATCH (e:Entity {name: $entity_name})
        MERGE (e)-[r:BELONGS_TO]->(c)
        SET r.version = $version
        """

        community_id = f"{version}_{comm_id}"

        for entity_name in members:
            try:
                self.neo4j.execute_write_query(query, {
                    "community_id": community_id,
                    "entity_name": entity_name,
                    "version": version
                })
            except Exception as e:
                logger.warning(f"Failed to create membership for {entity_name}: {e}")

    def _store_metadata(
        self,
        version: str,
        community_result: CommunityResult,
        community_summaries: Optional[Dict[int, CommunitySummary]]
    ) -> None:
        """Store community detection metadata."""
        metadata = {
            "version": version,
            "algorithm": community_result.algorithm,
            "num_communities": community_result.num_communities,
            "modularity": community_result.modularity,
            "execution_time": community_result.execution_time,
            "created_at": datetime.now().isoformat()
        }

        # Create metadata node
        self.neo4j.create_node(
            labels=["CommunityMetadata"],
            properties=metadata,
            return_node=False
        )

    def load_communities(
        self,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load community detection results from Neo4j.

        Args:
            version: Version to load (latest if None)

        Returns:
            Dictionary with community data
        """
        if version is None:
            # Get latest version
            query = """
            MATCH (m:CommunityMetadata)
            RETURN m.version as version
            ORDER BY m.created_at DESC
            LIMIT 1
            """
            result = self.neo4j.execute_query(query)
            if not result:
                logger.warning("No community versions found")
                return None
            version = result[0]["version"]

        logger.info(f"Loading communities (version: {version})")

        # Load community nodes
        query = """
        MATCH (c:Community {version: $version})
        RETURN c
        """
        communities = self.neo4j.execute_query(query, {"version": version})

        if not communities:
            logger.warning(f"No communities found for version {version}")
            return None

        # Load memberships
        memberships = self._load_memberships(version)

        # Load metadata
        metadata = self._load_metadata(version)

        return {
            "version": version,
            "communities": communities,
            "memberships": memberships,
            "metadata": metadata
        }

    def _load_memberships(self, version: str) -> Dict[str, List[str]]:
        """Load community memberships."""
        query = """
        MATCH (e:Entity)-[r:BELONGS_TO {version: $version}]->(c:Community)
        RETURN c.community_id as community_id, e.name as entity_name
        """

        results = self.neo4j.execute_query(query, {"version": version})

        memberships = {}
        for result in results:
            comm_id = result["community_id"]
            entity_name = result["entity_name"]

            if comm_id not in memberships:
                memberships[comm_id] = []
            memberships[comm_id].append(entity_name)

        return memberships

    def _load_metadata(self, version: str) -> Optional[Dict[str, Any]]:
        """Load community detection metadata."""
        query = """
        MATCH (m:CommunityMetadata {version: $version})
        RETURN m
        """

        results = self.neo4j.execute_query(query, {"version": version})

        if results:
            return dict(results[0]["m"])
        return None

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all stored community versions."""
        query = """
        MATCH (m:CommunityMetadata)
        RETURN m.version as version, m.created_at as created_at,
               m.num_communities as num_communities, m.algorithm as algorithm
        ORDER BY m.created_at DESC
        """

        results = self.neo4j.execute_query(query)
        return results

    def delete_version(self, version: str) -> bool:
        """Delete a specific community version."""
        logger.info(f"Deleting community version: {version}")

        try:
            # Delete memberships
            query = """
            MATCH ()-[r:BELONGS_TO {version: $version}]->()
            DELETE r
            """
            self.neo4j.execute_write_query(query, {"version": version})

            # Delete community nodes
            query = """
            MATCH (c:Community {version: $version})
            DELETE c
            """
            self.neo4j.execute_write_query(query, {"version": version})

            # Delete metadata
            query = """
            MATCH (m:CommunityMetadata {version: $version})
            DELETE m
            """
            self.neo4j.execute_write_query(query, {"version": version})

            logger.info(f"Version {version} deleted successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to delete version {version}: {e}")
            return False

    def get_entity_communities(
        self,
        entity_name: str,
        version: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get communities that an entity belongs to."""
        if version:
            query = """
            MATCH (e:Entity {name: $entity_name})-[r:BELONGS_TO {version: $version}]->(c:Community)
            RETURN c, r
            """
            params = {"entity_name": entity_name, "version": version}
        else:
            query = """
            MATCH (e:Entity {name: $entity_name})-[r:BELONGS_TO]->(c:Community)
            RETURN c, r, c.version as version
            ORDER BY c.created_at DESC
            """
            params = {"entity_name": entity_name}

        results = self.neo4j.execute_query(query, params)
        return results

    def get_community_members(
        self,
        community_id: str,
        version: str
    ) -> List[str]:
        """Get members of a specific community."""
        full_id = f"{version}_{community_id}" if not community_id.startswith(version) else community_id

        query = """
        MATCH (e:Entity)-[:BELONGS_TO]->(c:Community {community_id: $community_id})
        RETURN e.name as entity_name
        """

        results = self.neo4j.execute_query(query, {"community_id": full_id})
        return [r["entity_name"] for r in results]

    def store_global_summary(
        self,
        global_summary: GlobalSummary,
        version: str
    ) -> None:
        """Store global summary for a version."""
        properties = {
            "version": version,
            "num_communities": global_summary.num_communities,
            "main_themes": json.dumps(global_summary.main_themes),
            "overall_structure": global_summary.overall_structure,
            "key_insights": json.dumps(global_summary.key_insights),
            "metadata": json.dumps(global_summary.metadata),
            "created_at": datetime.now().isoformat()
        }

        self.neo4j.create_node(
            labels=["GlobalSummary"],
            properties=properties,
            return_node=False
        )

        logger.info(f"Global summary stored for version {version}")

    def load_global_summary(self, version: str) -> Optional[Dict[str, Any]]:
        """Load global summary for a version."""
        query = """
        MATCH (g:GlobalSummary {version: $version})
        RETURN g
        """

        results = self.neo4j.execute_query(query, {"version": version})

        if results:
            return dict(results[0]["g"])
        return None
