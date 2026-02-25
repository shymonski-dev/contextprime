"""
Graph Builder for Knowledge Graph Construction.

Builds a comprehensive knowledge graph in Neo4j from extracted entities and relationships.

Creates:
- Document nodes with metadata
- Entity nodes (deduplicated)
- Relationship edges with properties
- Hierarchical structure (Document → Section → Chunk → Entity)
- Cross-document connections
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

from loguru import logger

from .entity_extractor import Entity, EntityExtractionResult
from .relationship_extractor import Relationship, RelationshipExtractionResult
from .entity_resolver import EntityCluster, ResolutionResult
from .neo4j_manager import Neo4jManager, GraphNode, GraphRelationship


@dataclass
class DocumentMetadata:
    """Metadata for a document."""
    doc_id: str
    title: Optional[str] = None
    source: Optional[str] = None
    created_at: Optional[datetime] = None
    processed_at: Optional[datetime] = None
    content_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    section_id: Optional[str] = None
    embedding: Optional[List[float]] = None


@dataclass
class GraphBuildResult:
    """Result of graph building."""
    nodes_created: int
    relationships_created: int
    entities_linked: int
    documents_processed: int
    statistics: Dict[str, Any] = field(default_factory=dict)


class GraphBuilder:
    """
    Main graph construction system for building knowledge graphs in Neo4j.

    Features:
    - Document node creation with rich metadata
    - Entity node creation with deduplication
    - Relationship creation with properties
    - Hierarchical document structure
    - Cross-document entity linking
    - Batch operations for performance
    - Transaction management
    - Index creation for fast queries
    """

    def __init__(
        self,
        neo4j_manager: Optional[Neo4jManager] = None,
        batch_size: int = 1000,
        create_indexes: bool = True
    ):
        """
        Initialize graph builder.

        Args:
            neo4j_manager: Neo4j manager instance
            batch_size: Batch size for bulk operations
            create_indexes: Whether to create indexes
        """
        self.neo4j_manager = neo4j_manager or Neo4jManager()
        self.batch_size = batch_size

        # Track created nodes
        self.entity_node_map: Dict[str, str] = {}  # entity_key -> neo4j_id
        self.doc_node_map: Dict[str, str] = {}  # doc_id -> neo4j_id

        # Initialize schema
        if create_indexes:
            self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Initialize Neo4j schema with constraints and indexes."""
        logger.info("Initializing graph schema...")

        # Create constraints
        self.neo4j_manager.create_schema_constraints()

        # Create indexes
        self.neo4j_manager.create_indexes()

        # Create additional KG-specific indexes
        additional_indexes = [
            """
            CREATE INDEX entity_name IF NOT EXISTS
            FOR (e:Entity)
            ON (e.name)
            """,
            """
            CREATE INDEX entity_canonical IF NOT EXISTS
            FOR (e:Entity)
            ON (e.canonical_name)
            """,
            """
            CREATE INDEX chunk_doc_id IF NOT EXISTS
            FOR (c:Chunk)
            ON (c.doc_id)
            """,
        ]

        for index_query in additional_indexes:
            try:
                self.neo4j_manager.execute_query(index_query)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    logger.warning(f"Index creation warning: {e}")

        logger.info("Schema initialization complete")

    def build_document_graph(
        self,
        doc_metadata: DocumentMetadata,
        chunks: List[ChunkMetadata],
        entity_result: EntityExtractionResult,
        relationship_result: RelationshipExtractionResult,
        resolution_result: Optional[ResolutionResult] = None
    ) -> GraphBuildResult:
        """
        Build graph for a single document.

        Args:
            doc_metadata: Document metadata
            chunks: List of text chunks
            entity_result: Entity extraction result
            relationship_result: Relationship extraction result
            resolution_result: Optional entity resolution result

        Returns:
            GraphBuildResult with statistics
        """
        nodes_created = 0
        relationships_created = 0

        # 1. Create document node
        doc_node_id = self._create_document_node(doc_metadata)
        nodes_created += 1

        # 2. Create chunk nodes
        chunk_node_ids = self._create_chunk_nodes(chunks, doc_node_id)
        nodes_created += len(chunk_node_ids)

        # 3. Create entity nodes (using resolved entities if available)
        if resolution_result:
            entity_node_ids = self._create_resolved_entity_nodes(
                resolution_result.clusters,
                doc_metadata.doc_id
            )
        else:
            entity_node_ids = self._create_entity_nodes(
                entity_result.entities,
                doc_metadata.doc_id
            )
        nodes_created += len(entity_node_ids)

        # 4. Link entities to chunks
        entity_chunk_rels = self._link_entities_to_chunks(
            entity_result.entities,
            chunks,
            entity_node_ids,
            chunk_node_ids
        )
        relationships_created += entity_chunk_rels

        # 5. Create relationships between entities
        rel_count = self._create_entity_relationships(
            relationship_result.relationships,
            entity_node_ids
        )
        relationships_created += rel_count

        # 6. Link document to entities (CONTAINS relationships)
        doc_entity_rels = self._link_document_to_entities(
            doc_node_id,
            entity_node_ids
        )
        relationships_created += doc_entity_rels

        result = GraphBuildResult(
            nodes_created=nodes_created,
            relationships_created=relationships_created,
            entities_linked=len(entity_node_ids),
            documents_processed=1,
            statistics={
                "document_id": doc_metadata.doc_id,
                "chunks": len(chunks),
                "entities": len(entity_result.entities),
                "unique_entities": len(entity_node_ids),
                "relationships": len(relationship_result.relationships)
            }
        )

        logger.info(
            f"Built graph for document {doc_metadata.doc_id}: "
            f"{nodes_created} nodes, {relationships_created} relationships"
        )

        return result

    def _create_document_node(
        self,
        doc_metadata: DocumentMetadata
    ) -> str:
        """Create a document node in Neo4j."""
        properties = {
            "doc_id": doc_metadata.doc_id,
            "title": doc_metadata.title or "Untitled",
            "source": doc_metadata.source,
            "content_type": doc_metadata.content_type,
            "tags": doc_metadata.tags,
            "created_at": doc_metadata.created_at.isoformat() if doc_metadata.created_at else None,
            "processed_at": doc_metadata.processed_at.isoformat() if doc_metadata.processed_at else datetime.now().isoformat(),
            **doc_metadata.extra
        }

        # Remove None values
        properties = {k: v for k, v in properties.items() if v is not None}

        query = """
        MERGE (d:Document {doc_id: $doc_id})
        SET d += $properties
        RETURN elementId(d) as node_id
        """

        result = self.neo4j_manager.execute_write_query(
            query,
            {"doc_id": doc_metadata.doc_id, "properties": properties}
        )

        node_id = result[0]["node_id"]
        self.doc_node_map[doc_metadata.doc_id] = node_id

        return node_id

    def _create_chunk_nodes(
        self,
        chunks: List[ChunkMetadata],
        doc_node_id: str
    ) -> List[str]:
        """Create chunk nodes and link to document."""
        if not chunks:
            return []

        chunk_data = []
        for chunk in chunks:
            properties = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "text": chunk.text,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "section_id": chunk.section_id
            }

            # Add embedding if available
            if chunk.embedding:
                properties["embedding"] = chunk.embedding

            chunk_data.append(properties)

        # Batch create chunks
        query = """
        UNWIND $chunks as chunk
        CREATE (c:Chunk)
        SET c = chunk
        WITH c
        MATCH (d:Document)
        WHERE elementId(d) = $doc_node_id
        CREATE (d)-[:HAS_CHUNK]->(c)
        RETURN elementId(c) as node_id
        """

        result = self.neo4j_manager.execute_write_query(
            query,
            {"chunks": chunk_data, "doc_node_id": doc_node_id}
        )

        return [r["node_id"] for r in result]

    def _create_entity_nodes(
        self,
        entities: List[Entity],
        doc_id: str
    ) -> Dict[str, str]:
        """
        Create entity nodes in Neo4j.

        Returns:
            Dict mapping entity key to node ID
        """
        if not entities:
            return {}

        entity_node_ids = {}
        entity_data = []

        for entity in entities:
            # Create unique key for entity
            entity_key = self._entity_key(entity)

            properties = {
                "name": entity.text,
                "canonical_name": entity.text.lower(),
                "type": entity.type,
                "confidence": entity.confidence,
                "source": entity.source,
                **entity.attributes
            }

            entity_data.append({
                "key": entity_key,
                "properties": properties
            })

        # Batch create entities with MERGE to handle duplicates
        query = """
        UNWIND $entities as ent
        MERGE (e:Entity {canonical_name: ent.properties.canonical_name, type: ent.properties.type})
        ON CREATE SET e = ent.properties, e.created_at = datetime()
        ON MATCH SET e.confidence = CASE WHEN ent.properties.confidence > e.confidence
                                         THEN ent.properties.confidence
                                         ELSE e.confidence END
        RETURN ent.key as entity_key, elementId(e) as node_id
        """

        result = self.neo4j_manager.execute_write_query(
            query,
            {"entities": entity_data}
        )

        # Check for empty results
        if not result:
            logger.warning(f"No entity nodes created for doc {doc_id} - query returned empty result")
            return entity_node_ids

        # Safe key access to avoid KeyError
        for index, r in enumerate(result):
            entity_key = r.get("entity_key")
            node_id = r.get("node_id")

            # Some drivers/tests may drop the projected key; fall back to ordered entity list
            if not entity_key and index < len(entity_data):
                entity_key = entity_data[index]["key"]

            if entity_key and node_id:
                entity_node_ids[entity_key] = node_id
                self.entity_node_map[entity_key] = node_id
            else:
                logger.warning(f"Incomplete result record in doc {doc_id}: {r}")

        return entity_node_ids

    def _create_resolved_entity_nodes(
        self,
        clusters: List[EntityCluster],
        doc_id: str
    ) -> Dict[str, str]:
        """
        Create entity nodes from resolved clusters.

        Returns:
            Dict mapping entity key to node ID
        """
        if not clusters:
            return {}

        entity_node_ids = {}
        entity_data = []

        for cluster in clusters:
            canonical = cluster.canonical_entity
            entity_key = self._entity_key(canonical)

            # Collect all variant names
            variant_names = [canonical.text]
            variant_names.extend([v.text for v in cluster.variant_entities])

            properties = {
                "name": canonical.text,
                "canonical_name": canonical.text.lower(),
                "type": canonical.type,
                "confidence": cluster.confidence,
                "source": canonical.source,
                "variant_names": list(set(variant_names)),
                "occurrence_count": len(variant_names),
                **canonical.attributes
            }

            entity_data.append({
                "key": entity_key,
                "properties": properties
            })

            # Also map variant keys to same node
            for variant in cluster.variant_entities:
                variant_key = self._entity_key(variant)
                entity_node_ids[variant_key] = None  # Will be filled after query

        # Batch create entities
        query = """
        UNWIND $entities as ent
        MERGE (e:Entity {canonical_name: ent.properties.canonical_name, type: ent.properties.type})
        ON CREATE SET e = ent.properties, e.created_at = datetime()
        ON MATCH SET e.occurrence_count = e.occurrence_count + ent.properties.occurrence_count,
                     e.confidence = CASE WHEN ent.properties.confidence > e.confidence
                                         THEN ent.properties.confidence
                                         ELSE e.confidence END
        RETURN ent.key as entity_key, elementId(e) as node_id
        """

        result = self.neo4j_manager.execute_write_query(
            query,
            {"entities": entity_data}
        )

        for r in result:
            entity_node_ids[r["entity_key"]] = r["node_id"]
            self.entity_node_map[r["entity_key"]] = r["node_id"]

        # Fill in variant mappings
        for cluster in clusters:
            canonical_key = self._entity_key(cluster.canonical_entity)
            node_id = entity_node_ids.get(canonical_key)
            if node_id:
                for variant in cluster.variant_entities:
                    variant_key = self._entity_key(variant)
                    entity_node_ids[variant_key] = node_id

        return entity_node_ids

    def _link_entities_to_chunks(
        self,
        entities: List[Entity],
        chunks: List[ChunkMetadata],
        entity_node_ids: Dict[str, str],
        chunk_node_ids: List[str]
    ) -> int:
        """Link entities to the chunks they appear in."""
        relationships = []

        # Create chunk position lookup
        chunk_positions = []
        for i, chunk in enumerate(chunks):
            chunk_positions.append({
                "start": chunk.start_char,
                "end": chunk.end_char,
                "node_id": chunk_node_ids[i] if i < len(chunk_node_ids) else None
            })

        # Link each entity to its chunk
        for entity in entities:
            entity_key = self._entity_key(entity)
            entity_node_id = entity_node_ids.get(entity_key)

            if not entity_node_id:
                continue

            # Find which chunk contains this entity
            for chunk_info in chunk_positions:
                if (entity.start_char >= chunk_info["start"] and
                    entity.end_char <= chunk_info["end"] and
                    chunk_info["node_id"]):

                    relationships.append({
                        "entity_id": entity_node_id,
                        "chunk_id": chunk_info["node_id"],
                        "start_char": entity.start_char - chunk_info["start"],
                        "end_char": entity.end_char - chunk_info["start"]
                    })
                    break

        if not relationships:
            return 0

        # Batch create relationships
        query = """
        UNWIND $rels as rel
        MATCH (e:Entity), (c:Chunk)
        WHERE elementId(e) = rel.entity_id AND elementId(c) = rel.chunk_id
        MERGE (c)-[r:MENTIONS]->(e)
        SET r.start_char = rel.start_char,
            r.end_char = rel.end_char
        """

        self.neo4j_manager.execute_write_query(query, {"rels": relationships})

        return len(relationships)

    def _create_entity_relationships(
        self,
        relationships: List[Relationship],
        entity_node_ids: Dict[str, str]
    ) -> int:
        """Create relationships between entities."""
        if not relationships:
            return 0

        rel_data = []
        for rel in relationships:
            source_key = self._entity_key(rel.source_entity)
            target_key = self._entity_key(rel.target_entity)

            source_id = entity_node_ids.get(source_key)
            target_id = entity_node_ids.get(target_key)

            if not source_id or not target_id:
                continue

            rel_data.append({
                "source_id": source_id,
                "target_id": target_id,
                "type": rel.relation_type,
                "confidence": rel.confidence,
                "evidence": rel.evidence,
                "context": rel.context,
                "source": rel.source,
                **rel.properties
            })

        if not rel_data:
            return 0

        # Group by type for efficient creation
        from collections import defaultdict
        rels_by_type = defaultdict(list)

        for rel in rel_data:
            rel_type = rel.pop("type")
            rels_by_type[rel_type].append(rel)

        total_created = 0
        for rel_type, rels in rels_by_type.items():
            # Sanitize relationship type (Neo4j naming rules)
            safe_type = rel_type.replace(" ", "_").replace("-", "_").upper()

            query = f"""
            UNWIND $rels as rel
            MATCH (source:Entity), (target:Entity)
            WHERE elementId(source) = rel.source_id AND elementId(target) = rel.target_id
            MERGE (source)-[r:{safe_type}]->(target)
            SET r += rel
            """

            self.neo4j_manager.execute_write_query(query, {"rels": rels})
            total_created += len(rels)

        return total_created

    def _link_document_to_entities(
        self,
        doc_node_id: str,
        entity_node_ids: Dict[str, str]
    ) -> int:
        """Create CONTAINS relationships from document to entities."""
        if not entity_node_ids:
            return 0

        unique_entity_ids = list(set(entity_node_ids.values()))

        query = """
        MATCH (d:Document)
        WHERE elementId(d) = $doc_id
        WITH d
        UNWIND $entity_ids as entity_id
        MATCH (e:Entity)
        WHERE elementId(e) = entity_id
        MERGE (d)-[r:CONTAINS]->(e)
        """

        self.neo4j_manager.execute_write_query(
            query,
            {"doc_id": doc_node_id, "entity_ids": unique_entity_ids}
        )

        return len(unique_entity_ids)

    def create_cross_document_links(
        self,
        doc_ids: List[str],
        link_type: str = "RELATED_TO"
    ) -> int:
        """
        Create relationships between entities across documents.

        Args:
            doc_ids: List of document IDs to link
            link_type: Type of cross-document relationship

        Returns:
            Number of cross-document links created
        """
        # Find shared entities across documents
        query = """
        MATCH (d1:Document)-[:CONTAINS]->(e:Entity)<-[:CONTAINS]-(d2:Document)
        WHERE d1.doc_id IN $doc_ids
          AND d2.doc_id IN $doc_ids
          AND d1.doc_id < d2.doc_id
        WITH d1, d2, count(e) as shared_entities
        WHERE shared_entities > 0
        MERGE (d1)-[r:SHARES_ENTITIES]->(d2)
        SET r.shared_count = shared_entities
        RETURN count(r) as links_created
        """

        result = self.neo4j_manager.execute_write_query(
            query,
            {"doc_ids": doc_ids}
        )

        links_created = result[0]["links_created"] if result else 0

        logger.info(f"Created {links_created} cross-document links")
        return links_created

    def add_entity_embeddings(
        self,
        entity_embeddings: Dict[str, List[float]]
    ) -> int:
        """
        Add embeddings to entity nodes.

        Args:
            entity_embeddings: Dict mapping entity canonical name to embedding

        Returns:
            Number of entities updated
        """
        updates = []
        for canonical_name, embedding in entity_embeddings.items():
            updates.append({
                "canonical_name": canonical_name.lower(),
                "embedding": embedding
            })

        query = """
        UNWIND $updates as update
        MATCH (e:Entity {canonical_name: update.canonical_name})
        SET e.embedding = update.embedding
        RETURN count(e) as updated_count
        """

        result = self.neo4j_manager.execute_write_query(
            query,
            {"updates": updates}
        )

        updated = result[0]["updated_count"] if result else 0
        logger.info(f"Added embeddings to {updated} entities")

        return updated

    def _entity_key(self, entity: Entity) -> str:
        """Generate a unique key for an entity."""
        # Use canonical name + type as key
        key_str = f"{entity.text.lower()}:{entity.type}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return self.neo4j_manager.get_statistics()

    def clear_graph(self, confirm: bool = False) -> bool:
        """Clear the entire graph."""
        if confirm:
            return self.neo4j_manager.clear_database(confirm=True)
        else:
            logger.warning("Graph clear requires confirmation")
            return False
