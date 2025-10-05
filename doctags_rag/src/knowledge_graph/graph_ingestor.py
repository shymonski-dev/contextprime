"""Graph ingestion utilities for bridging DocTags output into Neo4j.

This module provides a lightweight ingestion manager that focuses on
upserting document, section, and chunk nodes required for the GraphRAG
workflow. It intentionally avoids the heavier knowledge-graph pipeline so
that document processing results can be persisted quickly before running
agentic reasoning or downstream analytics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Dict, Optional, Sequence

from loguru import logger

from .neo4j_manager import Neo4jManager


@dataclass
class GraphIngestionStats:
    """Summary of what was upserted for a document."""

    doc_id: str
    chunks_processed: int
    sections_linked: int
    subsections_linked: int


class GraphIngestionManager:
    """Handle Neo4j upserts for processed documents and chunks."""

    def __init__(
        self,
        neo4j_manager: Optional[Neo4jManager] = None,
        *,
        store_sections: bool = True,
        store_subsections: bool = True,
    ) -> None:
        self._neo4j_manager = neo4j_manager
        self.store_sections = store_sections
        self.store_subsections = store_subsections

    # Lazy accessor so tests can inject a fake manager without creating a
    # real Neo4j connection during import time.
    @property
    def neo4j(self) -> Neo4jManager:
        if self._neo4j_manager is None:
            self._neo4j_manager = Neo4jManager()
        return self._neo4j_manager

    def close(self) -> None:
        """Close the underlying Neo4j driver if we created it locally."""
        if self._neo4j_manager is not None:
            self._neo4j_manager.close()

    def ingest_document(
        self,
        doc_properties: Dict[str, object],
        chunks: Sequence[Dict[str, object]],
    ) -> GraphIngestionStats:
        """Upsert a document and its chunks into Neo4j.

        Args:
            doc_properties: Flat dictionary describing the document node.
            chunks: Iterable of dictionaries describing chunk nodes. Each
                chunk dictionary must contain ``chunk_id`` and ``doc_id``.

        Returns:
            GraphIngestionStats with basic counters for logging.
        """

        if not doc_properties:
            raise ValueError("doc_properties must not be empty")

        if not chunks:
            logger.warning(
                "Skipping Neo4j ingestion for %s because no chunks were provided",
                doc_properties.get("doc_id"),
            )
            return GraphIngestionStats(
                doc_id=str(doc_properties.get("doc_id", "unknown")),
                chunks_processed=0,
                sections_linked=0,
                subsections_linked=0,
            )

        doc_id = str(doc_properties.get("doc_id"))
        if not doc_id:
            raise ValueError("doc_properties must include a doc_id")

        timestamp = datetime.now(UTC).isoformat()
        doc_payload = self._serialise_properties(doc_properties)
        doc_payload["doc_id"] = doc_id

        logger.debug("Upserting Document node for %s", doc_id)
        self.neo4j.execute_write_query(
            """
            MERGE (d:Document {doc_id: $doc_id})
            SET d += $props,
                d.updated_at = $updated_at,
                d.ingested_via = 'document_ingestion_pipeline'
            """,
            {
                "doc_id": doc_id,
                "props": doc_payload,
                "updated_at": timestamp,
            },
        )

        sections_seen = set()
        subsections_seen = set()

        for chunk in chunks:
            chunk_id = str(chunk.get("chunk_id"))
            if not chunk_id:
                raise ValueError("Each chunk must include a chunk_id")

            chunk_props = self._serialise_properties(chunk)
            chunk_props.setdefault("doc_id", doc_id)
            chunk_props.setdefault("chunk_id", chunk_id)

            logger.debug("Upserting Chunk node %s", chunk_id)
            self.neo4j.execute_write_query(
                """
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c += $props,
                    c.updated_at = $updated_at
                WITH c
                MATCH (d:Document {doc_id: $doc_id})
                MERGE (d)-[r:HAS_CHUNK]->(c)
                SET r.position = $chunk_index
                """,
                {
                    "chunk_id": chunk_id,
                    "props": chunk_props,
                    "updated_at": timestamp,
                    "doc_id": doc_id,
                    "chunk_index": chunk_props.get("chunk_index", 0),
                },
            )

            context = chunk.get("context") or {}
            section = context.get("section") if isinstance(context, dict) else None
            subsection = context.get("subsection") if isinstance(context, dict) else None

            if self.store_sections and section:
                sections_seen.add(section)
                self._link_section(doc_id, section, chunk_id, timestamp)

            if self.store_subsections and subsection:
                key = (section or "__root__", subsection)
                subsections_seen.add(key)
                self._link_subsection(doc_id, section, subsection, chunk_id, timestamp)

        return GraphIngestionStats(
            doc_id=doc_id,
            chunks_processed=len(chunks),
            sections_linked=len(sections_seen),
            subsections_linked=len(subsections_seen),
        )

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _serialise_properties(self, props: Dict[str, object]) -> Dict[str, object]:
        """Prepare properties for safe storage in Neo4j."""

        serialised: Dict[str, object] = {}
        for key, value in props.items():
            if value is None:
                continue

            if isinstance(value, (str, int, float, bool)):
                serialised[key] = value
            elif isinstance(value, (list, tuple)):
                serialised[key] = [self._normalise_scalar(v) for v in value]
            elif isinstance(value, dict):
                serialised[f"{key}_json"] = json.dumps(value, ensure_ascii=False)
            else:
                serialised[key] = str(value)

        return serialised

    def _normalise_scalar(self, value: object) -> object:
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _link_section(
        self,
        doc_id: str,
        section_name: str,
        chunk_id: str,
        timestamp: str,
    ) -> None:
        self.neo4j.execute_write_query(
            """
            MERGE (s:Section {doc_id: $doc_id, name: $section})
            SET s.title = $section,
                s.doc_id = $doc_id,
                s.updated_at = $updated_at
            WITH s
            MATCH (d:Document {doc_id: $doc_id})
            MERGE (d)-[:HAS_SECTION]->(s)
            WITH s
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (s)-[:HAS_CHUNK]->(c)
            """,
            {
                "doc_id": doc_id,
                "section": section_name,
                "chunk_id": chunk_id,
                "updated_at": timestamp,
            },
        )

    def _link_subsection(
        self,
        doc_id: str,
        section_name: Optional[str],
        subsection_name: str,
        chunk_id: str,
        timestamp: str,
    ) -> None:
        self.neo4j.execute_write_query(
            """
            MERGE (sub:Subsection {doc_id: $doc_id, name: $subsection})
            SET sub.title = $subsection,
                sub.doc_id = $doc_id,
                sub.updated_at = $updated_at
            WITH sub
            MATCH (d:Document {doc_id: $doc_id})
            MERGE (d)-[:HAS_SUBSECTION]->(sub)
            WITH sub
            MATCH (c:Chunk {chunk_id: $chunk_id})
            MERGE (sub)-[:HAS_CHUNK]->(c)
            """,
            {
                "doc_id": doc_id,
                "subsection": subsection_name,
                "chunk_id": chunk_id,
                "updated_at": timestamp,
            },
        )

        if section_name:
            self.neo4j.execute_write_query(
                """
                MERGE (s:Section {doc_id: $doc_id, name: $section})
                MERGE (sub:Subsection {doc_id: $doc_id, name: $subsection})
                MERGE (s)-[:HAS_SUBSECTION]->(sub)
                """,
                {
                    "doc_id": doc_id,
                    "section": section_name,
                    "subsection": subsection_name,
                },
            )
