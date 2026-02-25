"""
Knowledge Graph module for Contextprime.

Provides comprehensive knowledge graph construction and querying:
- Entity extraction from documents
- Relationship extraction between entities
- Entity resolution and disambiguation
- Graph construction in Neo4j
- End-to-end pipeline orchestration
- Graph query interface
"""

from .neo4j_manager import (
    Neo4jManager,
    GraphNode,
    GraphRelationship,
    SearchResult,
)
from .graph_queries import GraphQueryInterface, QueryResult

# Heavy ML/NLP imports — optional; require sentence-transformers, spacy, etc.
try:
    from .entity_extractor import EntityExtractor, Entity, EntityExtractionResult
    from .relationship_extractor import RelationshipExtractor, Relationship, RelationshipExtractionResult
    from .entity_resolver import EntityResolver, EntityCluster, ResolutionResult
    from .graph_builder import GraphBuilder, DocumentMetadata, ChunkMetadata, GraphBuildResult
    from .graph_ingestor import GraphIngestionManager, GraphIngestionStats
    from .kg_pipeline import KnowledgeGraphPipeline, PipelineConfig, PipelineResult
except Exception:  # pragma: no cover
    from loguru import logger as _logger
    _logger.warning("knowledge_graph: optional ML components unavailable (sentence-transformers/spacy not installed)")

__all__ = [
    "Neo4jManager",
    "GraphNode",
    "GraphRelationship",
    "SearchResult",
    "EntityExtractor",
    "Entity",
    "EntityExtractionResult",
    "RelationshipExtractor",
    "Relationship",
    "RelationshipExtractionResult",
    "EntityResolver",
    "EntityCluster",
    "ResolutionResult",
    "GraphBuilder",
    "DocumentMetadata",
    "ChunkMetadata",
    "GraphBuildResult",
    "GraphIngestionManager",
    "GraphIngestionStats",
    "KnowledgeGraphPipeline",
    "PipelineConfig",
    "PipelineResult",
    "GraphQueryInterface",
    "QueryResult",
]
