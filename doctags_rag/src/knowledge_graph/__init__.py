"""
Knowledge Graph module for DocTags RAG System.

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
from .entity_extractor import EntityExtractor, Entity, EntityExtractionResult
from .relationship_extractor import RelationshipExtractor, Relationship, RelationshipExtractionResult
from .entity_resolver import EntityResolver, EntityCluster, ResolutionResult
from .graph_builder import GraphBuilder, DocumentMetadata, ChunkMetadata, GraphBuildResult
from .kg_pipeline import KnowledgeGraphPipeline, PipelineConfig, PipelineResult
from .graph_queries import GraphQueryInterface, QueryResult

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
    "KnowledgeGraphPipeline",
    "PipelineConfig",
    "PipelineResult",
    "GraphQueryInterface",
    "QueryResult",
]
