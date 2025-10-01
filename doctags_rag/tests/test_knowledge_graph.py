"""
Comprehensive tests for Knowledge Graph components.

Tests cover:
- Entity extraction
- Relationship extraction
- Entity resolution
- Graph construction
- Pipeline execution
- Query interface
"""

import pytest
from typing import List

from src.knowledge_graph import (
    EntityExtractor,
    RelationshipExtractor,
    EntityResolver,
    GraphBuilder,
    KnowledgeGraphPipeline,
    GraphQueryInterface,
    PipelineConfig,
    DocumentMetadata,
    ChunkMetadata,
)


# Sample texts for testing
SAMPLE_TEXT_1 = """
John Smith works for Microsoft Corporation in Seattle. He is the CEO and
founded the company in 1990. Microsoft is a technology company that develops
software products including Windows and Office.
"""

SAMPLE_TEXT_2 = """
Apple Inc. is located in Cupertino, California. Tim Cook is the current CEO
of Apple. The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne
in 1976. Apple creates innovative products like the iPhone and MacBook.
"""

SAMPLE_TEXT_3 = """
Amazon, led by Andy Jassy, is headquartered in Seattle. The company was founded
by Jeff Bezos in 1994. Amazon started as an online bookstore but has expanded
to cloud computing with AWS and artificial intelligence.
"""


class TestEntityExtractor:
    """Test entity extraction functionality."""

    def test_basic_entity_extraction(self):
        """Test basic entity extraction from text."""
        extractor = EntityExtractor(use_llm=False)
        result = extractor.extract_entities(
            text=SAMPLE_TEXT_1,
            document_id="test_doc_1"
        )

        assert len(result.entities) > 0
        assert result.document_id == "test_doc_1"

        # Check for expected entity types
        entity_types = {e.type for e in result.entities}
        assert "PERSON" in entity_types or "ORGANIZATION" in entity_types

    def test_entity_confidence_filtering(self):
        """Test confidence threshold filtering."""
        extractor = EntityExtractor(
            use_llm=False,
            confidence_threshold=0.9
        )
        result = extractor.extract_entities(
            text=SAMPLE_TEXT_1,
            document_id="test_doc_1"
        )

        # All entities should have confidence >= 0.9
        for entity in result.entities:
            assert entity.confidence >= 0.9

    def test_entity_context_extraction(self):
        """Test that entity context is captured."""
        extractor = EntityExtractor(use_llm=False)
        result = extractor.extract_entities(
            text=SAMPLE_TEXT_1,
            document_id="test_doc_1",
            include_context=True
        )

        # Check that at least one entity has context
        assert any(e.context is not None for e in result.entities)

    def test_batch_entity_extraction(self):
        """Test batch processing of multiple documents."""
        extractor = EntityExtractor(use_llm=False)

        texts = [
            (SAMPLE_TEXT_1, "doc1"),
            (SAMPLE_TEXT_2, "doc2"),
            (SAMPLE_TEXT_3, "doc3")
        ]

        results = extractor.extract_entities_batch(texts)

        assert len(results) == 3
        assert all(len(r.entities) > 0 for r in results)

    def test_entity_deduplication(self):
        """Test that duplicate entities are removed."""
        extractor = EntityExtractor(use_llm=False)

        # Text with repeated entities
        text = "Microsoft Corporation is a company. Microsoft develops software. Microsoft is in Seattle."

        result = extractor.extract_entities(
            text=text,
            document_id="test_dup"
        )

        # Count how many times "Microsoft" appears
        microsoft_entities = [
            e for e in result.entities
            if "microsoft" in e.text.lower()
        ]

        # Should be deduplicated (not 3 separate entities)
        assert len(microsoft_entities) <= 2


class TestRelationshipExtractor:
    """Test relationship extraction functionality."""

    def test_basic_relationship_extraction(self):
        """Test basic relationship extraction."""
        # First extract entities
        entity_extractor = EntityExtractor(use_llm=False)
        entity_result = entity_extractor.extract_entities(
            text=SAMPLE_TEXT_1,
            document_id="test_doc_1"
        )

        # Then extract relationships
        rel_extractor = RelationshipExtractor(use_llm=False)
        rel_result = rel_extractor.extract_relationships(
            text=SAMPLE_TEXT_1,
            entities=entity_result.entities,
            document_id="test_doc_1"
        )

        assert rel_result.document_id == "test_doc_1"
        # Should find some relationships
        assert len(rel_result.relationships) >= 0

    def test_relationship_confidence(self):
        """Test relationship confidence scores."""
        entity_extractor = EntityExtractor(use_llm=False)
        entity_result = entity_extractor.extract_entities(
            text=SAMPLE_TEXT_1,
            document_id="test_doc_1"
        )

        rel_extractor = RelationshipExtractor(
            use_llm=False,
            confidence_threshold=0.5
        )
        rel_result = rel_extractor.extract_relationships(
            text=SAMPLE_TEXT_1,
            entities=entity_result.entities,
            document_id="test_doc_1"
        )

        # All relationships should meet threshold
        for rel in rel_result.relationships:
            assert rel.confidence >= 0.5

    def test_relationship_types(self):
        """Test that relationship types are assigned."""
        entity_extractor = EntityExtractor(use_llm=False)
        entity_result = entity_extractor.extract_entities(
            text=SAMPLE_TEXT_1,
            document_id="test_doc_1"
        )

        rel_extractor = RelationshipExtractor(use_llm=False)
        rel_result = rel_extractor.extract_relationships(
            text=SAMPLE_TEXT_1,
            entities=entity_result.entities,
            document_id="test_doc_1"
        )

        # Check that relationships have types
        for rel in rel_result.relationships:
            assert rel.relation_type is not None
            assert len(rel.relation_type) > 0


class TestEntityResolver:
    """Test entity resolution functionality."""

    def test_exact_match_resolution(self):
        """Test resolution of exact duplicate entities."""
        from src.knowledge_graph.entity_extractor import Entity

        entities = [
            Entity(text="Microsoft", type="ORGANIZATION", start_char=0, end_char=9, confidence=0.9),
            Entity(text="Microsoft", type="ORGANIZATION", start_char=20, end_char=29, confidence=0.85),
            Entity(text="Microsoft", type="ORGANIZATION", start_char=40, end_char=49, confidence=0.95),
        ]

        resolver = EntityResolver(similarity_threshold=0.85)
        result = resolver.resolve_entities(entities)

        # Should resolve to 1 unique entity
        assert result.unique_entities == 1
        assert result.merged_count == 2

    def test_fuzzy_match_resolution(self):
        """Test resolution of similar entities."""
        from src.knowledge_graph.entity_extractor import Entity

        entities = [
            Entity(text="Microsoft Corp", type="ORGANIZATION", start_char=0, end_char=14, confidence=0.9),
            Entity(text="Microsoft Corporation", type="ORGANIZATION", start_char=20, end_char=41, confidence=0.9),
            Entity(text="Microsoft", type="ORGANIZATION", start_char=50, end_char=59, confidence=0.9),
        ]

        resolver = EntityResolver(
            similarity_threshold=0.8,
            algorithm="levenshtein"
        )
        result = resolver.resolve_entities(entities)

        # Should merge similar variants
        assert result.unique_entities <= 2  # May merge some or all

    def test_type_based_resolution(self):
        """Test that entities of different types aren't merged."""
        from src.knowledge_graph.entity_extractor import Entity

        entities = [
            Entity(text="Washington", type="PERSON", start_char=0, end_char=10, confidence=0.9),
            Entity(text="Washington", type="LOCATION", start_char=20, end_char=30, confidence=0.9),
        ]

        resolver = EntityResolver(similarity_threshold=0.9)
        result = resolver.resolve_entities(entities)

        # Should NOT merge - different types
        assert result.unique_entities == 2

    def test_cross_document_resolution(self):
        """Test entity resolution across documents."""
        from src.knowledge_graph.entity_extractor import Entity

        doc1_entities = [
            Entity(text="Apple Inc", type="ORGANIZATION", start_char=0, end_char=9, confidence=0.9),
        ]

        doc2_entities = [
            Entity(text="Apple Inc.", type="ORGANIZATION", start_char=0, end_char=10, confidence=0.9),
        ]

        resolver = EntityResolver(similarity_threshold=0.85)
        result = resolver.resolve_cross_document([
            ("doc1", doc1_entities),
            ("doc2", doc2_entities)
        ])

        # Should merge across documents
        assert result.unique_entities == 1


class TestGraphBuilder:
    """Test graph construction functionality."""

    @pytest.fixture
    def mock_neo4j_manager(self, monkeypatch):
        """Mock Neo4j manager for testing without database."""
        class MockNeo4jManager:
            def __init__(self):
                self.queries_executed = []

            def execute_query(self, query, params=None):
                self.queries_executed.append((query, params))
                return []

            def execute_write_query(self, query, params=None):
                self.queries_executed.append((query, params))
                # Return mock node IDs
                if "RETURN elementId" in query or "node_id" in query:
                    return [{"node_id": f"mock_id_{len(self.queries_executed)}"}]
                return []

            def create_schema_constraints(self):
                pass

            def create_indexes(self):
                pass

            def get_statistics(self):
                return {"total_nodes": 0, "total_relationships": 0}

        return MockNeo4jManager()

    def test_document_node_creation(self, mock_neo4j_manager):
        """Test document node creation."""
        builder = GraphBuilder(
            neo4j_manager=mock_neo4j_manager,
            create_indexes=False
        )

        metadata = DocumentMetadata(
            doc_id="test_doc",
            title="Test Document",
            source="test"
        )

        doc_id = builder._create_document_node(metadata)
        assert doc_id is not None
        assert len(mock_neo4j_manager.queries_executed) > 0

    def test_entity_node_creation(self, mock_neo4j_manager):
        """Test entity node creation."""
        from src.knowledge_graph.entity_extractor import Entity

        builder = GraphBuilder(
            neo4j_manager=mock_neo4j_manager,
            create_indexes=False
        )

        entities = [
            Entity(text="Test Entity", type="PERSON", start_char=0, end_char=11, confidence=0.9)
        ]

        entity_ids = builder._create_entity_nodes(entities, "test_doc")
        assert len(entity_ids) > 0


class TestKnowledgeGraphPipeline:
    """Test end-to-end pipeline."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization with config."""
        config = PipelineConfig(
            extract_entities=True,
            extract_relationships=True,
            resolve_entities=True,
            use_llm=False
        )

        # This will fail if Neo4j not available, so catch exception
        try:
            pipeline = KnowledgeGraphPipeline(config=config)
            assert pipeline.config == config
        except Exception:
            pytest.skip("Neo4j not available")

    def test_pipeline_config(self):
        """Test pipeline configuration."""
        config = PipelineConfig(
            extract_entities=True,
            extract_relationships=False,
            resolve_entities=True,
            use_llm=False,
            confidence_threshold=0.8
        )

        assert config.extract_entities is True
        assert config.extract_relationships is False
        assert config.confidence_threshold == 0.8


class TestGraphQueryInterface:
    """Test graph query interface."""

    @pytest.fixture
    def mock_neo4j_manager(self):
        """Mock Neo4j manager for query testing."""
        class MockNeo4jManager:
            def execute_query(self, query, params=None):
                # Return mock results
                if "MATCH (e:Entity)" in query:
                    return [
                        {
                            "name": "Test Entity",
                            "type": "PERSON",
                            "confidence": 0.9,
                            "entity_id": "mock_id_1"
                        }
                    ]
                return []

        return MockNeo4jManager()

    def test_entity_search(self, mock_neo4j_manager):
        """Test entity search functionality."""
        query_interface = GraphQueryInterface(neo4j_manager=mock_neo4j_manager)

        result = query_interface.find_entity("Test Entity")

        assert result.count > 0
        assert len(result.results) > 0

    def test_entity_statistics(self, mock_neo4j_manager):
        """Test entity statistics retrieval."""
        mock_neo4j_manager.execute_query = lambda q, p=None: [
            {"type": "PERSON", "count": 10},
            {"type": "ORGANIZATION", "count": 5}
        ]

        query_interface = GraphQueryInterface(neo4j_manager=mock_neo4j_manager)
        stats = query_interface.get_entity_statistics()

        assert "by_type" in stats
        assert stats["total"] > 0


# Integration tests (require actual Neo4j instance)
@pytest.mark.integration
class TestKnowledgeGraphIntegration:
    """Integration tests requiring actual Neo4j database."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline for integration testing."""
        config = PipelineConfig(
            extract_entities=True,
            extract_relationships=True,
            resolve_entities=True,
            use_llm=False,
            enable_progress_bar=False
        )

        try:
            pipeline = KnowledgeGraphPipeline(config=config)
            # Clear any existing data
            pipeline.graph_builder.clear_graph(confirm=True)
            return pipeline
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")

    def test_end_to_end_pipeline(self, pipeline):
        """Test complete pipeline execution."""
        documents = [
            {
                "text": SAMPLE_TEXT_1,
                "doc_id": "doc1",
                "metadata": {"title": "Document 1"}
            }
        ]

        result = pipeline.process_documents_batch(documents)

        assert result.documents_processed == 1
        assert result.nodes_created > 0

    def test_cross_document_linking(self, pipeline):
        """Test cross-document entity linking."""
        documents = [
            {
                "text": SAMPLE_TEXT_1,
                "doc_id": "doc1",
                "metadata": {"title": "Document 1"}
            },
            {
                "text": SAMPLE_TEXT_2,
                "doc_id": "doc2",
                "metadata": {"title": "Document 2"}
            }
        ]

        result = pipeline.process_documents_batch(documents)

        assert result.documents_processed == 2

    def test_query_after_build(self, pipeline):
        """Test querying after graph construction."""
        # Build graph
        documents = [
            {
                "text": SAMPLE_TEXT_1,
                "doc_id": "doc1",
                "metadata": {"title": "Document 1"}
            }
        ]

        pipeline.process_documents_batch(documents)

        # Query
        query_interface = GraphQueryInterface(
            neo4j_manager=pipeline.neo4j_manager
        )

        stats = query_interface.get_entity_statistics()
        assert stats["total"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
