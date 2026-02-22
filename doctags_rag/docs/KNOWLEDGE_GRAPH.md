# Knowledge Graph System Documentation

## Overview

The Contextprime Knowledge Graph system provides comprehensive entity and relationship extraction from documents, building a connected knowledge graph in Neo4j. This enables semantic understanding, relationship discovery, and advanced querying capabilities.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Knowledge Graph Pipeline                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Documents → Entity Extraction → Relationship Extraction     │
│                      ↓                    ↓                   │
│              Entity Resolution    →   Graph Construction     │
│                                           ↓                   │
│                                      Neo4j Graph              │
│                                           ↓                   │
│                                    Query Interface            │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Entity Extractor (`entity_extractor.py`)

Extracts named entities from text using multiple approaches:

**Features:**
- **spaCy NER**: Base entity recognition for standard types (PERSON, ORG, LOC, DATE, etc.)
- **LLM Enhancement**: Optional LLM-based extraction for domain-specific entities
- **Custom Entity Types**: Support for CONCEPT, TECHNOLOGY, METHOD, METRIC, DATASET
- **Confidence Scoring**: Each entity has a confidence score
- **Context Extraction**: Captures surrounding text for disambiguation
- **Batch Processing**: Efficient processing of multiple documents

**Usage:**
```python
from src.knowledge_graph import EntityExtractor

extractor = EntityExtractor(
    spacy_model="en_core_web_lg",
    use_llm=False,
    confidence_threshold=0.7
)

result = extractor.extract_entities(
    text="John Smith works for Microsoft in Seattle.",
    document_id="doc1",
    extract_attributes=True,
    include_context=True
)

print(f"Extracted {len(result.entities)} entities")
for entity in result.entities:
    print(f"  - {entity.text} ({entity.type}) - {entity.confidence}")
```

### 2. Relationship Extractor (`relationship_extractor.py`)

Extracts relationships between entities using multiple techniques:

**Features:**
- **Dependency Parsing**: Grammatical relationships from sentence structure
- **Pattern Matching**: Rule-based patterns (e.g., "X works for Y")
- **LLM Inference**: Optional LLM for complex/implicit relationships
- **Relationship Types**: WORKS_FOR, LOCATED_IN, OWNS, CREATED_BY, PART_OF, etc.
- **Confidence Scoring**: Each relationship has a confidence score
- **Evidence Tracking**: Captures supporting text

**Usage:**
```python
from src.knowledge_graph import RelationshipExtractor

rel_extractor = RelationshipExtractor(
    spacy_model="en_core_web_lg",
    use_llm=False,
    confidence_threshold=0.7
)

rel_result = rel_extractor.extract_relationships(
    text="John Smith works for Microsoft in Seattle.",
    entities=entity_result.entities,
    document_id="doc1"
)

for rel in rel_result.relationships:
    print(f"{rel.source_entity.text} --[{rel.relation_type}]--> {rel.target_entity.text}")
```

### 3. Entity Resolver (`entity_resolver.py`)

Resolves and disambiguates entities across documents:

**Features:**
- **String Similarity**: Levenshtein distance, Jaro-Winkler, fuzzy matching
- **Embedding Similarity**: Semantic similarity using sentence transformers
- **Hybrid Approach**: Combines string and semantic matching
- **Cross-Document Linking**: Merges entities across documents
- **Abbreviation Resolution**: Expands abbreviations to full forms
- **Merge History**: Tracks entity merge provenance

**Usage:**
```python
from src.knowledge_graph import EntityResolver

resolver = EntityResolver(
    similarity_threshold=0.85,
    algorithm="hybrid"
)

resolution_result = resolver.resolve_entities(entities)

print(f"Resolved {len(entities)} entities to {resolution_result.unique_entities} unique entities")
print(f"Merged {resolution_result.merged_count} duplicates")
```

### 4. Graph Builder (`graph_builder.py`)

Constructs the knowledge graph in Neo4j:

**Features:**
- **Document Nodes**: Rich metadata and properties
- **Entity Nodes**: Deduplicated with variant tracking
- **Relationship Edges**: Typed with properties
- **Hierarchical Structure**: Document → Chunk → Entity
- **Cross-Document Links**: SHARES_ENTITIES relationships
- **Batch Operations**: Efficient bulk creation
- **Schema Management**: Automatic constraints and indexes

**Graph Schema:**
```
(Document)-[:HAS_CHUNK]->(Chunk)-[:MENTIONS]->(Entity)
(Document)-[:CONTAINS]->(Entity)
(Entity)-[:WORKS_FOR|LOCATED_IN|...]->(Entity)
(Document)-[:SHARES_ENTITIES]->(Document)
(Chunk)-[:REFERENCES]->(Chunk)
```

**Usage:**
```python
from src.knowledge_graph import GraphBuilder, DocumentMetadata, ChunkMetadata

builder = GraphBuilder()

metadata = DocumentMetadata(
    doc_id="doc1",
    title="Sample Document",
    source="test"
)

result = builder.build_document_graph(
    doc_metadata=metadata,
    chunks=chunks,
    entity_result=entity_result,
    relationship_result=relationship_result,
    resolution_result=resolution_result
)
```

### 5. Knowledge Graph Pipeline (`kg_pipeline.py`)

Orchestrates the complete graph construction process:

**Features:**
- **Multi-Stage Processing**: Extraction → Resolution → Construction
- **Batch Processing**: Process multiple documents efficiently
- **Progress Tracking**: Real-time progress updates
- **Error Recovery**: Continues processing on errors
- **Incremental Updates**: Add/update documents without rebuilding
- **Statistics Generation**: Comprehensive metrics

**Usage:**
```python
from src.knowledge_graph import KnowledgeGraphPipeline, PipelineConfig

config = PipelineConfig(
    extract_entities=True,
    extract_relationships=True,
    resolve_entities=True,
    use_llm=False,
    confidence_threshold=0.7
)

pipeline = KnowledgeGraphPipeline(config=config)

documents = [
    {
        "text": "Your document text...",
        "doc_id": "doc1",
        "metadata": {"title": "Document 1"}
    }
]

result = pipeline.process_documents_batch(documents)

print(f"Processed {result.documents_processed} documents")
print(f"Created {result.nodes_created} nodes and {result.edges_created} edges")
```

### 6. Graph Query Interface (`graph_queries.py`)

High-level interface for querying the knowledge graph:

**Query Types:**
- **Entity Queries**: Search, filter, find by type
- **Relationship Queries**: Find relationships, patterns
- **Path Finding**: Shortest paths, all paths, common neighbors
- **Document Queries**: Entities in document, similar documents
- **Analytics**: Most connected entities, PageRank, communities

**Usage:**
```python
from src.knowledge_graph import GraphQueryInterface

query_interface = GraphQueryInterface()

# Find entity
result = query_interface.find_entity("OpenAI", fuzzy=True)

# Get entity neighbors
neighbors = query_interface.get_entity_neighbors(entity_id, limit=10)

# Find shortest path
path = query_interface.find_shortest_path("OpenAI", "GPT-4")

# Get document entities
entities = query_interface.get_document_entities("doc1")

# Find similar documents
similar = query_interface.find_similar_documents("doc1", min_shared_entities=3)

# Get most connected entities
top_entities = query_interface.get_most_connected_entities(limit=20)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg
```

### 2. Start Neo4j

```bash
docker run -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### 3. Run Sample Demo

```bash
python scripts/build_sample_kg.py
```

### 4. Use in Your Code

```python
from src.knowledge_graph import KnowledgeGraphPipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    extract_entities=True,
    extract_relationships=True,
    resolve_entities=True,
    use_llm=False
)

# Initialize pipeline
pipeline = KnowledgeGraphPipeline(config=config)

# Process documents
documents = [
    {
        "text": "Your text here...",
        "doc_id": "doc1",
        "metadata": {"title": "Document 1"}
    }
]

result = pipeline.process_documents_batch(documents)

# Query the graph
from src.knowledge_graph import GraphQueryInterface
query_interface = GraphQueryInterface()

# Search entities
entities = query_interface.search_entities(entity_type="ORGANIZATION")

# Find relationships
rels = query_interface.find_relationships("Company Name")
```

## Configuration

Edit `config/config.yaml`:

```yaml
knowledge_graph:
  entity_extraction:
    model: "spacy"
    spacy_model: "en_core_web_lg"
    batch_size: 32
    confidence_threshold: 0.7
  relationship_extraction:
    enable: true
    max_distance: 5
  entity_resolution:
    similarity_threshold: 0.85
    algorithm: "hybrid"  # levenshtein, embedding, or hybrid
```

## Advanced Features

### Custom Entity Types

```python
extractor = EntityExtractor()

# Add custom entity type
extractor.add_custom_entity_type(
    entity_type="FRAMEWORK",
    patterns=["TensorFlow", "PyTorch", "Scikit-learn"],
    case_sensitive=False
)
```

### Entity Resolution with Embeddings

```python
resolver = EntityResolver(
    similarity_threshold=0.85,
    embedding_model="all-MiniLM-L6-v2",
    use_embeddings=True,
    algorithm="hybrid"
)
```

### Cross-Document Entity Linking

```python
# Resolve entities across multiple documents
entity_sets = [
    ("doc1", doc1_entities),
    ("doc2", doc2_entities),
    ("doc3", doc3_entities)
]

result = resolver.resolve_cross_document(entity_sets)
```

### Custom Cypher Queries

```python
query_interface = GraphQueryInterface()

custom_query = """
MATCH (e:Entity)-[r:WORKS_FOR]->(org:Entity {type: 'ORGANIZATION'})
RETURN e.name as employee, org.name as company
ORDER BY company
"""

result = query_interface.execute_custom_query(custom_query)
```

## Performance Optimization

### Batch Processing

```python
# Process documents in batches
pipeline.config.batch_size = 50

documents = load_large_document_set()
result = pipeline.process_documents_batch(documents)
```

### Incremental Updates

```python
# Update single document
pipeline.update_document(
    text=updated_text,
    doc_id="doc1",
    doc_metadata=metadata
)

# Delete document
pipeline.delete_document("doc1")
```

### Index Optimization

The system automatically creates indexes for:
- Document IDs
- Entity names
- Entity types
- Chunk document IDs

For vector similarity search:
```python
# Initialize vector index
neo4j_manager.initialize_vector_index(
    index_name="entity_embeddings",
    label="Entity",
    property_name="embedding",
    dimensions=384,
    similarity_function="cosine"
)
```

## Legal Cross-Reference Edges

For UK/EU legal documents the graph is enriched with explicit citation edges between chunks.

### How it works

1. After each chunk is ingested into Qdrant and Neo4j, `DocumentIngestionPipeline._store_cross_references` calls `extract_cross_references(chunk_id, content, doc_id)` to detect patterns such as:
   - `Article 6`, `Article 17(3)` (article references)
   - `Section 12.3` (section references)
   - `Schedule 2`, `Annex I` (schedule/annex references)
   - `paragraph 3(a)` (paragraph references)

2. Detected references are stored as `(:Chunk)-[:REFERENCES]->(:Chunk)` edges via `Neo4jManager.store_cross_references()`. The Cypher uses `MERGE` so repeated ingestion is idempotent.

3. Unresolvable targets (no matching Chunk node found) are silently skipped.

### Updated Graph Schema

```
(Document)-[:HAS_CHUNK]->(Chunk)-[:MENTIONS]->(Entity)
(Document)-[:CONTAINS]->(Entity)
(Entity)-[:WORKS_FOR|LOCATED_IN|...]->(Entity)
(Document)-[:SHARES_ENTITIES]->(Document)
(Chunk)-[:REFERENCES]->(Chunk)          ← legal cross-reference edges
```

### Querying cross-references

```python
from src.knowledge_graph import Neo4jManager

manager = Neo4jManager()

# Find all chunks that cite Article 6
results = manager.execute_read_query(
    """
    MATCH (src:Chunk)-[r:REFERENCES]->(tgt:Chunk)
    WHERE r.target_label = 'article_6'
    RETURN src.chunk_id, tgt.chunk_id, r.ref_type
    """,
    {}
)
```

### Standalone extraction

```python
from src.processing.cross_reference_extractor import extract_cross_references

refs = extract_cross_references(
    chunk_id="gdpr_chunk_0042",
    content="The controller shall comply pursuant to Article 17(3).",
    doc_id="gdpr",
)
# [CrossRef(source_chunk_id='gdpr_chunk_0042', target_label='article_17(3)',
#           ref_type='article', doc_id='gdpr')]
```

### Ingestion report

`IngestionReport.cross_references_stored` counts the total edges created during a pipeline run:

```python
report = pipeline.process_files([Path("gdpr.pdf")])
print(f"Cross-reference edges: {report.cross_references_stored}")
```

## Testing

Run tests:
```bash
# Unit tests
pytest tests/test_knowledge_graph.py -v
pytest tests/test_cross_reference_extractor.py -v

# Integration tests (requires Neo4j)
pytest tests/test_knowledge_graph.py -v -m integration
pytest tests/test_indexing.py::TestNeo4jCrossReferenceEdges -v
```

## Troubleshooting

### Neo4j Connection Issues

```python
# Check connection
from src.knowledge_graph import Neo4jManager

manager = Neo4jManager()
if manager.health_check():
    print("Connected to Neo4j")
else:
    print("Connection failed")
```

### Entity Extraction Issues

```bash
# Download spaCy model
python -m spacy download en_core_web_lg

# Verify installation
python -c "import spacy; nlp = spacy.load('en_core_web_lg'); print('OK')"
```

### Memory Issues with Large Documents

```python
# Use smaller batch size
config = PipelineConfig(batch_size=10)

# Process documents one at a time
for doc in documents:
    pipeline.process_document(doc["text"], doc["doc_id"])
```

## Best Practices

1. **Entity Resolution**: Always enable for multi-document graphs
2. **Confidence Thresholds**: Adjust based on your precision/recall needs
3. **Batch Processing**: Use for large document sets
4. **Custom Patterns**: Add domain-specific entity patterns
5. **Index Optimization**: Create vector indexes for similarity search
6. **Regular Maintenance**: Periodically clean up low-confidence entities

## Examples

See `scripts/build_sample_kg.py` for a complete working example.

## API Reference

Full API documentation:
- [Entity Extractor API](api/entity_extractor.md)
- [Relationship Extractor API](api/relationship_extractor.md)
- [Entity Resolver API](api/entity_resolver.md)
- [Graph Builder API](api/graph_builder.md)
- [Pipeline API](api/kg_pipeline.md)
- [Query Interface API](api/graph_queries.md)

## Contributing

When adding new features:
1. Add type hints to all functions
2. Include docstrings with examples
3. Add tests to `tests/test_knowledge_graph.py`
4. Update this documentation

## License

See main project LICENSE file.
