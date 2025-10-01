# Knowledge Graph Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies

```bash
cd /Users/simonkelly/SUPER_RAG/doctags_rag

# Install Python packages
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_lg
```

### 2. Start Neo4j

**Option A: Docker (Recommended)**
```bash
docker run -d \
  --name neo4j-doctags \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

**Option B: Local Installation**
- Download from https://neo4j.com/download/
- Start Neo4j Desktop or service
- Set password to "password" (or update config.yaml)

### 3. Verify Connection

```bash
# Check Neo4j is running
curl http://localhost:7474

# Or open in browser
open http://localhost:7474
```

### 4. Run Demo

```bash
python scripts/build_sample_kg.py
```

Expected output:
```
================================================================================
  DocTags RAG - Knowledge Graph Construction Demo
================================================================================

Configuring knowledge graph pipeline...
  - Entity extraction: True
  - Relationship extraction: True
  - Entity resolution: True
  - Use LLM: False
  - Confidence threshold: 0.7

Processing Documents
Processing 4 sample documents...

✓ Processing complete in 12.45s

Results:
  - Documents processed: 4
  - Total entities extracted: 87
  - Unique entities: 45
  - Total relationships: 23
  - Nodes created: 105
  - Edges created: 68
```

## Basic Usage

### Example 1: Simple Entity Extraction

```python
from src.knowledge_graph import EntityExtractor

# Initialize extractor
extractor = EntityExtractor()

# Extract entities
result = extractor.extract_entities(
    text="Apple was founded by Steve Jobs in Cupertino.",
    document_id="doc1"
)

# Print results
for entity in result.entities:
    print(f"{entity.text} ({entity.type}) - confidence: {entity.confidence:.2f}")
```

Output:
```
Apple (ORGANIZATION) - confidence: 0.92
Steve Jobs (PERSON) - confidence: 0.95
Cupertino (LOCATION) - confidence: 0.88
```

### Example 2: Full Pipeline

```python
from src.knowledge_graph import KnowledgeGraphPipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    extract_entities=True,
    extract_relationships=True,
    resolve_entities=True,
    use_llm=False,
    confidence_threshold=0.7
)

# Initialize
pipeline = KnowledgeGraphPipeline(config=config)

# Process document
documents = [{
    "text": "OpenAI created GPT-4. Sam Altman leads OpenAI.",
    "doc_id": "ai_doc",
    "metadata": {"title": "AI News"}
}]

result = pipeline.process_documents_batch(documents)

print(f"Created {result.nodes_created} nodes")
print(f"Created {result.edges_created} relationships")
```

### Example 3: Query the Graph

```python
from src.knowledge_graph import GraphQueryInterface

# Initialize query interface
query = GraphQueryInterface()

# Find entity
result = query.find_entity("OpenAI")
print(f"Found {result.count} matches")

# Get most connected entities
top = query.get_most_connected_entities(limit=10)
for entity in top.results:
    print(f"{entity['name']} - {entity['degree']} connections")

# Find relationships
rels = query.find_relationships("OpenAI")
for rel in rels.results:
    print(f"{rel['source']} --[{rel['relationship']}]--> {rel['target']}")
```

## Common Tasks

### Task 1: Process Your Documents

```python
from src.knowledge_graph import KnowledgeGraphPipeline

pipeline = KnowledgeGraphPipeline()

# Load your documents
documents = [
    {
        "text": open("doc1.txt").read(),
        "doc_id": "doc1",
        "metadata": {"title": "Document 1", "source": "file"}
    },
    {
        "text": open("doc2.txt").read(),
        "doc_id": "doc2",
        "metadata": {"title": "Document 2", "source": "file"}
    }
]

# Process
result = pipeline.process_documents_batch(documents)
```

### Task 2: Find Similar Documents

```python
from src.knowledge_graph import GraphQueryInterface

query = GraphQueryInterface()

# Find documents similar to doc1
similar = query.find_similar_documents(
    doc_id="doc1",
    min_shared_entities=3,
    limit=5
)

for doc in similar.results:
    print(f"{doc['doc_id']}: {doc['shared_entities']} shared entities")
```

### Task 3: Extract Entities by Type

```python
from src.knowledge_graph import GraphQueryInterface

query = GraphQueryInterface()

# Get all organizations
orgs = query.search_entities(entity_type="ORGANIZATION", limit=50)

print(f"Found {orgs.count} organizations:")
for org in orgs.results:
    print(f"  - {org['name']}")
```

### Task 4: Find Relationships Between Entities

```python
from src.knowledge_graph import GraphQueryInterface

query = GraphQueryInterface()

# Find path between two entities
path = query.find_shortest_path("OpenAI", "GPT-4", max_depth=5)

if path.count > 0:
    entities = path.results[0]['entity_names']
    relationships = path.results[0]['relationship_types']

    print("Path found:")
    for i, entity in enumerate(entities):
        print(f"  {entity}")
        if i < len(relationships):
            print(f"    --[{relationships[i]}]-->")
```

### Task 5: Update Existing Documents

```python
from src.knowledge_graph import KnowledgeGraphPipeline

pipeline = KnowledgeGraphPipeline()

# Update document
pipeline.update_document(
    text="Updated content...",
    doc_id="doc1",
    doc_metadata={"title": "Updated Document"}
)

# Delete document
pipeline.delete_document("doc2")
```

## Neo4j Browser Queries

Open http://localhost:7474 and try these Cypher queries:

### View all nodes
```cypher
MATCH (n) RETURN n LIMIT 25
```

### View entities and relationships
```cypher
MATCH (e:Entity)-[r]-(other:Entity)
RETURN e, r, other
LIMIT 50
```

### Find specific entity type
```cypher
MATCH (e:Entity {type: 'ORGANIZATION'})
RETURN e.name, e.confidence
ORDER BY e.confidence DESC
LIMIT 20
```

### Documents with most entities
```cypher
MATCH (d:Document)-[:CONTAINS]->(e:Entity)
WITH d, count(e) as entity_count
RETURN d.doc_id, d.title, entity_count
ORDER BY entity_count DESC
LIMIT 10
```

### Most connected entities
```cypher
MATCH (e:Entity)-[r]-()
WITH e, count(r) as degree
RETURN e.name, e.type, degree
ORDER BY degree DESC
LIMIT 20
```

### Find relationship patterns
```cypher
MATCH (person:Entity {type: 'PERSON'})-[r:WORKS_FOR]->(org:Entity {type: 'ORGANIZATION'})
RETURN person.name, org.name
LIMIT 20
```

## Configuration

Edit `config/config.yaml`:

```yaml
knowledge_graph:
  entity_extraction:
    confidence_threshold: 0.7  # Lower = more entities, higher = more precision
  entity_resolution:
    similarity_threshold: 0.85  # Lower = more merging, higher = more unique entities
    algorithm: "hybrid"  # levenshtein, embedding, or hybrid
```

## Troubleshooting

### Neo4j Connection Error
```
Error: Could not connect to Neo4j
```

**Solution:**
1. Check Neo4j is running: `docker ps | grep neo4j`
2. Verify port 7687 is accessible: `nc -zv localhost 7687`
3. Check password in `config/config.yaml`

### spaCy Model Not Found
```
OSError: [E050] Can't find model 'en_core_web_lg'
```

**Solution:**
```bash
python -m spacy download en_core_web_lg
```

### Out of Memory
```
MemoryError: Unable to allocate array
```

**Solution:**
```python
# Use smaller batch size
config = PipelineConfig(batch_size=5)
pipeline = KnowledgeGraphPipeline(config=config)
```

### Slow Processing
**Solution:**
```python
# Disable entity resolution for faster processing
config = PipelineConfig(
    extract_entities=True,
    extract_relationships=True,
    resolve_entities=False  # Faster but more duplicate entities
)
```

## Next Steps

1. **Read Full Documentation**: See `docs/KNOWLEDGE_GRAPH.md`
2. **Run Tests**: `pytest tests/test_knowledge_graph.py -v`
3. **Explore Neo4j Browser**: http://localhost:7474
4. **Try Advanced Features**: Custom entity types, LLM extraction, embeddings
5. **Integrate with RAG**: Use graph for enhanced retrieval

## Getting Help

- Check logs in console for detailed error messages
- Review Neo4j Browser for query debugging
- See test examples in `tests/test_knowledge_graph.py`
- Read component documentation in `docs/KNOWLEDGE_GRAPH.md`

## Performance Tips

1. **Batch Processing**: Process documents in batches for efficiency
2. **Entity Resolution**: Disable for speed, enable for accuracy
3. **Confidence Threshold**: Higher = fewer entities but higher quality
4. **Neo4j Indexes**: Automatically created, but verify with `SHOW INDEXES`
5. **Memory**: Use smaller batch sizes for large documents

## Example Output

After running the demo, you should see:

```
✓ Processing complete in 12.45s

Entity Statistics:
  by_type:
    ORGANIZATION: 15
    PERSON: 12
    PRODUCT: 8
    LOCATION: 6
  total: 45

Most Connected Entities:
  - OpenAI (ORGANIZATION) - 8 connections
  - GPT-4 (PRODUCT) - 6 connections
  - Google (ORGANIZATION) - 5 connections
```

## Ready to Scale?

Once you're comfortable with the basics:

1. Process your entire document collection
2. Enable LLM extraction for better accuracy
3. Use embedding-based entity resolution
4. Create custom entity types for your domain
5. Build advanced queries for your use case

Happy graph building! 🚀
