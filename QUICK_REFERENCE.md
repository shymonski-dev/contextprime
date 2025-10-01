# Dual Indexing Infrastructure - Quick Reference

## Quick Start

```bash
# One-command setup
./quickstart.sh

# Manual setup
docker-compose up -d neo4j qdrant
cd doctags_rag
pip install -r requirements.txt
python scripts/setup_databases.py
```

## Common Commands

### Docker Management
```bash
# Start databases
docker-compose up -d neo4j qdrant

# Stop databases
docker-compose down

# View logs
docker logs doctags-neo4j
docker logs doctags-qdrant

# Reset everything (WARNING: deletes all data)
docker-compose down -v
```

### Database Access
```bash
# Neo4j Browser
http://localhost:7474
# Username: neo4j, Password: password

# Qdrant Dashboard
http://localhost:6333/dashboard

# Qdrant API
curl http://localhost:6333/collections
```

### Testing
```bash
# Run all tests
pytest tests/test_indexing.py -v

# Run specific test class
pytest tests/test_indexing.py::TestNeo4jManager -v

# With coverage
pytest tests/test_indexing.py --cov=src --cov-report=html
```

## Code Examples

### Initialize Managers
```python
from src.knowledge_graph.neo4j_manager import Neo4jManager
from src.retrieval.qdrant_manager import QdrantManager
from src.retrieval.hybrid_retriever import HybridRetriever

# Individual managers
neo4j = Neo4jManager()
qdrant = QdrantManager()

# Hybrid retriever (includes both)
retriever = HybridRetriever()
```

### Index a Document
```python
# In Neo4j (graph)
neo4j.create_node(
    labels=["Document"],
    properties={
        "doc_id": "doc_001",
        "title": "My Document",
        "text": "Content...",
        "embedding": [0.1, 0.2, ...],  # Your embedding
        "metadata": {...}
    }
)

# In Qdrant (vector)
qdrant.insert_vector(
    vector=[0.1, 0.2, ...],  # Your embedding
    metadata={
        "doc_id": "doc_001",
        "title": "My Document",
        "text": "Content..."
    },
    vector_id="doc_001"
)
```

### Batch Indexing
```python
from src.knowledge_graph.neo4j_manager import GraphNode
from src.retrieval.qdrant_manager import VectorPoint

# Neo4j batch
nodes = [
    GraphNode(
        id=None,
        labels=["Document"],
        properties={"doc_id": f"doc_{i}", "title": f"Doc {i}"}
    )
    for i in range(100)
]
neo4j.create_nodes_batch(nodes, batch_size=1000)

# Qdrant batch
vectors = [
    VectorPoint(
        id=f"doc_{i}",
        vector=get_embedding(i),
        metadata={"doc_id": f"doc_{i}"}
    )
    for i in range(100)
]
qdrant.insert_vectors_batch(vectors, batch_size=100)
```

### Search
```python
# Hybrid search (recommended)
results, metrics = retriever.search(
    query_vector=query_embedding,
    query_text="What is machine learning?",
    top_k=10
)

# Vector-only search
results, metrics = retriever.search(
    query_vector=query_embedding,
    query_text="factual query",
    strategy=SearchStrategy.VECTOR_ONLY
)

# Graph-only search
results, metrics = retriever.search(
    query_vector=query_embedding,
    query_text="How are X and Y related?",
    strategy=SearchStrategy.GRAPH_ONLY,
    vector_index_name="document_embeddings"
)
```

### Create Relationships
```python
# Single relationship
neo4j.create_relationship(
    start_node_id="node_id_1",
    end_node_id="node_id_2",
    rel_type="REFERENCES",
    properties={"weight": 0.8}
)

# Batch relationships
from src.knowledge_graph.neo4j_manager import GraphRelationship

relationships = [
    GraphRelationship(
        type="REFERENCES",
        start_node="id_1",
        end_node="id_2",
        properties={"weight": 0.8}
    )
    for _ in range(100)
]
neo4j.create_relationships_batch(relationships)
```

### Graph Traversal
```python
# Traverse from a node
paths = neo4j.traverse_graph(
    start_node_id="node_id",
    relationship_types=["REFERENCES", "MENTIONS"],
    direction="both",
    max_depth=3,
    limit=20
)

# Find shortest path
path = neo4j.find_shortest_path(
    start_node_id="node_1",
    end_node_id="node_2",
    relationship_types=["REFERENCES"]
)
```

### Vector Index
```python
# Create vector index in Neo4j
neo4j.initialize_vector_index(
    index_name="document_embeddings",
    label="Document",
    property_name="embedding",
    dimensions=1536,  # OpenAI embeddings
    similarity_function="cosine"
)

# Vector search
results = neo4j.vector_similarity_search(
    index_name="document_embeddings",
    query_vector=query_embedding,
    top_k=10,
    filters={"category": "technical"}
)
```

### Filtering
```python
# Qdrant filtering
results = qdrant.search(
    query_vector=query_embedding,
    top_k=10,
    filters={
        "category": "technical",  # Exact match
        "year": {"gte": 2020, "lte": 2024},  # Range
        "tags": ["ai", "ml"]  # Match any
    }
)
```

### Health Checks
```python
# Individual health checks
neo4j_ok = neo4j.health_check()
qdrant_ok = qdrant.health_check()

# Combined health check
health = retriever.health_check()
print(health)  # {"neo4j": True, "qdrant": True}
```

### Statistics
```python
# Neo4j statistics
stats = neo4j.get_statistics()
print(stats)
# {
#     "total_nodes": 1000,
#     "total_relationships": 500,
#     "nodes_by_label": {"Document": 800, "Entity": 200},
#     "relationships_by_type": {"REFERENCES": 500}
# }

# Qdrant statistics
stats = qdrant.get_statistics()
print(stats)
# {
#     "collection_name": "doctags_vectors",
#     "total_vectors": 1000,
#     "vector_dimension": 1536,
#     "distance_metric": "Cosine"
# }

# Combined statistics
stats = retriever.get_statistics()
```

### Query Routing
```python
# Automatic routing
query_type = retriever.detect_query_type("What is X?")
# QueryType.FACTUAL

strategy = retriever.route_query(query_type)
# SearchStrategy.VECTOR_ONLY

# Use in search
results, metrics = retriever.search(
    query_vector=query_embedding,
    query_text=query_text,
    strategy=strategy  # Auto-selected
)
```

### Result Processing
```python
# Process search results
for result in results:
    print(f"ID: {result.id}")
    print(f"Score: {result.score:.3f}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Source: {result.source}")  # "vector", "graph", or "hybrid"
    print(f"Content: {result.content}")

    if result.graph_context:
        print(f"Labels: {result.graph_context['labels']}")

# Rerank results
reranked = retriever.rerank_results(
    results,
    query_text,
    use_diversity=True
)

# Enrich with graph context
enriched = retriever.enrich_with_graph_context(
    result,
    max_depth=2
)
```

### Custom Configuration
```python
from src.core.config import Neo4jConfig, QdrantConfig

# Custom Neo4j config
neo4j_config = Neo4jConfig(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="your_password",
    max_connection_pool_size=100
)
neo4j = Neo4jManager(neo4j_config)

# Custom Qdrant config
qdrant_config = QdrantConfig(
    host="localhost",
    port=6333,
    collection_name="my_collection",
    vector_size=768  # Different embedding size
)
qdrant = QdrantManager(qdrant_config)

# Custom retriever weights
retriever = HybridRetriever(
    neo4j_manager=neo4j,
    qdrant_manager=qdrant,
    vector_weight=0.8,  # 80% vector
    graph_weight=0.2    # 20% graph
)
```

## Configuration Quick Reference

### Environment Variables
```bash
# .env file
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=password
QDRANT_HOST=localhost
QDRANT_PORT=6333
OPENAI_API_KEY=your_api_key
```

### Config File (`config/config.yaml`)
```yaml
neo4j:
  max_connection_pool_size: 50
  connection_timeout: 30

qdrant:
  vector_size: 1536
  distance_metric: "cosine"

retrieval:
  hybrid_search:
    vector_weight: 0.7
    graph_weight: 0.3
  max_results: 10
```

## Troubleshooting

### Connection Issues
```python
# Check if databases are running
docker ps | grep neo4j
docker ps | grep qdrant

# Test connections
from src.core.config import get_settings
settings = get_settings()
results = settings.validate_connections()
print(results)  # {"neo4j": True, "qdrant": True, ...}
```

### Performance Issues
```python
# Check database stats
neo4j_stats = neo4j.get_statistics()
qdrant_stats = qdrant.get_statistics()

# Check search metrics
results, metrics = retriever.search(...)
print(f"Vector time: {metrics.vector_time_ms}ms")
print(f"Graph time: {metrics.graph_time_ms}ms")
print(f"Total time: {metrics.total_time_ms}ms")
```

### Clear Data
```python
# Clear Qdrant collection
qdrant.clear_collection(confirm=True)

# Clear Neo4j database
neo4j.clear_database(confirm=True)
```

## File Locations

```
/Users/simonkelly/SUPER_RAG/
├── doctags_rag/
│   ├── src/
│   │   ├── knowledge_graph/neo4j_manager.py    # Neo4j manager
│   │   └── retrieval/
│   │       ├── qdrant_manager.py               # Qdrant manager
│   │       └── hybrid_retriever.py             # Hybrid retriever
│   ├── scripts/
│   │   ├── setup_databases.py                  # Setup script
│   │   └── example_usage.py                    # Examples
│   ├── tests/test_indexing.py                  # Tests
│   └── config/config.yaml                      # Configuration
├── docker-compose.yml                           # Docker setup
├── quickstart.sh                                # Quick start
└── DUAL_INDEXING_SETUP.md                      # Full documentation
```

## Additional Resources

- **Full Documentation**: `/Users/simonkelly/SUPER_RAG/DUAL_INDEXING_SETUP.md`
- **Implementation Details**: `/Users/simonkelly/SUPER_RAG/IMPLEMENTATION_SUMMARY.md`
- **Examples**: `/Users/simonkelly/SUPER_RAG/doctags_rag/scripts/example_usage.py`
- **Tests**: `/Users/simonkelly/SUPER_RAG/doctags_rag/tests/test_indexing.py`

## Support

### Documentation
- Neo4j: https://neo4j.com/docs/
- Qdrant: https://qdrant.tech/documentation/

### Logs
```bash
# Application logs (if using loguru)
tail -f logs/app.log

# Docker logs
docker logs -f doctags-neo4j
docker logs -f doctags-qdrant
```

### Monitoring
```bash
# With monitoring profile
docker-compose --profile monitoring up -d
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```
