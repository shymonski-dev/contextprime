# Dual Indexing Infrastructure - Setup Guide

## Overview

This document describes the production-ready dual indexing infrastructure that combines Neo4j graph database with Qdrant vector database for the Contextprime system.

## Components

### 1. Neo4j Manager (`src/knowledge_graph/neo4j_manager.py`)

A comprehensive graph database manager with:

- **Connection Management**: Connection pooling with configurable pool size and timeouts
- **CRUD Operations**: Create, read, update, delete nodes and relationships
- **Vector Search**: HNSW vector indexes for similarity search
- **Graph Traversal**: Path finding, pattern matching, neighborhood exploration
- **Batch Operations**: Efficient bulk inserts for nodes and relationships
- **Schema Management**: Constraints and indexes for data integrity
- **Error Handling**: Automatic retry on transient errors with exponential backoff

**Key Features:**
- Parameterized queries for SQL injection prevention
- Transaction support with automatic rollback
- Health checks and connection validation
- Comprehensive statistics and monitoring

### 2. Qdrant Manager (`src/retrieval/qdrant_manager.py`)

A complete vector database manager with:

- **Collection Management**: Create, delete, and manage vector collections
- **Vector Operations**: Insert, update, delete, and retrieve vectors
- **Similarity Search**: Fast vector search with metadata filtering
- **Batch Operations**: Efficient bulk vector insertion
- **Multiple Distance Metrics**: Cosine, Euclidean, Dot product
- **Advanced Filtering**: Range queries, match conditions, complex filters

**Key Features:**
- Connection pooling and retry logic
- Automatic index optimization
- Scroll API for large result sets
- Collection statistics and monitoring

### 3. Hybrid Retriever (`src/retrieval/hybrid_retriever.py`)

An intelligent retrieval system that combines both databases:

- **Reciprocal Rank Fusion**: Combines results using RRF algorithm
- **Query Routing**: Automatic query type detection and strategy selection
- **Configurable Weights**: Adjustable vector vs graph weights
- **Confidence Scoring**: Multi-factor confidence calculation
- **Result Deduplication**: Removes redundant results
- **Context Enrichment**: Adds graph context to results

**Query Types:**
- **Factual**: "What is X?" → Vector search
- **Relationship**: "How are X and Y related?" → Graph search
- **Complex**: "Explain why..." → Hybrid search

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Docker and Docker Compose** (for databases)

### Step 1: Install Python Dependencies

```bash
cd /Users/simonkelly/SUPER_RAG/doctags_rag
pip install -r requirements.txt
```

### Step 2: Start Databases with Docker Compose

```bash
cd /Users/simonkelly/SUPER_RAG
docker-compose up -d neo4j qdrant
```

This starts:
- **Neo4j** on ports 7474 (HTTP) and 7687 (Bolt)
- **Qdrant** on ports 6333 (HTTP) and 6334 (gRPC)

**Optional services** (use `--profile monitoring`):
```bash
docker-compose --profile monitoring up -d
```

This additionally starts:
- **Redis** on port 6379 (caching)
- **Prometheus** on port 9090 (metrics)
- **Grafana** on port 3000 (visualization)

### Step 3: Configure Environment

Create a `.env` file in the `doctags_rag` directory:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=replace_with_strong_password

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333

# API Keys (if using embeddings)
OPENAI_API_KEY=your_api_key_here
```

### Step 4: Initialize Databases

Run the setup script to create indexes, collections, and constraints:

```bash
cd /Users/simonkelly/SUPER_RAG/doctags_rag
python scripts/setup_databases.py
```

This will:
- Test database connections
- Create Neo4j constraints and indexes
- Create vector indexes in Neo4j
- Create Qdrant collections
- Validate the setup

## Usage

### Basic Example

```python
from src.knowledge_graph.neo4j_manager import Neo4jManager, GraphNode
from src.retrieval.qdrant_manager import QdrantManager, VectorPoint
from src.retrieval.hybrid_retriever import HybridRetriever

# Initialize managers
neo4j = Neo4jManager()
qdrant = QdrantManager()
retriever = HybridRetriever(neo4j_manager=neo4j, qdrant_manager=qdrant)

# Index a document
doc_id = "doc_001"
embedding = [0.1, 0.2, ...]  # Your embedding vector

# Index in Neo4j (graph)
neo4j.create_node(
    labels=["Document"],
    properties={
        "doc_id": doc_id,
        "title": "My Document",
        "text": "Document content...",
        "embedding": embedding
    }
)

# Index in Qdrant (vector)
qdrant.insert_vector(
    vector=embedding,
    metadata={
        "doc_id": doc_id,
        "title": "My Document",
        "text": "Document content..."
    },
    vector_id=doc_id
)

# Search
query_embedding = [0.15, 0.25, ...]  # Query embedding
results, metrics = retriever.search(
    query_vector=query_embedding,
    query_text="What is this about?",
    top_k=5
)

# Process results
for result in results:
    print(f"Score: {result.score:.3f}, Confidence: {result.confidence:.3f}")
    print(f"Content: {result.content}")
    print(f"Source: {result.source}")  # "vector", "graph", or "hybrid"
```

### Running Examples

Run the comprehensive example script:

```bash
python scripts/example_usage.py
```

This demonstrates:
1. Indexing documents in both databases
2. Hybrid search with different strategies
3. Automatic query routing
4. Graph traversal
5. Batch operations
6. Health monitoring

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest tests/test_indexing.py -v

# Run specific test class
pytest tests/test_indexing.py::TestNeo4jManager -v

# Run with coverage
pytest tests/test_indexing.py --cov=src --cov-report=html
```

Test coverage includes:
- Connection management
- CRUD operations
- Search functionality
- Error handling
- Integration scenarios
- Performance tests

## Configuration

Edit `config/config.yaml` to customize:

### Neo4j Settings

```yaml
neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "password"
  database: "doctags"
  max_connection_pool_size: 50
  connection_timeout: 30
```

### Qdrant Settings

```yaml
qdrant:
  host: "localhost"
  port: 6333
  api_key: null  # For cloud deployment
  collection_name: "doctags_vectors"
  vector_size: 1536  # OpenAI embeddings
  distance_metric: "cosine"
```

### Retrieval Settings

```yaml
retrieval:
  hybrid_search:
    enable: true
    vector_weight: 0.7  # 70% weight to vector search
    graph_weight: 0.3   # 30% weight to graph search
  max_results: 10
  rerank: true
  confidence_scoring:
    enable: true
    min_confidence: 0.5
```

## Architecture

### Data Flow

```
Query Input
    ↓
Query Analysis (Type Detection)
    ↓
Query Routing (Strategy Selection)
    ↓
    ├──→ Vector Search (Qdrant)
    │         ↓
    │    Semantic Similarity
    │         ↓
    │    Top-K Results
    │         ↓
    └──→ Graph Search (Neo4j)
              ↓
         Vector + Traversal
              ↓
         Context-Rich Results
              ↓
    Reciprocal Rank Fusion
              ↓
    Confidence Scoring
              ↓
    Deduplication & Reranking
              ↓
    Final Results
```

### Neo4j Schema

```
Nodes:
- Document (doc_id, title, text, embedding, metadata)
- Section (section_id, doc_id, text, embedding)
- Entity (name, type, embedding)

Relationships:
- CONTAINS (Document → Section)
- MENTIONS (Document → Entity)
- REFERENCES (Document → Document)
- RELATED_TO (Entity → Entity)
```

### Qdrant Collections

- `doctags_vectors`: Main document embeddings
- `doctags_chunks`: Chunk-level embeddings
- `doctags_summaries`: RAPTOR summary embeddings
- `doctags_entities`: Entity embeddings

## API Reference

### Neo4jManager

```python
# Connection
manager = Neo4jManager(config)
manager.health_check() -> bool
manager.close()

# Node operations
manager.create_node(labels, properties) -> dict
manager.create_nodes_batch(nodes, batch_size=1000) -> int
manager.get_node_by_id(node_id) -> dict
manager.update_node(node_id, properties, merge=True) -> bool
manager.delete_node(node_id, detach=True) -> bool

# Relationship operations
manager.create_relationship(start_id, end_id, rel_type, properties) -> dict
manager.create_relationships_batch(relationships, batch_size=1000) -> int

# Vector operations
manager.initialize_vector_index(index_name, label, property, dimensions) -> bool
manager.vector_similarity_search(index_name, query_vector, top_k) -> List[SearchResult]

# Graph operations
manager.traverse_graph(start_id, rel_types, direction, max_depth) -> List[dict]
manager.find_shortest_path(start_id, end_id, rel_types) -> dict
manager.pattern_match(pattern, parameters, limit) -> List[dict]

# Utilities
manager.execute_query(query, parameters) -> List[dict]
manager.get_statistics() -> dict
```

### QdrantManager

```python
# Connection
manager = QdrantManager(config)
manager.health_check() -> bool
manager.close()

# Collection operations
manager.create_collection(name, vector_size, distance_metric) -> bool
manager.collection_exists(name) -> bool
manager.delete_collection(name) -> bool

# Vector operations
manager.insert_vector(vector, metadata, vector_id) -> str
manager.insert_vectors_batch(vectors, batch_size=100) -> int
manager.get_vector(vector_id, with_vector=True) -> SearchResult
manager.update_vector(vector_id, vector, metadata) -> bool
manager.delete_vector(vector_id) -> bool

# Search operations
manager.search(query_vector, top_k, filters, score_threshold) -> List[SearchResult]
manager.scroll_collection(limit, offset, filters) -> Tuple[List, offset]

# Utilities
manager.get_collection_info(name) -> dict
manager.get_statistics(name) -> dict
```

### HybridRetriever

```python
# Initialization
retriever = HybridRetriever(
    neo4j_manager=neo4j,
    qdrant_manager=qdrant,
    vector_weight=0.7,
    graph_weight=0.3
)

# Search
results, metrics = retriever.search(
    query_vector=embedding,
    query_text=text,
    top_k=10,
    strategy=SearchStrategy.HYBRID,
    filters={"category": "tech"},
    min_confidence=0.5,
    vector_index_name="document_embeddings",
    collection_name="doctags_vectors"
)

# Query analysis
query_type = retriever.detect_query_type(text) -> QueryType
strategy = retriever.route_query(query_type) -> SearchStrategy

# Result enhancement
enriched = retriever.enrich_with_graph_context(result, max_depth=2)
reranked = retriever.rerank_results(results, query_text, use_diversity=True)

# Utilities
health = retriever.health_check() -> dict
stats = retriever.get_statistics() -> dict
retriever.close()
```

## Performance Tuning

### Neo4j

1. **Memory Configuration** (in docker-compose.yml):
   ```yaml
   - NEO4J_dbms_memory_heap_max__size=2G
   - NEO4J_dbms_memory_pagecache_size=1G
   ```

2. **Connection Pool**:
   ```yaml
   max_connection_pool_size: 50
   ```

3. **Batch Sizes**:
   - Nodes: 1000 per batch
   - Relationships: 1000 per batch

### Qdrant

1. **Batch Insertion**:
   - Default: 100 vectors per batch
   - Adjust based on vector size and memory

2. **Search Parameters**:
   ```python
   search_params = SearchParams(
       hnsw_ef=128,  # Higher = more accurate but slower
       exact=False    # True for exact search
   )
   ```

3. **Collection Optimization**:
   - Use appropriate distance metric
   - Consider quantization for large collections

### Hybrid Retriever

1. **Weights**: Adjust based on your use case
   - Factual queries: Higher vector weight
   - Relationship queries: Higher graph weight

2. **Top-K**: Get 2x results from each source for fusion

3. **Caching**: Consider Redis for frequent queries

## Monitoring

### Health Checks

```python
# Individual checks
neo4j_healthy = neo4j_manager.health_check()
qdrant_healthy = qdrant_manager.health_check()

# Combined check
health = retriever.health_check()
```

### Metrics

```python
# Database statistics
neo4j_stats = neo4j_manager.get_statistics()
qdrant_stats = qdrant_manager.get_statistics()

# Search metrics
results, metrics = retriever.search(...)
print(f"Total time: {metrics.total_time_ms}ms")
print(f"Vector time: {metrics.vector_time_ms}ms")
print(f"Graph time: {metrics.graph_time_ms}ms")
print(f"Fusion time: {metrics.fusion_time_ms}ms")
```

### Prometheus (with monitoring profile)

Access metrics at: http://localhost:9090

Key metrics to monitor:
- Query latency
- Result counts
- Error rates
- Connection pool usage

### Grafana (with monitoring profile)

Access dashboards at: http://localhost:3000

Default credentials: admin/admin

## Troubleshooting

### Connection Issues

**Problem**: "Neo4j connection failed"
```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Check Neo4j logs
docker logs doctags-neo4j

# Test connection
curl http://localhost:7474
```

**Problem**: "Qdrant connection failed"
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Check Qdrant health
curl http://localhost:6333/health

# Check logs
docker logs doctags-qdrant
```

### Performance Issues

**Problem**: Slow searches

1. Check index status:
   ```cypher
   SHOW INDEXES
   ```

2. Verify vector index:
   ```cypher
   SHOW INDEXES WHERE name = 'document_embeddings'
   ```

3. Check Qdrant collection status:
   ```python
   info = qdrant_manager.get_collection_info()
   print(info['optimizer_status'])
   ```

### Memory Issues

**Problem**: Out of memory errors

1. Reduce batch sizes
2. Increase Docker memory limits
3. Adjust Neo4j heap size
4. Consider pagination for large result sets

## Best Practices

### Indexing

1. **Use Batch Operations**: 10-100x faster than individual inserts
2. **Consistent IDs**: Use same ID in both databases for easy linking
3. **Metadata**: Store rich metadata for filtering
4. **Embeddings**: Use consistent embedding model

### Searching

1. **Query Routing**: Let the system auto-route queries
2. **Filters**: Use metadata filters to narrow results
3. **Top-K**: Start with 10, adjust based on precision/recall
4. **Confidence Threshold**: Filter low-confidence results

### Graph Modeling

1. **Node Labels**: Use specific labels (Document, Entity, Section)
2. **Relationships**: Add properties (weight, confidence, timestamp)
3. **Indexes**: Create indexes on frequently queried properties
4. **Constraints**: Use constraints for data integrity

### Error Handling

1. **Retry Logic**: Already implemented with exponential backoff
2. **Transactions**: Batch operations are transactional
3. **Health Checks**: Monitor before critical operations
4. **Logging**: Use loguru for comprehensive logging

## Production Checklist

- [ ] Change default passwords in docker-compose.yml
- [ ] Set up proper authentication and authorization
- [ ] Configure backup strategies for both databases
- [ ] Set up monitoring and alerting (Prometheus + Grafana)
- [ ] Configure proper logging and log aggregation
- [ ] Implement rate limiting for API endpoints
- [ ] Set up SSL/TLS for database connections
- [ ] Test disaster recovery procedures
- [ ] Document runbooks for common operations
- [ ] Set up CI/CD for automated testing

## Support and Resources

### Neo4j
- Documentation: https://neo4j.com/docs/
- Driver Manual: https://neo4j.com/docs/python-manual/current/
- APOC Procedures: https://neo4j.com/labs/apoc/

### Qdrant
- Documentation: https://qdrant.tech/documentation/
- Python Client: https://github.com/qdrant/qdrant-client
- Cloud Service: https://cloud.qdrant.io/

### Code Location

All code is located in `/Users/simonkelly/SUPER_RAG/doctags_rag/`:
- Neo4j Manager: `src/knowledge_graph/neo4j_manager.py`
- Qdrant Manager: `src/retrieval/qdrant_manager.py`
- Hybrid Retriever: `src/retrieval/hybrid_retriever.py`
- Tests: `tests/test_indexing.py`
- Scripts: `scripts/`
- Config: `config/config.yaml`

## License

This code is part of the Contextprime system.
