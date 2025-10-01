# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

A production-ready implementation of hierarchical document understanding and multi-level retrieval for the DocTags RAG system.

## Overview

RAPTOR creates hierarchical tree structures from documents through recursive clustering and summarization, enabling:

- **Multi-level retrieval** - Access information at different granularities
- **Contextual understanding** - Maintain document structure and relationships
- **Efficient search** - Navigate tree structure for targeted retrieval
- **Scalable summarization** - Handle documents of varying lengths

## Architecture

```
Document → Chunks → Embeddings → Clustering → Summaries → Tree
                                                              ↓
Query → Embedding → Hierarchical Search → Ranked Results
```

### Components

1. **ClusterManager** (`cluster_manager.py`)
   - UMAP + HDBSCAN for density-based clustering
   - K-means for centroid-based clustering
   - Semantic similarity clustering
   - Adaptive method selection

2. **SummaryGenerator** (`summary_generator.py`)
   - LLM-based abstractive summarization
   - Multi-level summary generation (leaf, intermediate, root)
   - Quality assessment and fact extraction
   - Fallback extractive summarization

3. **TreeBuilder** (`tree_builder.py`)
   - Bottom-up tree construction
   - Recursive clustering and summarization
   - Balanced tree structure maintenance
   - Parent-child-sibling relationships

4. **TreeStorage** (`tree_storage.py`)
   - Neo4j for tree structure
   - Qdrant for embeddings (per-level collections)
   - Metadata and versioning
   - Efficient load/save operations

5. **HierarchicalRetriever** (`hierarchical_retriever.py`)
   - Top-down traversal
   - Bottom-up aggregation
   - Mid-level balanced retrieval
   - Adaptive strategy selection

6. **RAPTORPipeline** (`raptor_pipeline.py`)
   - End-to-end orchestration
   - Batch processing
   - Incremental updates
   - Quality validation

7. **TreeVisualizer** (`tree_visualizer.py`)
   - ASCII tree rendering
   - HTML interactive visualization
   - GraphML export for Neo4j
   - Statistics and analysis

## Installation

```bash
pip install umap-learn hdbscan anthropic
```

All other dependencies are included in the main `requirements.txt`.

## Quick Start

### 1. Build a Tree

```python
from src.summarization import RAPTORPipeline, PipelineConfig
from src.processing.doctags_processor import DocTagsDocument
from sentence_transformers import SentenceTransformer

# Configure pipeline
config = PipelineConfig(
    chunk_size=1000,
    max_tree_depth=4,
    target_branching_factor=5
)

# Initialize
embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
pipeline = RAPTORPipeline(
    config=config,
    embeddings_model=embeddings_model,
    neo4j_manager=neo4j_manager,
    qdrant_manager=qdrant_manager,
    api_key=openai_api_key
)

# Process document
result = pipeline.process_document(doc, save_tree=True)
print(f"Tree built: {result.tree_id}")
print(f"Nodes: {result.stats.total_nodes}")
print(f"Depth: {result.stats.max_depth}")
```

### 2. Query the Tree

```python
# Query with adaptive strategy
results = pipeline.query(
    query_text="What are the main findings?",
    tree_id=result.tree_id,
    return_context=True
)

for r in results:
    print(f"Score: {r['score']:.3f}, Level: {r['level']}")
    print(f"Content: {r['content'][:200]}")
    print(f"Context: {r['context']}")
```

### 3. Visualize the Tree

```python
from src.summarization import TreeVisualizer

visualizer = TreeVisualizer()

# ASCII visualization
ascii_tree = visualizer.visualize_ascii(
    root=result.root,
    all_nodes=result.all_nodes,
    show_content=True
)
print(ascii_tree)

# Export to HTML
visualizer.export_to_html(
    root=result.root,
    all_nodes=result.all_nodes,
    output_path="tree.html"
)
```

## Retrieval Strategies

### Top-Down
Starts from root, traverses down following high-similarity paths.
- **Best for**: Broad queries, overview understanding
- **Example**: "What is this document about?"

### Bottom-Up
Starts from leaves, aggregates relevant ancestors.
- **Best for**: Specific queries, detail-focused searches
- **Example**: "What is the exact definition of X?"

### Mid-Level
Focuses on intermediate levels for balanced detail.
- **Best for**: General queries needing moderate detail
- **Example**: "How does process X work?"

### Adaptive
Automatically selects strategy based on query characteristics.
- **Best for**: Mixed query types, production systems
- **Example**: Any query

## Configuration

### Pipeline Configuration

```python
config = PipelineConfig(
    # Chunking
    chunk_size=1000,
    chunk_overlap=200,

    # Clustering
    clustering_method=ClusteringMethod.AUTO,
    min_cluster_size=3,
    max_cluster_size=50,

    # Tree building
    max_tree_depth=5,
    target_branching_factor=5,

    # LLM
    llm_provider="openai",
    llm_model="gpt-3.5-turbo",
    llm_temperature=0.1,

    # Retrieval
    retrieval_strategy=RetrievalStrategy.ADAPTIVE,
    top_k=10
)
```

### Clustering Methods

- `UMAP_HDBSCAN`: Density-based, good for varied cluster sizes
- `KMEANS`: Fast, good for balanced clusters
- `HIERARCHICAL`: Good for small datasets
- `SEMANTIC`: Similarity-threshold based
- `AUTO`: Automatic selection based on data

## Advanced Usage

### Incremental Updates

```python
# Add new chunks to existing tree
new_chunks = chunker.chunk_document(new_doc)

pipeline.incremental_update(
    tree_id=tree_id,
    new_chunks=new_chunks
)
```

### Strategy Comparison

```python
# Compare all retrieval strategies
comparison = pipeline.compare_retrieval_strategies(
    query_text="How does X work?",
    tree_id=tree_id
)

for strategy, results in comparison.items():
    print(f"{strategy}: {len(results)} results")
```

### Tree Validation

```python
# Validate tree integrity
validation = pipeline.validate_tree(tree_id)

if validation['valid']:
    print("Tree is valid")
else:
    print(f"Issues found: {validation['issues']}")
```

## Performance Optimization

### 1. Batch Processing

```python
# Process multiple documents
results = pipeline.process_documents_batch(
    documents=doc_list,
    save_trees=True
)
```

### 2. Embedding Caching

```python
# Use cached embeddings
from diskcache import Cache

cache = Cache('/tmp/embedding_cache')

@cache.memoize()
def get_embeddings(texts):
    return embeddings_model.encode(texts)
```

### 3. Parallel Clustering

```python
# Configure for parallel processing
cluster_manager = ClusterManager(
    method=ClusteringMethod.KMEANS,  # Parallelizable
    min_cluster_size=3
)
```

## Tree Storage

### Neo4j Structure

```cypher
// Nodes
(:RAPTORNode {
    node_id: string,
    tree_id: string,
    content: string,
    level: int,
    is_leaf: boolean,
    metadata: json
})

// Relationships
(:RAPTORNode)-[:HAS_PARENT]->(:RAPTORNode)
(:RAPTORNode)-[:SIBLING_OF]->(:RAPTORNode)
```

### Qdrant Collections

- Separate collection per tree level
- Format: `raptor_L{level}`
- Enables efficient level-specific search

## Evaluation Metrics

```python
# Get tree statistics
summary = pipeline.get_tree_summary(tree_id)
print(f"Nodes: {summary['stats']['total_nodes']}")
print(f"Avg branching: {summary['stats']['avg_branching_factor']:.2f}")

# Validate retrieval quality
results = pipeline.query(query, tree_id)
avg_score = np.mean([r['score'] for r in results])
print(f"Average relevance: {avg_score:.3f}")
```

## Best Practices

1. **Chunk Size**: 500-1500 characters depending on content density
2. **Tree Depth**: 3-5 levels for most documents
3. **Branching Factor**: 3-7 children per parent
4. **Min Cluster Size**: 2-5 depending on document size
5. **Summary Quality**: Monitor quality scores, refine prompts

## Troubleshooting

### Issue: Too many small clusters
**Solution**: Increase `min_cluster_size` or use `merge_small_clusters()`

### Issue: Tree too deep
**Solution**: Reduce `max_tree_depth` or increase `target_branching_factor`

### Issue: Poor summary quality
**Solution**:
- Provide better context in prompts
- Use more powerful LLM model
- Adjust temperature parameter

### Issue: Slow retrieval
**Solution**:
- Use level-specific retrieval
- Enable Qdrant indexing
- Reduce `top_k` parameter

## Testing

```bash
# Run all tests
pytest tests/test_summarization.py -v

# Run specific test
pytest tests/test_summarization.py::TestTreeBuilder -v

# Run with coverage
pytest tests/test_summarization.py --cov=src/summarization
```

## Demo

```bash
# Run complete demo
python scripts/demo_raptor.py

# The demo will:
# 1. Build a tree from sample document
# 2. Visualize tree structure
# 3. Demonstrate multi-level retrieval
# 4. Compare retrieval strategies
# 5. Show performance metrics
```

## Integration with DocTags RAG

The RAPTOR system integrates seamlessly with existing DocTags components:

```python
from src.processing.doctags_processor import DocTagsProcessor
from src.retrieval.hybrid_retriever import HybridRetriever

# Process document with DocTags
processor = DocTagsProcessor()
doc = processor.process_pdf("document.pdf")

# Build RAPTOR tree
result = raptor_pipeline.process_document(doc)

# Use with hybrid retrieval
hybrid_retriever = HybridRetriever(
    vector_weight=0.5,
    graph_weight=0.2,
    raptor_weight=0.3  # New: hierarchical retrieval
)
```

## References

- RAPTOR Paper: [Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
- UMAP: [Uniform Manifold Approximation and Projection](https://umap-learn.readthedocs.io/)
- HDBSCAN: [Hierarchical Density-Based Clustering](https://hdbscan.readthedocs.io/)

## License

Same as DocTags RAG system.

## Contributing

Contributions welcome! Please ensure:
- Type hints throughout
- Comprehensive docstrings
- Test coverage >80%
- Follows existing code style
