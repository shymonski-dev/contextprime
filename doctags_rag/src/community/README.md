# Community Detection Module

A comprehensive community detection system for the Contextprime system, enabling detection of entity communities and document clusters with global insights similar to Microsoft's GraphRAG approach.

## Overview

This module provides end-to-end community detection and analysis capabilities for knowledge graphs, including:

- **Multiple Detection Algorithms**: Louvain, Leiden, Label Propagation, Spectral Clustering
- **Graph Analysis**: Centrality measures, community quality metrics, entity importance
- **LLM-based Summarization**: Automatic generation of community descriptions
- **Document Clustering**: Entity-based, semantic, temporal, and hierarchical clustering
- **Cross-Document Analysis**: Co-occurrence patterns, theme evolution, knowledge synthesis
- **Global Query Handling**: Answer queries requiring global understanding
- **Persistence**: Store and retrieve community structures in Neo4j
- **Visualization**: Static and interactive visualizations

## Architecture

```
community/
├── community_detector.py       # Multi-algorithm community detection
├── graph_analyzer.py          # Graph metrics and analysis
├── community_summarizer.py    # LLM-based summarization
├── document_clusterer.py      # Document clustering methods
├── cross_document_analyzer.py # Cross-document analysis
├── global_query_handler.py    # Global query answering
├── community_storage.py       # Neo4j persistence
├── community_pipeline.py      # Orchestration pipeline
└── community_visualizer.py    # Visualization and export
```

## Key Components

### 1. CommunityDetector

Detects communities using multiple algorithms with automatic selection:

```python
from src.community import CommunityDetector, CommunityAlgorithm

detector = CommunityDetector(
    algorithm=CommunityAlgorithm.AUTO,  # Auto-select best algorithm
    resolution=1.0,
    min_community_size=3
)

result = detector.detect_communities(graph)
print(f"Found {result.num_communities} communities")
print(f"Modularity: {result.modularity:.3f}")
```

**Supported Algorithms:**
- **Louvain**: Fast modularity optimization with hierarchical structure
- **Leiden**: Improved Louvain with better quality guarantees
- **Label Propagation**: Fast, memory-efficient approach
- **Spectral Clustering**: For well-separated communities
- **AUTO**: Automatically selects best algorithm based on graph properties

### 2. GraphAnalyzer

Computes comprehensive graph and community metrics:

```python
from src.community import GraphAnalyzer

analyzer = GraphAnalyzer()

# Graph-level metrics
graph_metrics = analyzer.analyze_graph(graph)
print(f"Density: {graph_metrics.density:.3f}")
print(f"Average degree: {graph_metrics.avg_degree:.2f}")

# Node-level metrics
node_metrics = analyzer.compute_node_metrics(graph)
for node_id, metrics in node_metrics.items():
    print(f"{node_id}: PageRank={metrics.pagerank:.4f}")

# Community quality
community_metrics = analyzer.analyze_communities(graph, community_result)
for comm_id, metrics in community_metrics.items():
    print(f"Community {comm_id}: conductance={metrics.conductance:.3f}")
```

**Metrics Computed:**
- Node centrality (degree, betweenness, closeness, eigenvector, PageRank)
- Graph statistics (density, clustering, diameter, path length)
- Community quality (modularity, conductance, coverage)
- Hub and authority scores

### 3. CommunitySummarizer

Generates human-readable summaries using LLMs:

```python
from src.community import CommunitySummarizer

summarizer = CommunitySummarizer(api_key="your-openai-key")

# Summarize all communities
community_summaries = summarizer.summarize_all_communities(
    graph,
    community_result,
    include_detailed=True
)

# Generate global summary
global_summary = summarizer.generate_global_summary(
    graph,
    community_result,
    community_summaries
)

print(f"Main themes: {global_summary.main_themes}")
print(f"Key insights: {global_summary.key_insights}")
```

**Features:**
- Brief and detailed summaries for each community
- Automatic theme and topic extraction
- Representative entity selection
- Global cross-community insights
- Fallback to rule-based summarization without API key

### 4. DocumentClusterer

Clusters documents using various methods:

```python
from src.community import DocumentClusterer, ClusteringMethod

clusterer = DocumentClusterer()

# Entity-based clustering
doc_entities = {
    "doc_1": {"entity_A", "entity_B"},
    "doc_2": {"entity_B", "entity_C"}
}
result = clusterer.cluster_by_entities(doc_entities)

# Semantic clustering
doc_embeddings = {...}  # Document embeddings
result = clusterer.cluster_by_semantics(doc_embeddings)

# Hybrid clustering
result = clusterer.hybrid_cluster(doc_embeddings, doc_entities)
```

**Methods:**
- Entity-based: Cluster by shared entities (Jaccard similarity)
- Semantic: K-means on document embeddings
- Temporal: Time-based grouping
- Hierarchical: Agglomerative clustering
- Hybrid: Combines multiple signals

### 5. CrossDocumentAnalyzer

Analyzes patterns across multiple documents:

```python
from src.community import CrossDocumentAnalyzer

analyzer = CrossDocumentAnalyzer()

# Entity co-occurrence
patterns = analyzer.analyze_entity_cooccurrence(doc_entities)
for pattern in patterns:
    print(f"{pattern.entity_pair}: {pattern.cooccurrence_count} times")

# Build co-occurrence graph
graph = analyzer.build_cooccurrence_graph(doc_entities)

# Document similarity
similarity = analyzer.compute_document_similarity(
    "doc_1", "doc_2",
    doc_embeddings,
    doc_entities
)
```

**Capabilities:**
- Entity co-occurrence analysis
- Co-occurrence graph construction
- Multi-faceted document similarity
- Contradiction detection
- Consensus identification
- Knowledge aggregation

### 6. GlobalQueryHandler

Answers queries requiring global understanding:

```python
from src.community import GlobalQueryHandler

handler = GlobalQueryHandler(api_key="your-openai-key")

response = handler.answer_query(
    "What are the main themes?",
    graph,
    community_result,
    community_summaries,
    global_summary
)

print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence:.2f}")
```

**Query Types:**
- Theme queries: "What are the main themes?"
- Community queries: "How many communities exist?"
- Relationship queries: "How are topics related?"
- Structure queries: "How is the graph organized?"

### 7. CommunityStorage

Persists community detection results in Neo4j:

```python
from src.community import CommunityStorage

storage = CommunityStorage()

# Store communities
version = storage.store_communities(
    community_result,
    community_summaries,
    version="v1.0"
)

# Load communities
data = storage.load_communities(version="v1.0")

# List all versions
versions = storage.list_versions()
```

**Features:**
- Version control for community structures
- Membership relationships in Neo4j
- Metadata storage
- Efficient querying

### 8. CommunityPipeline

Orchestrates the complete workflow:

```python
from src.community import CommunityPipeline

pipeline = CommunityPipeline(
    algorithm=CommunityAlgorithm.AUTO,
    api_key="your-openai-key",
    enable_storage=True,
    enable_summaries=True
)

# Run complete pipeline
result = pipeline.run(
    graph=graph,
    load_from_neo4j=False,
    store_results=True
)

# Answer global queries
response = pipeline.answer_global_query(
    "What are the main themes?",
    graph=graph
)
```

**Pipeline Steps:**
1. Graph loading/construction
2. Community detection
3. Quality assessment
4. Summary generation
5. Storage in Neo4j
6. Query preparation

### 9. CommunityVisualizer

Creates visualizations and exports:

```python
from src.community import CommunityVisualizer

visualizer = CommunityVisualizer()

# Static visualization
visualizer.visualize_communities(
    graph,
    community_result,
    "communities.png",
    layout="spring"
)

# Interactive HTML
visualizer.create_interactive_visualization(
    graph,
    community_result,
    "communities.html"
)

# Export formats
visualizer.export_to_graphml(graph, community_result, "graph.graphml")
visualizer.export_to_d3_json(graph, community_result, "graph.json")
```

**Capabilities:**
- Static plots (matplotlib)
- Interactive visualizations (pyvis)
- Multiple export formats (GraphML, GEXF, D3 JSON)
- Community size distributions
- Algorithm comparisons

## Usage Examples

### Basic Community Detection

```python
import networkx as nx
from src.community import CommunityDetector, GraphAnalyzer

# Create or load graph
graph = nx.karate_club_graph()

# Detect communities
detector = CommunityDetector()
result = detector.detect_communities(graph)

# Analyze
analyzer = GraphAnalyzer()
metrics = analyzer.analyze_graph(graph)
community_metrics = analyzer.analyze_communities(graph, result)

print(f"Communities: {result.num_communities}")
print(f"Modularity: {result.modularity:.3f}")
```

### Complete Pipeline

```python
from src.community import CommunityPipeline

pipeline = CommunityPipeline()
result = pipeline.run(graph=my_graph)

# Answer queries
response = pipeline.answer_global_query(
    "What are the main themes?",
    graph=my_graph
)
print(response.answer)
```

### Document Clustering

```python
from src.community import DocumentClusterer

doc_entities = {
    "paper_1": {"ML", "Deep Learning", "Neural Networks"},
    "paper_2": {"ML", "Neural Networks", "CNN"},
    "paper_3": {"NLP", "Text Classification", "BERT"}
}

clusterer = DocumentClusterer()
result = clusterer.cluster_by_entities(doc_entities)

for cluster_id, cluster in result.clusters.items():
    print(f"Cluster {cluster_id}:")
    print(f"  Documents: {cluster.document_ids}")
    print(f"  Shared entities: {cluster.shared_entities}")
```

## Installation

Install required dependencies:

```bash
pip install networkx scikit-learn python-louvain leidenalg python-igraph \
    matplotlib pyvis openai numpy loguru
```

## Testing

Run comprehensive tests:

```bash
# All tests
pytest tests/test_community.py -v

# Specific test class
pytest tests/test_community.py::TestCommunityDetector -v

# With coverage
pytest tests/test_community.py --cov=src.community --cov-report=html
```

## Demo

Run the demonstration script:

```bash
python scripts/demo_community.py
```

This will:
1. Create a sample knowledge graph
2. Detect communities using multiple algorithms
3. Generate summaries and visualizations
4. Demonstrate global query answering
5. Compare algorithm performance
6. Export to various formats

## Performance Considerations

### Scalability

- **Small graphs (<1K nodes)**: All algorithms work well
- **Medium graphs (1K-10K nodes)**: Louvain/Leiden recommended
- **Large graphs (>10K nodes)**: Label Propagation for speed

### Optimization Tips

1. Use `algorithm=CommunityAlgorithm.AUTO` for automatic selection
2. Set `min_community_size` to filter small communities
3. Disable detailed summaries for large graphs
4. Use sampling for very large graph metrics
5. Cache results using `CommunityStorage`

## Integration with Contextprime

The community detection module integrates with:

- **Knowledge Graph**: Load entities and relationships from Neo4j
- **Document Processing**: Cluster processed documents
- **Retrieval**: Use community structure for better retrieval
- **Summarization**: Leverage community summaries for context
- **Query Routing**: Route queries based on community structure

## Advanced Features

### Hierarchical Communities

Louvain algorithm provides hierarchical community structure:

```python
result = detector.detect_communities(graph, algorithm=CommunityAlgorithm.LOUVAIN)
for level, communities in result.hierarchical_levels.items():
    print(f"Level {level}: {len(communities)} communities")
```

### Algorithm Comparison

Compare multiple algorithms:

```python
results = detector.compare_algorithms(graph)
for algo, result in results.items():
    print(f"{algo}: modularity={result.modularity:.3f}")
```

### Incremental Updates

Update communities when graph changes:

```python
result = pipeline.run_incremental_update(
    new_graph,
    previous_version="v1.0"
)
```

## References

- [Louvain Method](https://en.wikipedia.org/wiki/Louvain_method)
- [Leiden Algorithm](https://www.nature.com/articles/s41598-019-41695-z)
- [Microsoft GraphRAG](https://www.microsoft.com/en-us/research/project/graphrag/)
- [NetworkX Documentation](https://networkx.org/)

## Statistics

- **Total Lines of Code**: 4,417
- **Test Lines**: 618
- **Demo Lines**: 483
- **Components**: 9 main modules
- **Algorithms**: 4 community detection algorithms
- **Export Formats**: 4 (PNG, HTML, GraphML, JSON)

## License

Part of the Contextprime system.
