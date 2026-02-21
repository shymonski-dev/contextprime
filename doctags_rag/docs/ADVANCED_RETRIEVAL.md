# Advanced Retrieval Features

This document describes the advanced retrieval features implemented in the Contextprime system.

## Overview

The advanced retrieval system implements state-of-the-art techniques for improving retrieval quality, including:

1. **Confidence Scoring (CRAG-style)** - Multi-signal assessment of retrieval quality
2. **Query Routing** - Intelligent routing based on query type and complexity
3. **Iterative Refinement** - Self-reflection and multi-round retrieval
4. **Result Reranking** - Advanced reranking with cross-encoder and feature-based scoring
5. **Query Expansion** - Synonym, semantic, and contextual query expansion
6. **Caching** - Multi-level caching with semantic similarity matching
7. **Advanced Pipeline** - Orchestration of all features

## Components

### 1. Confidence Scorer (`confidence_scorer.py`)

CRAG-style confidence scoring system that assesses retrieval quality using multiple signals.

#### Features
- Multi-signal confidence assessment
- Three confidence levels: CORRECT, AMBIGUOUS, INCORRECT
- Corrective action recommendations
- Entity-aware scoring
- Graph connectivity analysis

#### Signals
- **Semantic Similarity**: Vector search score
- **Keyword Overlap**: Query-document keyword matching
- **Entity Match**: Named entity alignment
- **Graph Connectivity**: Relationship strength in knowledge graph
- **Length Appropriateness**: Content length quality
- **Source Reliability**: Source authority indicators

#### Usage
```python
from src.retrieval import ConfidenceScorer

scorer = ConfidenceScorer()

# Score a single result
score = scorer.score_result(
    query="What is machine learning?",
    result_content="Machine learning is...",
    vector_score=0.85,
    metadata={"source": "textbook"}
)

print(f"Confidence: {score.overall_score:.2f}")
print(f"Level: {score.level.value}")
print(f"Action: {score.corrective_action.value}")

# Score multiple results
scores = scorer.score_results_batch(query, results)

# Aggregate statistics
stats = scorer.aggregate_confidence(scores)
```

### 2. Query Router (`query_router.py`)

Intelligent query routing system with learning capabilities.

#### Features
- Multi-dimensional query classification
- 9 query types (factual, relationship, comparison, etc.)
- 3 complexity levels (simple, moderate, complex)
- Performance-based learning
- Adaptive routing

#### Query Types
- **FACTUAL**: Simple fact retrieval
- **DEFINITION**: Definition queries
- **RELATIONSHIP**: Relationship/connection queries
- **COMPARISON**: Comparison between entities
- **ANALYTICAL**: Analysis or reasoning
- **MULTI_HOP**: Multi-step reasoning
- **PROCEDURAL**: How-to queries
- **TEMPORAL**: Time-based queries
- **AGGREGATION**: Statistical queries

#### Usage
```python
from src.retrieval import QueryRouter

router = QueryRouter(enable_learning=True)

# Analyze query
analysis = router.analyze_query("How is Python related to Java?")
print(f"Type: {analysis.query_type.value}")
print(f"Complexity: {analysis.complexity.value}")

# Route query
strategy, analysis = router.route_query(query)

# Record performance for learning
router.record_performance(
    query=query,
    strategy=strategy,
    query_type=analysis.query_type,
    success=True,
    confidence=0.8
)
```

### 3. Iterative Refiner (`iterative_refiner.py`)

Self-reflection and multi-round retrieval system.

#### Features
- Multi-round retrieval with refinement
- Information gap analysis
- Query refinement generation
- Result deduplication
- Convergence detection
- Provenance tracking

#### Usage
```python
from src.retrieval import IterativeRefiner

refiner = IterativeRefiner(
    max_iterations=3,
    min_confidence_threshold=0.7
)

def retrieval_func(query, context=None):
    # Your retrieval implementation
    return results

refined_results, steps = refiner.refine_retrieval(
    original_query=query,
    initial_results=initial_results,
    retrieval_func=retrieval_func
)

# Get summary
summary = refiner.get_refinement_summary(steps)
print(f"Iterations: {summary['total_iterations']}")
print(f"Final confidence: {summary['final_confidence']:.2f}")
```

### 4. Reranker (`reranker.py`)

Advanced result reranking with multiple strategies.

#### Features
- Cross-encoder reranking (optional)
- Feature-based scoring
- Temporal relevance
- Source authority
- Entity coverage
- Diversity optimization (MMR)

#### Reranking Features
- **Semantic Score**: Original similarity score
- **Cross-Encoder**: Pairwise relevance (if enabled)
- **Recency**: Temporal relevance
- **Authority**: Source credibility
- **Entity Coverage**: Query entity coverage
- **Diversity**: Result diversity
- **Length**: Content length quality

#### Usage
```python
from src.retrieval import Reranker

reranker = Reranker(
    enable_cross_encoder=True,
    diversity_lambda=0.5
)

reranked = reranker.rerank(
    query=query,
    results=results,
    top_k=10,
    enable_diversity=True
)

# Explain ranking
explanation = reranker.explain_ranking(reranked[0])
print(f"Top features: {explanation['top_features']}")
```

### 5. Query Expander (`query_expansion.py`)

Comprehensive query expansion system.

#### Features
- Synonym expansion (WordNet)
- Entity expansion from knowledge graph
- Semantic expansion (embeddings)
- Contextual expansion (session history)
- Domain-specific expansions

#### Expansion Strategies
- **Conservative**: Only most relevant synonyms
- **Comprehensive**: Balanced expansion
- **Aggressive**: All available expansions

#### Usage
```python
from src.retrieval import QueryExpander

expander = QueryExpander(
    enable_wordnet=True,
    enable_semantic=True
)

# Expand query
expanded = expander.expand_query(
    query="What is ML?",
    strategy="comprehensive",
    graph_entities=["machine learning", "AI"]
)

print(f"Original: {expanded.original_query}")
print(f"Expanded: {expanded.expanded_query}")
print(f"Synonyms: {expanded.synonyms}")

# Multiple strategies
expansions = expander.expand_multi_strategy(query)

# Add domain expansion
expander.add_domain_expansion("ml", ["machine learning", "AI"])
```

### 6. Cache Manager (`cache_manager.py`)

Multi-level caching with semantic matching.

#### Features
- Query result caching with semantic similarity
- Embedding caching
- LRU eviction
- TTL support
- Persistent disk cache
- Thread-safe operations

#### Cache Types
- **Query Cache**: Semantic similarity-based query matching
- **Embedding Cache**: Vector embedding caching
- **Result Cache**: General result caching

#### Usage
```python
from src.retrieval import CacheManager
import numpy as np

cache_manager = CacheManager(
    cache_dir=Path("cache"),
    query_ttl=3600
)

# Cache query results
query_vector = np.random.rand(384)
cache_manager.cache_query_results(query, query_vector, results)

# Retrieve (supports semantic matching)
cached = cache_manager.get_query_results(query, query_vector)

# Cache embeddings
cache_manager.cache_embedding(text, embedding, persist=True)

# Statistics
stats = cache_manager.get_statistics()
print(f"Hit rate: {stats['query_cache']['hit_rate']:.2%}")
```

### 7. Advanced Pipeline (`advanced_pipeline.py`)

Orchestrates all advanced features in a complete pipeline.

#### Pipeline Stages
1. Query analysis and routing
2. Query expansion (optional)
3. Initial retrieval
4. Confidence scoring
5. Iterative refinement (if needed)
6. Result reranking
7. Final aggregation

#### Usage
```python
from src.retrieval import (
    AdvancedRetrievalPipeline,
    PipelineConfig,
    HybridRetriever
)

# Initialize
hybrid_retriever = HybridRetriever()

config = PipelineConfig(
    enable_query_expansion=True,
    enable_iterative_refinement=True,
    enable_reranking=True,
    enable_caching=True,
    max_refinement_iterations=3
)

pipeline = AdvancedRetrievalPipeline(
    hybrid_retriever=hybrid_retriever,
    config=config
)

# Retrieve
result = pipeline.retrieve(
    query="What is deep learning?",
    query_vector=query_embedding,
    top_k=10
)

# Results
print(f"Results: {len(result.results)}")
print(f"Avg Confidence: {result.metrics.avg_confidence:.2f}")
print(f"Total Time: {result.metrics.total_time_ms:.2f}ms")

# Explain result
explanation = pipeline.explain_result(result, result_index=0)
print(explanation)
```

## Configuration

### Pipeline Configuration

```python
config = PipelineConfig(
    # Feature toggles
    enable_query_expansion=True,
    enable_iterative_refinement=True,
    enable_reranking=True,
    enable_caching=True,
    enable_confidence_scoring=True,

    # Parameters
    max_refinement_iterations=3,
    min_confidence_threshold=0.7,
    use_cross_encoder=True,
    cache_ttl=3600,
    top_k=10,
    min_results=3
)
```

### Component Configuration

Each component can be configured independently:

```python
# Confidence scorer
scorer = ConfidenceScorer()
scorer.thresholds.correct_threshold = 0.8
scorer.thresholds.semantic_weight = 0.35

# Query router
router = QueryRouter(
    enable_learning=True,
    performance_file=Path("routing_perf.json")
)

# Reranker
reranker = Reranker(
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    enable_cross_encoder=True,
    diversity_lambda=0.5
)
```

## Performance Metrics

The pipeline provides comprehensive metrics:

```python
result = pipeline.retrieve(query)

metrics = result.metrics
print(f"Total Time: {metrics.total_time_ms:.2f}ms")
print(f"Query Analysis: {metrics.query_analysis_time_ms:.2f}ms")
print(f"Expansion: {metrics.expansion_time_ms:.2f}ms")
print(f"Retrieval: {metrics.retrieval_time_ms:.2f}ms")
print(f"Confidence Scoring: {metrics.confidence_scoring_time_ms:.2f}ms")
print(f"Refinement: {metrics.refinement_time_ms:.2f}ms")
print(f"Reranking: {metrics.reranking_time_ms:.2f}ms")

print(f"\nResults: {metrics.results_count}")
print(f"Avg Confidence: {metrics.avg_confidence:.2f}")
print(f"Refinement Iterations: {metrics.refinement_iterations}")
print(f"Cache Hit: {metrics.cache_hit}")
```

## Demo

Run the interactive demo to see all features in action:

```bash
python scripts/demo_advanced_retrieval.py
```

The demo includes:
1. Confidence scoring demonstration
2. Query routing examples
3. Query expansion strategies
4. Reranking comparison
5. Caching performance
6. Iterative refinement
7. Performance comparison

## Testing

Run the comprehensive test suite:

```bash
# All tests
pytest tests/test_advanced_retrieval.py -v

# Specific component
pytest tests/test_advanced_retrieval.py::TestConfidenceScorer -v

# With coverage
pytest tests/test_advanced_retrieval.py --cov=src.retrieval
```

## Dependencies

Required packages:
- `sentence-transformers>=2.2.2` - For cross-encoder reranking
- `nltk>=3.8.1` - For WordNet synonym expansion
- `spacy>=3.7.2` - For entity extraction and NLP
- `diskcache>=5.6.3` - For persistent caching
- `numpy>=1.24.3` - For numerical operations
- `scikit-learn>=1.3.2` - For feature combination

Optional models:
```bash
# spaCy model
python -m spacy download en_core_web_sm

# NLTK data
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Best Practices

### 1. Query Routing
- Enable learning mode for production systems
- Persist performance metrics to improve routing over time
- Review routing statistics regularly

### 2. Confidence Scoring
- Set appropriate thresholds for your domain
- Monitor confidence distributions
- Use corrective actions to improve results

### 3. Iterative Refinement
- Limit iterations to prevent excessive latency
- Set appropriate convergence thresholds
- Monitor refinement patterns

### 4. Reranking
- Enable cross-encoder for best quality (adds latency)
- Adjust feature weights based on your use case
- Use diversity optimization for varied results

### 5. Caching
- Enable disk cache for production
- Set appropriate TTLs based on data freshness
- Monitor cache hit rates
- Use semantic similarity matching for flexibility

### 6. Pipeline Configuration
- Start with all features enabled
- Disable features based on performance requirements
- A/B test configurations
- Monitor metrics continuously

## Performance Considerations

### Latency
- Query expansion: ~10ms
- Confidence scoring: ~20ms per result
- Reranking (no cross-encoder): ~30ms
- Reranking (with cross-encoder): ~200-500ms
- Iterative refinement: Adds 1-3 retrieval rounds

### Optimization Tips
1. **Disable cross-encoder** if latency is critical
2. **Use caching** aggressively for repeated queries
3. **Limit refinement iterations** to 2-3
4. **Batch operations** when possible
5. **Adjust top_k** based on reranking needs

## Future Enhancements

Planned improvements:
- [ ] Learned feature weights for reranking
- [ ] Query understanding with LLMs
- [ ] Federated retrieval across multiple sources
- [ ] Active learning for routing
- [ ] Multi-modal retrieval support
- [ ] Real-time performance monitoring dashboard

## Troubleshooting

### Common Issues

**Issue**: spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

**Issue**: NLTK data not available
```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

**Issue**: Cross-encoder slow
```python
# Disable cross-encoder
reranker = Reranker(enable_cross_encoder=False)
```

**Issue**: Cache not persisting
```python
# Check cache directory permissions
cache_manager = CacheManager(
    cache_dir=Path("cache"),
    enable_disk_cache=True
)
```

## Support

For issues, questions, or contributions, please refer to the main project documentation.
