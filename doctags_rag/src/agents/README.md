# Agentic Feedback Loop System

A comprehensive multi-agent system with reinforcement learning capabilities for autonomous RAG improvement.

## Overview

This module implements a complete agentic system that enables:

- **Autonomous Query Processing**: Multi-agent coordination for complex queries
- **Self-Improvement**: Reinforcement learning for continuous optimization
- **Adaptive Strategy Selection**: Dynamic query routing based on learned patterns
- **Quality Assessment**: Automated evaluation with feedback generation
- **Memory Systems**: Short and long-term memory for contextual awareness
- **Performance Monitoring**: Real-time metrics and anomaly detection

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Agentic Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Planning   │  │  Execution   │  │  Evaluation  │      │
│  │    Agent     │→ │    Agent     │→ │    Agent     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         ↓                                      ↓             │
│  ┌──────────────┐                     ┌──────────────┐      │
│  │   Learning   │←────────────────────│   Feedback   │      │
│  │    Agent     │                     │  Aggregator  │      │
│  └──────────────┘                     └──────────────┘      │
│         ↓                                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Agent Coordinator                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                               │
│  Supporting Systems:                                         │
│  • Memory System (Short/Long-term)                          │
│  • RL Module (Q-learning, Multi-armed bandits)              │
│  • Performance Monitor                                       │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Base Agent (`base_agent.py`)

Foundation for all agents providing:
- State management (idle, busy, waiting, error)
- Action history tracking
- Goal management with priorities
- Inter-agent communication protocol
- Comprehensive metrics collection

**Key Classes:**
- `BaseAgent`: Abstract base class
- `AgentRole`: Enum of agent roles
- `AgentMessage`: Structured communication protocol
- `AgentState`: State enumeration

### 2. Planning Agent (`planning_agent.py`)

Strategic query analysis and plan generation:
- Query complexity assessment
- Multi-part query decomposition
- Dependency graph creation
- Strategy selection (vector, hybrid, graph, RAPTOR, community)
- Resource estimation and optimization

**Capabilities:**
- Simple to very complex query handling
- Temporal, comparison, and aggregation query detection
- Parallel vs sequential execution planning
- Contingency plan generation

### 3. Execution Agent (`execution_agent.py`)

Reliable action execution with error handling:
- Retrieval operations
- Graph queries
- Summarization tasks
- Community analysis
- RAPTOR hierarchical queries
- Automatic retry with exponential backoff

**Features:**
- Progress monitoring
- Confidence scoring
- Provenance tracking
- Graceful degradation

### 4. Evaluation Agent (`evaluation_agent.py`)

Quality assessment and feedback generation:
- Relevance scoring
- Completeness checking
- Consistency validation
- Hallucination detection
- Multi-dimensional quality assessment

**Assessment Dimensions:**
- Relevance (query-result alignment)
- Completeness (sufficient information)
- Consistency (no contradictions)
- Hallucination (source verification)

### 5. Learning Agent (`learning_agent.py`)

Pattern recognition and system adaptation:
- Success/failure pattern identification
- Strategy performance tracking
- Model weight updates
- Knowledge base evolution
- Performance optimization recommendations

**Learning Methods:**
- Pattern-based learning from execution history
- Strategy performance tracking
- Automatic optimization trigger detection
- Persistent knowledge storage

### 6. Agent Coordinator (`coordinator.py`)

Central orchestration hub:
- Agent lifecycle management (spawn/terminate)
- Message routing with priorities
- Workflow orchestration
- Conflict resolution
- Consensus building

**Coordination Features:**
- Broadcast messaging
- Role-based routing
- Load balancing
- Parallel execution coordination

### 7. Feedback Aggregator (`feedback_aggregator.py`)

Multi-source feedback collection:
- Agent feedback aggregation
- User satisfaction tracking
- System metrics integration
- Weighted scoring
- Trend analysis

**Feedback Sources:**
- Agent evaluations
- User ratings
- System performance metrics
- External validators

### 8. Reinforcement Learning Module (`reinforcement_learning.py`)

Continuous improvement through RL:
- Q-learning for strategy selection
- Multi-armed bandits for exploration
- Reward function combining quality, speed, cost
- State-action value learning

**RL Components:**
- Q-table with state-action values
- Epsilon-greedy exploration
- Multi-objective reward calculation
- Persistent Q-table storage

### 9. Memory System (`memory_system.py`)

Multi-tier memory architecture:
- **Short-term**: Current session context (limited capacity)
- **Long-term**: Persistent important memories
- **Episodic**: Complete interaction records
- Memory consolidation and forgetting

**Memory Features:**
- Importance-based retention
- Automatic consolidation
- Search and retrieval
- Persistent storage

### 10. Performance Monitor (`performance_monitor.py`)

Real-time system monitoring:
- Latency tracking (P50, P95, P99)
- Throughput measurement
- Success/error rate tracking
- Cache hit rate monitoring
- Anomaly detection

**Monitoring Capabilities:**
- Sliding window metrics
- Trend analysis
- Alert generation
- Optimization recommendations

### 11. Agentic Pipeline (`agentic_pipeline.py`)

Complete orchestration layer:
- Multi-stage query processing
- Iterative improvement loops
- Memory-aware execution
- Learning integration
- Performance tracking

**Pipeline Stages:**
1. Query reception & memory recall
2. Adaptive planning
3. Multi-agent execution
4. Quality evaluation
5. Learning & optimization
6. Response generation
7. Memory update

## Operating Modes

### FAST Mode
- Speed optimized
- Minimal iterations
- Basic retrieval strategies
- Quick responses

### STANDARD Mode (Default)
- Balanced approach
- Quality + speed optimization
- Adaptive strategies
- Moderate iterations

### DEEP Mode
- Quality optimized
- Multiple improvement iterations
- Comprehensive analysis
- Advanced strategies

### LEARNING Mode
- Exploration focused
- Higher epsilon (exploration rate)
- Diverse strategy testing
- Pattern discovery

## Usage Examples

### Basic Query Processing

```python
from src.agents.agentic_pipeline import AgenticPipeline, AgenticMode

# Initialize pipeline
pipeline = AgenticPipeline(
    mode=AgenticMode.STANDARD,
    enable_learning=True
)

# Process query
result = await pipeline.process_query(
    "What is machine learning?",
    max_iterations=2,
    min_quality_threshold=0.7
)

print(f"Quality: {result.assessment.overall_score:.2f}")
print(f"Answer: {result.answer}")
```

### With Custom Components

```python
from src.retrieval.advanced_pipeline import AdvancedRetrievalPipeline
from src.knowledge_graph.graph_queries import GraphQueries
from src.summarization.raptor_pipeline import RAPTORPipeline

# Initialize with existing components
pipeline = AgenticPipeline(
    retrieval_pipeline=your_retrieval_pipeline,
    graph_queries=your_graph_queries,
    raptor_pipeline=your_raptor_pipeline,
    mode=AgenticMode.DEEP,
    enable_learning=True
)
```

### Iterative Improvement

```python
# Set high quality threshold to trigger improvements
result = await pipeline.process_query(
    "Complex multi-part question...",
    max_iterations=5,
    min_quality_threshold=0.85
)

print(f"Iterations: {result.iteration}")
print(f"Improved: {result.improved}")
print(f"Final quality: {result.assessment.overall_score:.2f}")
```

### Learning and Adaptation

```python
# Process multiple queries for learning
queries = ["query1", "query2", "query3"]

for query in queries:
    result = await pipeline.process_query(query)

# Check learning progress
stats = pipeline.get_statistics()
print(f"Patterns learned: {stats['learner']['patterns']}")
print(f"Q-table size: {stats['rl']['q_table_size']}")

# Consolidate knowledge
await pipeline.consolidate_knowledge()
```

## Configuration

### Pipeline Configuration

```python
pipeline = AgenticPipeline(
    mode=AgenticMode.STANDARD,
    enable_learning=True,
    storage_path=Path("data/agentic"),
    # External components
    retrieval_pipeline=retrieval_pipeline,
    graph_queries=graph_queries,
    raptor_pipeline=raptor_pipeline,
    community_pipeline=community_pipeline
)
```

### Agent Configuration

Agents can be configured individually:

```python
# Planning Agent
planner = PlanningAgent(agent_id="custom_planner")

# Execution Agent with custom retry logic
executor = ExecutionAgent(
    agent_id="executor",
    max_retries=5,
    retry_delay_ms=1000
)

# Evaluator with custom threshold
evaluator = EvaluationAgent(
    min_quality_threshold=0.8
)

# Learner with custom learning rate
learner = LearningAgent(
    learning_rate=0.15,
    storage_path=Path("custom/path")
)
```

## Testing

Run comprehensive test suite:

```bash
# All tests
pytest tests/test_agents.py -v

# Specific test class
pytest tests/test_agents.py::TestPlanningAgent -v

# Benchmarks
pytest tests/test_agents.py -m benchmark -v

# With coverage
pytest tests/test_agents.py --cov=src/agents --cov-report=html
```

## Demo

Run the comprehensive demo:

```bash
python scripts/demo_agentic.py
```

Demos include:
1. Basic query processing
2. Mode comparison
3. Iterative improvement
4. Learning progression
5. Complex query handling
6. Memory and context
7. Performance monitoring
8. Statistics and insights

## Performance

### Typical Metrics

- **Planning**: 50-200ms per query
- **Execution**: 500-2000ms depending on strategy
- **Evaluation**: 100-300ms
- **Learning**: 50-150ms
- **Total**: 700-2650ms for standard queries

### Optimization Tips

1. **Use FAST mode** for simple queries
2. **Enable caching** for repeated queries
3. **Limit iterations** if speed is critical
4. **Batch queries** when possible
5. **Monitor performance** and adjust thresholds

## Advanced Features

### Custom Reward Functions

```python
def custom_reward(quality, latency, cost, satisfaction):
    # Prioritize quality heavily
    return quality * 15.0 - (latency / 1000) - (cost * 5)

pipeline.rl_module.calculate_reward = custom_reward
```

### Custom Memory Consolidation

```python
# Aggressive consolidation
pipeline.memory_system.long_term.consolidate(min_importance=0.6)

# Promote recent important memories
stats = pipeline.memory_system.consolidate_memories()
print(f"Promoted: {stats['promoted_to_long_term']}")
```

### Custom Evaluation Criteria

```python
evaluator = EvaluationAgent()

# Override relevance assessment
def custom_relevance(query, results):
    # Custom logic
    return score

evaluator._assess_relevance = custom_relevance
```

## Architecture Principles

### 1. Autonomous Operation
- Agents make independent decisions
- No manual intervention required
- Self-correcting through learning

### 2. Multi-Agent Collaboration
- Specialized agents for specific tasks
- Coordinated through message passing
- Consensus building for conflicts

### 3. Continuous Improvement
- Reinforcement learning from outcomes
- Pattern recognition and adaptation
- Performance optimization over time

### 4. Graceful Degradation
- Retry logic for failures
- Fallback strategies
- Quality thresholds with escalation

### 5. Explainability
- Comprehensive logging
- Action history tracking
- Decision provenance

## Troubleshooting

### Issue: Low Quality Scores

**Solutions:**
- Increase `max_iterations`
- Lower `min_quality_threshold` initially
- Check evaluation agent configuration
- Review retrieval pipeline performance

### Issue: High Latency

**Solutions:**
- Switch to FAST mode
- Reduce `max_iterations`
- Enable caching
- Optimize retrieval pipeline

### Issue: Learning Not Improving

**Solutions:**
- Increase training episodes
- Check reward function
- Verify Q-table persistence
- Review learning rate

### Issue: Memory Overflow

**Solutions:**
- Reduce short-term capacity
- Increase consolidation frequency
- Adjust importance thresholds
- Clear old episodes

## Future Enhancements

1. **Multi-modal Support**: Images, audio, video
2. **Distributed Agents**: Cross-machine coordination
3. **Advanced RL**: Policy gradients, actor-critic
4. **Meta-learning**: Learning to learn
5. **Explainable AI**: Detailed decision explanations
6. **Human-in-loop**: Interactive refinement
7. **Federated Learning**: Privacy-preserving updates

## References

- Deep Q-Learning: Mnih et al., 2015
- Multi-Armed Bandits: Sutton & Barto, 2018
- RAPTOR: Sarthi et al., 2024
- Agentic RAG: Lewis et al., 2020

## License

Part of the DocTags RAG system.
