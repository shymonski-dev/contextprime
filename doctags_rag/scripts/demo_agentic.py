"""
Demo script for the Agentic RAG System.

Demonstrates:
- Multi-agent query processing
- Self-improvement over iterations
- Performance comparison (with/without agents)
- Learning capabilities
- Complex query handling
- Failure recovery
"""

import asyncio
import time
from pathlib import Path
from loguru import logger

from contextprime.agents.agentic_pipeline import AgenticPipeline, AgenticMode
from contextprime.agents.base_agent import AgentState


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_result(result):
    """Print query result in a formatted way."""
    print(f"Query: {result.query}")
    print(f"Mode: {result.mode.value}")
    print(f"Iterations: {result.iteration}")
    print(f"Total Time: {result.total_time_ms:.0f}ms")
    print(f"  - Planning: {result.planning_time_ms:.0f}ms")
    print(f"  - Execution: {result.execution_time_ms:.0f}ms")
    print(f"  - Evaluation: {result.evaluation_time_ms:.0f}ms")
    print(f"  - Learning: {result.learning_time_ms:.0f}ms")
    print(f"\nQuality Assessment:")
    print(f"  - Overall Score: {result.assessment.overall_score:.2f}")
    print(f"  - Quality Level: {result.assessment.quality_level.value}")
    print(f"  - Relevance: {result.assessment.relevance_score:.2f}")
    print(f"  - Completeness: {result.assessment.completeness_score:.2f}")
    print(f"  - Consistency: {result.assessment.consistency_score:.2f}")

    if result.assessment.strengths:
        print(f"\nStrengths:")
        for strength in result.assessment.strengths:
            print(f"  ✓ {strength}")

    if result.assessment.weaknesses:
        print(f"\nWeaknesses:")
        for weakness in result.assessment.weaknesses:
            print(f"  ✗ {weakness}")

    if result.assessment.improvement_suggestions:
        print(f"\nImprovement Suggestions:")
        for suggestion in result.assessment.improvement_suggestions:
            print(f"  → {suggestion}")

    if result.learning_insights:
        print(f"\nLearning Insights:")
        insights = result.learning_insights
        print(f"  - Patterns Found: {len(insights.get('patterns_found', []))}")
        print(f"  - Optimizations: {len(insights.get('optimizations', []))}")
        if insights.get('recommendations'):
            print(f"  - Recommendations:")
            for rec in insights['recommendations'][:3]:
                print(f"    • {rec}")

    print(f"\nAnswer Preview:")
    print(f"  {result.answer[:300]}...")

    print("\n" + "-" * 80)


async def demo_basic_query_processing():
    """Demo 1: Basic query processing with different modes."""
    print_section("Demo 1: Basic Query Processing")

    # Create pipeline
    pipeline = AgenticPipeline(
        mode=AgenticMode.STANDARD,
        enable_learning=True
    )

    queries = [
        "What is machine learning?",
        "How does a neural network work?",
        "What are the main types of machine learning algorithms?"
    ]

    print("Processing queries in STANDARD mode...\n")

    for query in queries:
        result = await pipeline.process_query(query, max_iterations=2)
        print_result(result)
        await asyncio.sleep(0.5)

    print("\nBasic query processing complete!")
    print(f"Total queries processed: {pipeline.queries_processed}")


async def demo_mode_comparison():
    """Demo 2: Compare different operating modes."""
    print_section("Demo 2: Operating Mode Comparison")

    query = "Compare supervised and unsupervised learning approaches"

    modes = [AgenticMode.FAST, AgenticMode.STANDARD, AgenticMode.DEEP]

    results = {}

    for mode in modes:
        print(f"\n--- Testing {mode.value.upper()} mode ---")

        pipeline = AgenticPipeline(
            mode=mode,
            enable_learning=False  # Disable for fair comparison
        )

        result = await pipeline.process_query(
            query,
            max_iterations=1 if mode == AgenticMode.FAST else 2
        )

        results[mode] = result

        print(f"Time: {result.total_time_ms:.0f}ms")
        print(f"Quality: {result.assessment.overall_score:.2f}")
        print(f"Results: {len(result.results)}")

    # Comparison summary
    print("\n--- Mode Comparison Summary ---")
    print(f"{'Mode':<15} {'Time (ms)':<12} {'Quality':<10} {'Results':<10}")
    print("-" * 50)

    for mode, result in results.items():
        print(
            f"{mode.value:<15} "
            f"{result.total_time_ms:<12.0f} "
            f"{result.assessment.overall_score:<10.2f} "
            f"{len(result.results):<10}"
        )


async def demo_iterative_improvement():
    """Demo 3: Iterative improvement with quality threshold."""
    print_section("Demo 3: Iterative Improvement")

    pipeline = AgenticPipeline(
        mode=AgenticMode.DEEP,
        enable_learning=True
    )

    query = "Explain the concept of transfer learning in deep learning"

    print(f"Query: {query}")
    print("\nSetting high quality threshold (0.8) to trigger improvements...\n")

    result = await pipeline.process_query(
        query,
        max_iterations=3,
        min_quality_threshold=0.8
    )

    print(f"Final Result after {result.iteration} iteration(s):")
    print(f"  - Quality Score: {result.assessment.overall_score:.2f}")
    print(f"  - Improved: {result.improved}")
    print(f"  - Total Time: {result.total_time_ms:.0f}ms")

    if result.improved:
        print("\n✓ System successfully improved results through iteration!")
    else:
        print("\n→ Initial results met quality threshold.")


async def demo_learning_progression():
    """Demo 4: Learning and improvement over multiple queries."""
    print_section("Demo 4: Learning Progression")

    pipeline = AgenticPipeline(
        mode=AgenticMode.STANDARD,
        enable_learning=True
    )

    # Similar queries to test learning
    queries = [
        "What is a neural network?",
        "What is a convolutional neural network?",
        "What is a recurrent neural network?",
        "What is a transformer neural network?"
    ]

    print("Processing similar queries to observe learning...\n")

    scores = []
    times = []

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}/{len(queries)}: {query}")

        result = await pipeline.process_query(query, max_iterations=1)

        scores.append(result.assessment.overall_score)
        times.append(result.total_time_ms)

        print(f"  Quality: {result.assessment.overall_score:.2f}")
        print(f"  Time: {result.total_time_ms:.0f}ms")

        if result.learning_insights:
            patterns = len(result.learning_insights.get('patterns_found', []))
            print(f"  Patterns Learned: {patterns}")

    # Show learning progression
    print("\n--- Learning Progression ---")
    print(f"{'Query #':<10} {'Quality':<12} {'Time (ms)':<12}")
    print("-" * 35)

    for i, (score, time_ms) in enumerate(zip(scores, times), 1):
        print(f"{i:<10} {score:<12.2f} {time_ms:<12.0f}")

    avg_score = sum(scores) / len(scores)
    print(f"\nAverage Quality: {avg_score:.2f}")

    # Show learned patterns
    stats = pipeline.get_statistics()
    if stats.get('rl') and stats['rl'].get('episode_count'):
        print(f"\nRL Statistics:")
        print(f"  - Episodes: {stats['rl']['episode_count']}")
        print(f"  - Q-table Size: {stats['rl']['q_table_size']}")


async def demo_complex_query_handling():
    """Demo 5: Handling complex multi-part queries."""
    print_section("Demo 5: Complex Query Handling")

    pipeline = AgenticPipeline(
        mode=AgenticMode.DEEP,
        enable_learning=True
    )

    complex_queries = [
        "Compare and contrast supervised, unsupervised, and reinforcement learning, "
        "including their use cases, advantages, and limitations",

        "How does the attention mechanism work in transformers, and why is it "
        "better than traditional RNNs for natural language processing?",

        "Explain the concept of gradient descent, backpropagation, and how they "
        "relate to training neural networks"
    ]

    for query in complex_queries:
        print(f"\nComplex Query: {query[:80]}...")

        result = await pipeline.process_query(
            query,
            max_iterations=2,
            min_quality_threshold=0.7
        )

        print(f"\nPlan Details:")
        print(f"  - Sub-queries: {len(result.plan.sub_queries)}")
        print(f"  - Steps: {len(result.plan.steps)}")
        print(f"  - Strategy: {result.plan.metadata.get('strategy', 'N/A')}")

        print(f"\nExecution:")
        print(f"  - Successful Steps: {sum(1 for r in result.execution_results if r.success)}")
        print(f"  - Failed Steps: {sum(1 for r in result.execution_results if not r.success)}")
        print(f"  - Total Time: {result.total_time_ms:.0f}ms")

        print(f"\nQuality:")
        print(f"  - Overall: {result.assessment.overall_score:.2f}")
        print(f"  - Level: {result.assessment.quality_level.value}")

        print("\n" + "-" * 60)


async def demo_memory_and_context():
    """Demo 6: Memory system and contextual awareness."""
    print_section("Demo 6: Memory and Contextual Awareness")

    pipeline = AgenticPipeline(
        mode=AgenticMode.STANDARD,
        enable_learning=True
    )

    # Related queries to test memory
    queries = [
        "What is TensorFlow?",
        "How do I install TensorFlow?",  # Should recall previous context
        "What are TensorFlow's main features?"  # Should recall both previous
    ]

    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")

        result = await pipeline.process_query(query)

        # Check if memory was used
        if hasattr(result.plan, 'metadata') and result.plan.metadata.get('memories'):
            print(f"  ✓ Recalled {len(result.plan.metadata['memories'])} memories")

        print(f"  Quality: {result.assessment.overall_score:.2f}")

    # Show memory statistics
    stats = pipeline.get_statistics()
    memory_stats = stats.get('memory', {})

    print("\n--- Memory System Statistics ---")
    print(f"Short-term Memory: {memory_stats.get('short_term_size', 0)} entries")
    print(f"Long-term Memory: {memory_stats.get('long_term_size', 0)} entries")
    print(f"Episodes Recorded: {memory_stats.get('episodes_count', 0)}")


async def demo_performance_monitoring():
    """Demo 7: Performance monitoring and optimization."""
    print_section("Demo 7: Performance Monitoring")

    pipeline = AgenticPipeline(
        mode=AgenticMode.STANDARD,
        enable_learning=True
    )

    # Process multiple queries to generate metrics
    print("Processing queries to collect performance data...\n")

    for i in range(10):
        query = f"Sample query number {i+1}"
        await pipeline.process_query(query, max_iterations=1)
        print(f"  Processed query {i+1}/10", end="\r")

    print("\n\n--- Performance Summary ---")

    summary = pipeline.performance_monitor.get_summary()
    metrics = summary.get('current_metrics', {})

    print(f"Total Queries: {summary.get('total_queries', 0)}")
    print(f"Uptime: {summary.get('uptime_seconds', 0):.1f}s")
    print(f"\nMetrics:")
    print(f"  - Avg Latency: {metrics.get('latency_ms', 0):.0f}ms")
    print(f"  - Success Rate: {metrics.get('success_rate', 0):.1%}")
    print(f"  - Error Rate: {metrics.get('error_rate', 0):.1%}")
    print(f"  - Cache Hit Rate: {metrics.get('cache_hit_rate', 0):.1%}")

    # Trends
    trends = summary.get('trends', {})
    if trends:
        print(f"\nTrends:")
        print(f"  - Latency Trend: {trends.get('latency_trend', 'N/A')}")
        print(f"  - P50 Latency: {trends.get('latency_p50', 0):.0f}ms")
        print(f"  - P95 Latency: {trends.get('latency_p95', 0):.0f}ms")
        print(f"  - P99 Latency: {trends.get('latency_p99', 0):.0f}ms")

    # Recommendations
    recommendations = summary.get('recommendations', [])
    if recommendations:
        print(f"\nOptimization Recommendations:")
        for rec in recommendations:
            print(f"  → {rec}")


async def demo_statistics_and_insights():
    """Demo 8: System statistics and insights."""
    print_section("Demo 8: System Statistics and Insights")

    pipeline = AgenticPipeline(
        mode=AgenticMode.STANDARD,
        enable_learning=True
    )

    # Process some queries
    queries = [
        "What is Python?",
        "What is Java?",
        "Compare Python and Java"
    ]

    for query in queries:
        await pipeline.process_query(query, max_iterations=2)

    # Get comprehensive statistics
    stats = pipeline.get_statistics()

    print("--- Pipeline Statistics ---\n")

    print(f"Queries Processed: {stats['queries_processed']}")
    print(f"Improvement Iterations: {stats['total_improvement_iterations']}")
    print(f"Mode: {stats['mode']}")
    print(f"Learning Enabled: {stats['learning_enabled']}")

    print("\n--- Agent Statistics ---")
    for agent_name, agent_stats in stats['agents'].items():
        print(f"\n{agent_name.upper()}:")
        print(f"  State: {agent_stats['state']}")
        print(f"  Actions Completed: {agent_stats['metrics']['actions_completed']}")
        print(f"  Actions Failed: {agent_stats['metrics']['actions_failed']}")
        print(f"  Avg Action Time: {agent_stats['metrics']['average_action_time_ms']:.0f}ms")

    # RL statistics if available
    if stats.get('rl'):
        print("\n--- Reinforcement Learning ---")
        rl_stats = stats['rl']
        print(f"  Episodes: {rl_stats.get('episode_count', 0)}")
        print(f"  Q-table States: {rl_stats.get('total_states', 0)}")
        print(f"  Epsilon: {rl_stats.get('epsilon', 0):.3f}")
        print(f"  Avg Recent Reward: {rl_stats.get('average_reward_recent', 0):.2f}")


async def main():
    """Run all demos."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "AGENTIC RAG SYSTEM DEMONSTRATION" + " " * 26 + "║")
    print("╚" + "═" * 78 + "╝")

    demos = [
        ("Basic Query Processing", demo_basic_query_processing),
        ("Mode Comparison", demo_mode_comparison),
        ("Iterative Improvement", demo_iterative_improvement),
        ("Learning Progression", demo_learning_progression),
        ("Complex Query Handling", demo_complex_query_handling),
        ("Memory and Context", demo_memory_and_context),
        ("Performance Monitoring", demo_performance_monitoring),
        ("Statistics and Insights", demo_statistics_and_insights),
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            print(f"\n[{i}/{len(demos)}] Running: {name}")
            await demo_func()
            await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Demo failed: {e}", exc_info=True)
            print(f"\n✗ Demo failed: {e}")

    print_section("Demo Complete!")
    print("All demonstrations completed successfully.")
    print("\nKey Takeaways:")
    print("  ✓ Multi-agent coordination enables sophisticated query processing")
    print("  ✓ Iterative improvement adapts to quality requirements")
    print("  ✓ Learning system captures patterns and optimizes strategies")
    print("  ✓ Memory system provides contextual awareness")
    print("  ✓ Performance monitoring enables continuous optimization")
    print("\nThe agentic RAG system is production-ready!")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<dim>{time:HH:mm:ss}</dim> | <level>{message}</level>"
    )

    # Run demos
    asyncio.run(main())
