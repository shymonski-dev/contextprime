#!/usr/bin/env python3
"""
Demo Script for Advanced Retrieval Features.

Demonstrates:
- Confidence scoring in action
- Query routing decisions
- Iterative refinement
- Reranking comparison
- Performance improvements
"""

import sys
from pathlib import Path
import time
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.confidence_scorer import ConfidenceScorer
from src.retrieval.query_router import QueryRouter
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.reranker import Reranker
from src.retrieval.cache_manager import CacheManager
from src.retrieval.iterative_refiner import IterativeRefiner

from loguru import logger


# Sample data for demo
SAMPLE_DOCUMENTS = [
    {
        "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on the development of computer programs that can access data and use it to learn for themselves.",
        "score": 0.85,
        "metadata": {"source": "ML Textbook", "author": "Andrew Ng", "year": 2020}
    },
    {
        "content": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks. Learning can be supervised, semi-supervised or unsupervised.",
        "score": 0.78,
        "metadata": {"source": "AI Research Paper", "year": 2021}
    },
    {
        "content": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
        "score": 0.72,
        "metadata": {"source": "NLP Overview", "year": 2019}
    },
    {
        "content": "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.",
        "score": 0.65,
        "metadata": {"source": "CV Handbook", "year": 2018}
    },
    {
        "content": "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward.",
        "score": 0.70,
        "metadata": {"source": "RL Tutorial", "year": 2022}
    }
]

TEST_QUERIES = [
    "What is machine learning?",
    "How are deep learning and machine learning related?",
    "Explain the differences between supervised and unsupervised learning",
    "What are the applications of NLP in modern AI systems?",
]


def print_separator(title: str = ""):
    """Print a section separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)
    print()


def print_result(result: Dict[str, Any], index: int):
    """Pretty print a single result."""
    print(f"\n[Result #{index + 1}]")
    print(f"Score: {result.get('score', 0):.3f}")
    if 'confidence' in result:
        print(f"Confidence: {result['confidence']:.3f}")
    print(f"Content: {result['content'][:150]}...")
    if result.get('metadata'):
        print(f"Metadata: {result['metadata']}")


def demo_confidence_scoring():
    """Demonstrate confidence scoring."""
    print_separator("DEMO 1: Confidence Scoring")

    scorer = ConfidenceScorer()
    query = TEST_QUERIES[0]

    print(f"Query: '{query}'\n")
    print("Scoring results with multi-signal analysis...\n")

    confidence_scores = scorer.score_results_batch(query, SAMPLE_DOCUMENTS)

    for i, (result, conf_score) in enumerate(zip(SAMPLE_DOCUMENTS, confidence_scores)):
        print(f"\n--- Result #{i + 1} ---")
        print(f"Content: {result['content'][:100]}...")
        print(f"\nConfidence Analysis:")
        print(f"  Overall Score: {conf_score.overall_score:.3f}")
        print(f"  Level: {conf_score.level.value.upper()}")
        print(f"  Recommended Action: {conf_score.corrective_action.value}")
        print(f"\n  Signal Breakdown:")
        for signal_name, signal_value in conf_score.signals.to_dict().items():
            print(f"    {signal_name}: {signal_value:.3f}")
        print(f"\n  Reasoning: {conf_score.reasoning}")

    # Aggregate statistics
    print("\n" + "-" * 80)
    aggregated = scorer.aggregate_confidence(confidence_scores)
    print("\nAggregated Confidence Statistics:")
    print(f"  Average Confidence: {aggregated['average_confidence']:.3f}")
    print(f"  Correct: {aggregated['correct_count']}")
    print(f"  Ambiguous: {aggregated['ambiguous_count']}")
    print(f"  Incorrect: {aggregated['incorrect_count']}")
    print(f"  Recommended Action: {aggregated['recommended_action']}")


def demo_query_routing():
    """Demonstrate query routing."""
    print_separator("DEMO 2: Query Routing")

    router = QueryRouter()

    print("Analyzing different query types...\n")

    for query in TEST_QUERIES:
        print(f"\nQuery: '{query}'")
        print("-" * 80)

        analysis = router.analyze_query(query)

        print(f"  Type: {analysis.query_type.value}")
        print(f"  Complexity: {analysis.complexity.value}")
        print(f"  Recommended Strategy: {analysis.recommended_strategy.value}")
        print(f"  Confidence: {analysis.confidence:.3f}")
        print(f"  Key Entities: {', '.join(analysis.key_entities) if analysis.key_entities else 'None'}")
        print(f"  Keywords: {', '.join(analysis.keywords[:5])}")


def demo_query_expansion():
    """Demonstrate query expansion."""
    print_separator("DEMO 3: Query Expansion")

    expander = QueryExpander()
    query = TEST_QUERIES[0]

    print(f"Original Query: '{query}'\n")

    # Try different strategies
    strategies = ["conservative", "comprehensive", "aggressive"]

    for strategy in strategies:
        expanded = expander.expand_query(query, strategy=strategy)

        print(f"\n{strategy.upper()} Strategy:")
        print(f"  Expanded: '{expanded.expanded_query}'")
        print(f"  Synonyms: {', '.join(expanded.synonyms) if expanded.synonyms else 'None'}")
        print(f"  Semantic Terms: {', '.join(expanded.semantic_terms) if expanded.semantic_terms else 'None'}")

    # Related queries
    print("\n" + "-" * 80)
    print("\nRelated Query Suggestions:")
    suggestions = expander.suggest_related_queries(query, num_suggestions=3)
    for i, suggestion in enumerate(suggestions, 1):
        print(f"  {i}. {suggestion}")


def demo_reranking():
    """Demonstrate result reranking."""
    print_separator("DEMO 4: Result Reranking")

    reranker = Reranker(enable_cross_encoder=False)
    query = TEST_QUERIES[0]

    print(f"Query: '{query}'\n")
    print("Original Results:")
    for i, result in enumerate(SAMPLE_DOCUMENTS[:3]):
        print(f"  {i + 1}. Score: {result['score']:.3f} - {result['content'][:80]}...")

    print("\nReranking with multiple features...")

    reranked = reranker.rerank(
        query=query,
        results=SAMPLE_DOCUMENTS,
        top_k=3,
        enable_diversity=True
    )

    print("\nReranked Results:")
    for i, result in enumerate(reranked):
        print(f"  {i + 1}. Score: {result.reranked_score:.3f} (was #{result.original_rank + 1}) - {result.content[:80]}...")

    # Explain top result
    print("\n" + "-" * 80)
    print("\nExplanation for Top Result:")
    explanation = reranker.explain_ranking(reranked[0])
    print(f"  Total Score: {explanation['total_score']:.3f}")
    print(f"  Rank Change: {explanation['rank_change']:+d} positions")
    print(f"  Top Contributing Features: {', '.join(explanation['top_features'])}")

    print("\n  Feature Contributions:")
    for feature, contribution in sorted(
        explanation['contributions'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"    {feature}: {contribution:.4f}")


def demo_caching():
    """Demonstrate caching."""
    print_separator("DEMO 5: Caching")

    import tempfile
    import numpy as np

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "cache"
        cache_manager = CacheManager(cache_dir=cache_dir)

        query = TEST_QUERIES[0]
        query_vector = np.random.rand(384)  # Mock embedding

        print(f"Query: '{query}'\n")

        # First retrieval (cache miss)
        print("First retrieval (cache miss)...")
        start = time.time()
        cached = cache_manager.get_query_results(query, query_vector)
        elapsed_miss = time.time() - start

        print(f"  Result: Cache MISS")
        print(f"  Time: {elapsed_miss * 1000:.2f}ms")

        # Cache results
        print("\nCaching results...")
        cache_manager.cache_query_results(query, query_vector, SAMPLE_DOCUMENTS)

        # Second retrieval (cache hit)
        print("\nSecond retrieval (cache hit)...")
        start = time.time()
        cached = cache_manager.get_query_results(query, query_vector)
        elapsed_hit = time.time() - start

        print(f"  Result: Cache HIT")
        print(f"  Time: {elapsed_hit * 1000:.2f}ms")
        print(f"  Speedup: {elapsed_miss / elapsed_hit:.1f}x")
        print(f"  Results Retrieved: {len(cached)}")

        # Show statistics
        print("\n" + "-" * 80)
        print("\nCache Statistics:")
        stats = cache_manager.get_statistics()
        for cache_name, cache_stats in stats.items():
            if isinstance(cache_stats, dict) and 'hit_rate' in cache_stats:
                print(f"\n  {cache_name}:")
                print(f"    Size: {cache_stats.get('size', 0)}")
                print(f"    Hits: {cache_stats.get('hits', 0)}")
                print(f"    Misses: {cache_stats.get('misses', 0)}")
                print(f"    Hit Rate: {cache_stats.get('hit_rate', 0):.2%}")


def demo_iterative_refinement():
    """Demonstrate iterative refinement."""
    print_separator("DEMO 6: Iterative Refinement")

    refiner = IterativeRefiner(max_iterations=2)
    query = "What are advanced machine learning techniques?"

    print(f"Query: '{query}'\n")

    # Simulate initial results with low confidence
    initial_results = [
        {
            "content": "Machine learning includes various techniques.",
            "score": 0.5,
            "confidence": 0.4,
            "metadata": {}
        },
        {
            "content": "Advanced techniques are used in AI research.",
            "score": 0.45,
            "confidence": 0.35,
            "metadata": {}
        }
    ]

    print("Initial Results (Low Confidence):")
    for i, result in enumerate(initial_results):
        print(f"  {i + 1}. Confidence: {result['confidence']:.2f} - {result['content']}")

    print("\nInitiating iterative refinement...\n")

    # Mock retrieval function
    iteration_count = [0]

    def mock_retrieval(refined_query, context=None):
        iteration_count[0] += 1
        print(f"  Iteration {iteration_count[0]}: Refined query = '{refined_query}'")

        # Simulate improved results
        return [
            {
                "content": f"Advanced machine learning includes deep learning, transfer learning, and ensemble methods (iteration {iteration_count[0]}).",
                "score": 0.7 + iteration_count[0] * 0.1,
                "confidence": 0.6 + iteration_count[0] * 0.1,
                "metadata": {}
            },
            {
                "content": f"Neural architecture search and meta-learning are cutting-edge techniques (iteration {iteration_count[0]}).",
                "score": 0.65 + iteration_count[0] * 0.1,
                "confidence": 0.55 + iteration_count[0] * 0.1,
                "metadata": {}
            }
        ]

    refined_results, steps = refiner.refine_retrieval(
        original_query=query,
        initial_results=initial_results,
        retrieval_func=mock_retrieval
    )

    print("\n" + "-" * 80)
    print("\nRefinement Summary:")
    summary = refiner.get_refinement_summary(steps)
    print(f"  Total Iterations: {summary['total_iterations']}")
    print(f"  New Results Added: {summary['total_new_results']}")
    print(f"  Final Confidence: {summary['final_confidence']:.2f}")

    print("\nFinal Results (After Refinement):")
    for i, result in enumerate(refined_results[:3]):
        print(f"  {i + 1}. Confidence: {result.confidence:.2f} - {result.content[:100]}...")


def demo_comparison():
    """Compare retrieval with and without advanced features."""
    print_separator("DEMO 7: Performance Comparison")

    query = TEST_QUERIES[1]
    print(f"Query: '{query}'\n")

    # Basic retrieval
    print("Basic Retrieval (no enhancements):")
    basic_start = time.time()
    basic_results = sorted(SAMPLE_DOCUMENTS, key=lambda x: x['score'], reverse=True)[:3]
    basic_time = time.time() - basic_start

    for i, result in enumerate(basic_results):
        print(f"  {i + 1}. Score: {result['score']:.3f} - {result['content'][:80]}...")
    print(f"\n  Time: {basic_time * 1000:.2f}ms")

    # Advanced retrieval
    print("\n" + "-" * 80)
    print("\nAdvanced Retrieval (with all enhancements):")
    advanced_start = time.time()

    # Apply all enhancements
    router = QueryRouter()
    expander = QueryExpander()
    scorer = ConfidenceScorer()
    reranker = Reranker(enable_cross_encoder=False)

    # Route
    strategy, analysis = router.route_query(query)

    # Expand
    expanded = expander.expand_query(query)

    # Score
    confidence_scores = scorer.score_results_batch(query, SAMPLE_DOCUMENTS)

    # Rerank
    results_with_conf = [
        {**doc, "confidence": score.overall_score}
        for doc, score in zip(SAMPLE_DOCUMENTS, confidence_scores)
    ]
    reranked = reranker.rerank(query, results_with_conf, top_k=3)

    advanced_time = time.time() - advanced_start

    for i, result in enumerate(reranked):
        print(f"  {i + 1}. Score: {result.reranked_score:.3f} (Conf: {result.metadata.get('confidence', 0):.3f}) - {result.content[:80]}...")
    print(f"\n  Time: {advanced_time * 1000:.2f}ms")

    # Summary
    print("\n" + "-" * 80)
    print("\nSummary:")
    print(f"  Query Type: {analysis.query_type.value}")
    print(f"  Query Expanded: Yes")
    print(f"  Average Confidence: {sum(r.metadata.get('confidence', 0) for r in reranked) / len(reranked):.3f}")
    print(f"  Processing Overhead: {(advanced_time - basic_time) * 1000:.2f}ms")


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "ADVANCED RETRIEVAL FEATURES DEMO" + " " * 31 + "║")
    print("║" + " " * 20 + "Contextprime" + " " * 40 + "║")
    print("╚" + "=" * 78 + "╝")

    demos = [
        ("Confidence Scoring", demo_confidence_scoring),
        ("Query Routing", demo_query_routing),
        ("Query Expansion", demo_query_expansion),
        ("Result Reranking", demo_reranking),
        ("Caching System", demo_caching),
        ("Iterative Refinement", demo_iterative_refinement),
        ("Performance Comparison", demo_comparison),
    ]

    print("\nAvailable Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print(f"  {len(demos) + 1}. Run All Demos")
    print("  0. Exit")

    while True:
        try:
            choice = input("\nSelect demo (0-{}): ".format(len(demos) + 1))
            choice = int(choice)

            if choice == 0:
                print("\nExiting demo. Goodbye!")
                break
            elif choice == len(demos) + 1:
                # Run all demos
                for name, demo_func in demos:
                    try:
                        demo_func()
                        input("\nPress Enter to continue to next demo...")
                    except KeyboardInterrupt:
                        print("\n\nDemo interrupted. Returning to menu...")
                        break
                    except Exception as e:
                        logger.error(f"Demo '{name}' failed: {e}")
                        input("\nPress Enter to continue...")
            elif 1 <= choice <= len(demos):
                name, demo_func = demos[choice - 1]
                try:
                    demo_func()
                    input("\nPress Enter to return to menu...")
                except Exception as e:
                    logger.error(f"Demo failed: {e}")
                    input("\nPress Enter to continue...")
            else:
                print("Invalid choice. Please try again.")

        except KeyboardInterrupt:
            print("\n\nExiting demo. Goodbye!")
            break
        except ValueError:
            print("Invalid input. Please enter a number.")


if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<level>{message}</level>"
    )

    main()
