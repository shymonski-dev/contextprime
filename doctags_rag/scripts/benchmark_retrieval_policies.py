#!/usr/bin/env python3
"""
Benchmark retrieval policy modes on a labeled query dataset.

Dataset format (JSON lines):
{"query": "...", "expected_ids": ["..."], "expected_terms": ["..."], "answer_terms": ["..."]}
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.policy_benchmark import (
    aggregate_policy_metrics,
    evaluate_policy_sample,
    load_benchmark_samples,
    metrics_to_dict,
    sample_metrics_to_dict,
)
from src.retrieval.benchmark_trends import (
    append_trend_records,
    extract_trend_records,
    load_trend_history,
    write_trend_markdown,
)


def _parse_policy_list(
    raw: str,
    *,
    valid_values: List[str],
    default_policy: str,
) -> List[str]:
    tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
    if not tokens:
        tokens = [default_policy]

    valid_set = set(valid_values)
    invalid = [token for token in tokens if token not in valid_set]
    if invalid:
        raise ValueError(f"Invalid graph policy values: {invalid}. Valid values: {sorted(valid_set)}")
    return tokens


def _parse_strategy(value: str, mapping: Mapping[str, Any]) -> Any:
    normalized = value.strip().lower()
    if normalized not in mapping:
        raise ValueError("strategy must be one of: vector, graph, hybrid")
    return mapping[normalized]


def _print_policy_summary(metrics: Any) -> None:
    print(
        f"{metrics.policy:>10} | samples={metrics.samples:4d} | "
        f"retrieval_hit={metrics.retrieval_hit_rate:.3f} | "
        f"id_hit={metrics.id_hit_rate:.3f} | "
        f"expected_term={metrics.mean_expected_term_coverage:.3f} | "
        f"answer_term={metrics.mean_answer_term_coverage:.3f} | "
        f"confidence={metrics.mean_confidence:.3f} | "
        f"latency_ms={metrics.mean_latency_ms:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval policy modes")
    parser.add_argument("--dataset", required=True, help="Path to benchmark dataset (JSON lines)")
    parser.add_argument(
        "--policies",
        default="standard,local,global,drift,community",
        help="Comma separated graph policies",
    )
    parser.add_argument(
        "--strategy",
        default="hybrid",
        help="Retrieval strategy to run (vector, graph, hybrid)",
    )
    parser.add_argument("--top-k", type=int, default=6, help="Top-k results per query")
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Optional vector collection override",
    )
    parser.add_argument(
        "--graph-vector-index",
        default=None,
        help="Optional graph vector index override",
    )
    parser.add_argument(
        "--output",
        default="reports/retrieval_policy_benchmark.json",
        help="Output report file path",
    )
    parser.add_argument(
        "--publish-trends",
        action="store_true",
        help="Append benchmark output to trend history and update markdown summary",
    )
    parser.add_argument(
        "--trend-history",
        default="reports/retrieval_policy_trend_history.jsonl",
        help="Trend history json lines path",
    )
    parser.add_argument(
        "--trend-markdown",
        default="reports/retrieval_policy_trends.md",
        help="Trend summary markdown path",
    )
    parser.add_argument(
        "--trend-max-runs",
        type=int,
        default=30,
        help="Maximum recent runs in trend markdown",
    )
    args = parser.parse_args()

    from src.embeddings import OpenAIEmbeddingModel
    from src.retrieval.hybrid_retriever import (
        GraphRetrievalPolicy,
        HybridRetriever,
        SearchStrategy,
    )

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    samples = load_benchmark_samples(dataset_path)
    if not samples:
        raise ValueError("Dataset has no usable samples")

    valid_policies = [item.value for item in GraphRetrievalPolicy]
    policies = _parse_policy_list(
        args.policies,
        valid_values=valid_policies,
        default_policy=GraphRetrievalPolicy.STANDARD.value,
    )
    strategy = _parse_strategy(
        args.strategy,
        mapping={
            "vector": SearchStrategy.VECTOR_ONLY,
            "graph": SearchStrategy.GRAPH_ONLY,
            "hybrid": SearchStrategy.HYBRID,
        },
    )

    embedder = OpenAIEmbeddingModel()
    retriever = HybridRetriever()

    query_vectors = embedder.encode([sample.query for sample in samples])

    report_rows: List[Dict[str, object]] = []
    print("Policy benchmark summary")
    print("-" * 130)
    for policy in policies:
        sample_rows: List[PolicySampleMetrics] = []
        for sample, query_vector in zip(samples, query_vectors):
            results, metrics = retriever.search(
                query_vector=query_vector,
                query_text=sample.query,
                top_k=max(1, int(args.top_k)),
                strategy=strategy,
                graph_policy=policy,
                collection_name=args.collection_name,
                vector_index_name=args.graph_vector_index,
            )
            sample_metrics = evaluate_policy_sample(
                policy=policy,
                sample=sample,
                results=results,
                latency_ms=metrics.total_time_ms,
            )
            sample_rows.append(sample_metrics)

        aggregate = aggregate_policy_metrics(policy, sample_rows)
        _print_policy_summary(aggregate)
        report_rows.append(
            {
                "policy": policy,
                "aggregate": metrics_to_dict(aggregate),
                "samples": [sample_metrics_to_dict(item) for item in sample_rows],
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "samples": len(samples),
        "top_k": int(args.top_k),
        "strategy": strategy.value,
        "collection_name": args.collection_name,
        "graph_vector_index": args.graph_vector_index,
        "policies": report_rows,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("-" * 130)
    print(f"Benchmark report saved to: {output_path}")

    if args.publish_trends:
        trend_records = extract_trend_records(payload, report_path=str(output_path))
        appended = append_trend_records(Path(args.trend_history), trend_records)
        trend_history = load_trend_history(Path(args.trend_history))
        write_trend_markdown(
            history_records=trend_history,
            output_path=Path(args.trend_markdown),
            max_runs=max(1, int(args.trend_max_runs)),
        )
        print(f"Trend records appended: {appended}")
        print(f"Trend history path: {Path(args.trend_history)}")
        print(f"Trend markdown path: {Path(args.trend_markdown)}")


if __name__ == "__main__":
    main()
