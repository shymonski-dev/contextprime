from dataclasses import dataclass

import pytest

from src.retrieval.policy_benchmark import (
    BenchmarkSample,
    aggregate_policy_metrics,
    compute_term_coverage,
    evaluate_policy_sample,
)


@dataclass
class DummyResult:
    id: str
    content: str
    confidence: float


def test_compute_term_coverage_handles_simple_overlap():
    coverage = compute_term_coverage(
        "retrieval with graph communities and supporting evidence",
        ["graph", "evidence", "missing"],
    )
    assert coverage == 2 / 3


def test_evaluate_policy_sample_uses_id_and_term_hits():
    sample = BenchmarkSample(
        query="how does community retrieval help",
        expected_ids=["chunk-1"],
        expected_terms=["community", "retrieval", "evidence"],
        answer_terms=["community", "evidence"],
    )
    results = [
        DummyResult(
            id="chunk-1",
            content="Community retrieval adds evidence from connected entities.",
            confidence=0.8,
        ),
        DummyResult(
            id="chunk-2",
            content="Additional retrieval context.",
            confidence=0.6,
        ),
    ]

    metrics = evaluate_policy_sample(
        policy="community",
        sample=sample,
        results=results,
        latency_ms=12.4,
    )

    assert metrics.retrieval_hit is True
    assert metrics.id_hit is True
    assert metrics.expected_term_coverage >= 2 / 3
    assert metrics.answer_term_coverage >= 1.0
    assert metrics.result_count == 2


def test_aggregate_policy_metrics_computes_means():
    sample = BenchmarkSample(query="sample")
    metrics_a = evaluate_policy_sample(
        policy="standard",
        sample=sample,
        results=[DummyResult(id="a", content="sample text", confidence=0.4)],
        latency_ms=10.0,
    )
    metrics_b = evaluate_policy_sample(
        policy="standard",
        sample=sample,
        results=[DummyResult(id="b", content="sample text", confidence=0.8)],
        latency_ms=20.0,
    )

    aggregate = aggregate_policy_metrics("standard", [metrics_a, metrics_b])
    assert aggregate.samples == 2
    assert aggregate.mean_confidence == pytest.approx(0.6)
    assert aggregate.mean_latency_ms == 15.0
