"""
Offline benchmark helpers for retrieval policy comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import json
import re


@dataclass
class BenchmarkSample:
    """Single benchmark input sample."""

    query: str
    expected_ids: List[str] = field(default_factory=list)
    expected_terms: List[str] = field(default_factory=list)
    answer_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicySampleMetrics:
    """Per-query metrics for one retrieval policy run."""

    policy: str
    query: str
    retrieval_hit: bool
    id_hit: bool
    expected_term_coverage: float
    answer_term_coverage: float
    average_confidence: float
    latency_ms: float
    result_count: int


@dataclass
class PolicyAggregateMetrics:
    """Aggregated metrics across all benchmark samples for a policy."""

    policy: str
    samples: int
    retrieval_hit_rate: float
    id_hit_rate: float
    mean_expected_term_coverage: float
    mean_answer_term_coverage: float
    mean_confidence: float
    mean_latency_ms: float
    mean_result_count: float


def load_benchmark_samples(path: str | Path) -> List[BenchmarkSample]:
    """Load benchmark samples from a JSONL file."""
    source = Path(path)
    samples: List[BenchmarkSample] = []

    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        query = str(payload.get("query", "")).strip()
        if not query:
            continue
        samples.append(
            BenchmarkSample(
                query=query,
                expected_ids=[str(item) for item in payload.get("expected_ids", [])],
                expected_terms=[str(item) for item in payload.get("expected_terms", [])],
                answer_terms=[str(item) for item in payload.get("answer_terms", [])],
                metadata=dict(payload.get("metadata") or {}),
            )
        )
    return samples


def evaluate_policy_sample(
    *,
    policy: str,
    sample: BenchmarkSample,
    results: Sequence[Any],
    latency_ms: float,
) -> PolicySampleMetrics:
    """
    Compute per-sample retrieval and answer-quality proxy metrics.

    `results` must provide attributes: `id`, `content`, and `confidence`.
    """
    result_ids = {str(getattr(result, "id", "")) for result in results}
    joined_context = " ".join(str(getattr(result, "content", "")) for result in results)
    top_context = " ".join(str(getattr(result, "content", "")) for result in results[:3])

    id_hit = False
    if sample.expected_ids:
        id_hit = any(expected_id in result_ids for expected_id in sample.expected_ids)

    expected_term_coverage = compute_term_coverage(joined_context, sample.expected_terms)
    answer_terms = sample.answer_terms or sample.expected_terms
    answer_term_coverage = compute_term_coverage(top_context, answer_terms)
    retrieval_hit = bool(id_hit or (expected_term_coverage >= 0.45))

    confidence_values = [float(getattr(result, "confidence", 0.0)) for result in results]
    average_confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else 0.0
    )

    return PolicySampleMetrics(
        policy=policy,
        query=sample.query,
        retrieval_hit=retrieval_hit,
        id_hit=id_hit,
        expected_term_coverage=expected_term_coverage,
        answer_term_coverage=answer_term_coverage,
        average_confidence=average_confidence,
        latency_ms=float(latency_ms),
        result_count=len(results),
    )


def aggregate_policy_metrics(
    policy: str,
    sample_metrics: Iterable[PolicySampleMetrics],
) -> PolicyAggregateMetrics:
    """Aggregate per-query metrics into a policy summary."""
    metrics = list(sample_metrics)
    count = len(metrics)
    if count == 0:
        return PolicyAggregateMetrics(
            policy=policy,
            samples=0,
            retrieval_hit_rate=0.0,
            id_hit_rate=0.0,
            mean_expected_term_coverage=0.0,
            mean_answer_term_coverage=0.0,
            mean_confidence=0.0,
            mean_latency_ms=0.0,
            mean_result_count=0.0,
        )

    retrieval_hit_rate = sum(1 for item in metrics if item.retrieval_hit) / count
    id_hit_rate = sum(1 for item in metrics if item.id_hit) / count
    mean_expected_term_coverage = sum(item.expected_term_coverage for item in metrics) / count
    mean_answer_term_coverage = sum(item.answer_term_coverage for item in metrics) / count
    mean_confidence = sum(item.average_confidence for item in metrics) / count
    mean_latency_ms = sum(item.latency_ms for item in metrics) / count
    mean_result_count = sum(item.result_count for item in metrics) / count

    return PolicyAggregateMetrics(
        policy=policy,
        samples=count,
        retrieval_hit_rate=retrieval_hit_rate,
        id_hit_rate=id_hit_rate,
        mean_expected_term_coverage=mean_expected_term_coverage,
        mean_answer_term_coverage=mean_answer_term_coverage,
        mean_confidence=mean_confidence,
        mean_latency_ms=mean_latency_ms,
        mean_result_count=mean_result_count,
    )


def metrics_to_dict(metrics: PolicyAggregateMetrics) -> Dict[str, Any]:
    """Convert aggregate metrics to dictionary."""
    return asdict(metrics)


def sample_metrics_to_dict(metrics: PolicySampleMetrics) -> Dict[str, Any]:
    """Convert sample metrics to dictionary."""
    return asdict(metrics)


def compute_term_coverage(text: str, terms: Sequence[str]) -> float:
    """Compute fraction of expected terms present in text."""
    normalized_terms = _normalize_terms(terms)
    if not normalized_terms:
        return 0.0

    haystack = str(text or "").lower()
    hits = 0
    for term in normalized_terms:
        if term in haystack:
            hits += 1
    return hits / float(len(normalized_terms))


def _normalize_terms(terms: Sequence[str]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for term in terms:
        token = re.sub(r"\s+", " ", str(term or "").strip().lower())
        if len(token) < 2 or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized
