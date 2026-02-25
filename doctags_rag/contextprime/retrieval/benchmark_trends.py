"""
Benchmark trend publishing helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
import json


@dataclass
class BenchmarkTrendRecord:
    """Single benchmark trend record."""

    recorded_at: str
    benchmark_generated_at: str
    policy: str
    samples: int
    retrieval_hit_rate: float
    id_hit_rate: float
    mean_expected_term_coverage: float
    mean_answer_term_coverage: float
    mean_confidence: float
    mean_latency_ms: float
    mean_result_count: float
    benchmark_dataset: str
    benchmark_strategy: str
    benchmark_top_k: int
    benchmark_report_path: str


def load_benchmark_report(path: str | Path) -> Dict[str, Any]:
    """Load benchmark report json payload."""
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def extract_trend_records(
    report_payload: Dict[str, Any],
    *,
    report_path: str,
    recorded_at: str | None = None,
) -> List[BenchmarkTrendRecord]:
    """Extract trend records from benchmark report payload."""
    now = recorded_at or datetime.now(timezone.utc).isoformat()
    generated_at = str(report_payload.get("generated_at", ""))
    dataset = str(report_payload.get("dataset", ""))
    strategy = str(report_payload.get("strategy", ""))
    top_k = int(report_payload.get("top_k", 0))

    records: List[BenchmarkTrendRecord] = []
    for policy_payload in report_payload.get("policies", []):
        policy = str(policy_payload.get("policy", "")).strip()
        aggregate = dict(policy_payload.get("aggregate") or {})
        if not policy:
            continue
        records.append(
            BenchmarkTrendRecord(
                recorded_at=now,
                benchmark_generated_at=generated_at,
                policy=policy,
                samples=int(aggregate.get("samples", 0)),
                retrieval_hit_rate=float(aggregate.get("retrieval_hit_rate", 0.0)),
                id_hit_rate=float(aggregate.get("id_hit_rate", 0.0)),
                mean_expected_term_coverage=float(aggregate.get("mean_expected_term_coverage", 0.0)),
                mean_answer_term_coverage=float(aggregate.get("mean_answer_term_coverage", 0.0)),
                mean_confidence=float(aggregate.get("mean_confidence", 0.0)),
                mean_latency_ms=float(aggregate.get("mean_latency_ms", 0.0)),
                mean_result_count=float(aggregate.get("mean_result_count", 0.0)),
                benchmark_dataset=dataset,
                benchmark_strategy=strategy,
                benchmark_top_k=top_k,
                benchmark_report_path=str(report_path),
            )
        )
    return records


def append_trend_records(path: str | Path, records: Iterable[BenchmarkTrendRecord]) -> int:
    """Append trend records to json lines history file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    if not rows:
        return 0
    with target.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")
    return len(rows)


def load_trend_history(path: str | Path) -> List[BenchmarkTrendRecord]:
    """Load trend history json lines file."""
    source = Path(path)
    if not source.exists():
        return []
    records: List[BenchmarkTrendRecord] = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = dict(json.loads(line))
        except json.JSONDecodeError:
            continue
        records.append(BenchmarkTrendRecord(**payload))
    return records


def write_trend_markdown(
    *,
    history_records: Sequence[BenchmarkTrendRecord],
    output_path: str | Path,
    max_runs: int = 30,
) -> None:
    """Publish benchmark trend markdown summary."""
    rows = list(history_records)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        target.write_text("# Retrieval Policy Trends\n\nNo records yet.\n", encoding="utf-8")
        return

    sorted_rows = sorted(rows, key=lambda row: row.recorded_at, reverse=True)
    recent_rows = sorted_rows[: max(1, int(max_runs)) * max(1, len({row.policy for row in rows}))]
    latest_snapshot = _latest_by_policy(sorted_rows)

    lines: List[str] = []
    lines.append("# Retrieval Policy Trends")
    lines.append("")
    lines.append(f"Last updated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append("## Latest Snapshot By Policy")
    lines.append("")
    lines.append(
        "| Policy | Retrieval Hit | Identifier Hit | Expected Coverage | Answer Coverage | Confidence | Latency (ms) | Samples |"
    )
    lines.append(
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in latest_snapshot:
        lines.append(
            f"| {row.policy} | {row.retrieval_hit_rate:.3f} | {row.id_hit_rate:.3f} | "
            f"{row.mean_expected_term_coverage:.3f} | {row.mean_answer_term_coverage:.3f} | "
            f"{row.mean_confidence:.3f} | {row.mean_latency_ms:.1f} | {row.samples} |"
        )

    lines.append("")
    lines.append("## Recent Records")
    lines.append("")
    lines.append(
        "| Recorded At | Policy | Retrieval Hit | Identifier Hit | Expected Coverage | Answer Coverage | Confidence | Latency (ms) | Dataset |"
    )
    lines.append(
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    for row in recent_rows:
        lines.append(
            f"| {row.recorded_at} | {row.policy} | {row.retrieval_hit_rate:.3f} | "
            f"{row.id_hit_rate:.3f} | {row.mean_expected_term_coverage:.3f} | "
            f"{row.mean_answer_term_coverage:.3f} | {row.mean_confidence:.3f} | "
            f"{row.mean_latency_ms:.1f} | {row.benchmark_dataset} |"
        )

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latest_by_policy(records: Sequence[BenchmarkTrendRecord]) -> List[BenchmarkTrendRecord]:
    latest: Dict[str, BenchmarkTrendRecord] = {}
    for row in records:
        current = latest.get(row.policy)
        if current is None or row.recorded_at > current.recorded_at:
            latest[row.policy] = row
    return sorted(latest.values(), key=lambda row: row.policy)
