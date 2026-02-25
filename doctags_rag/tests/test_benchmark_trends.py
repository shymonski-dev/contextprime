import json
from pathlib import Path

from contextprime.retrieval.benchmark_trends import (
    append_trend_records,
    extract_trend_records,
    load_trend_history,
    write_trend_markdown,
)


def test_benchmark_trend_append_and_markdown(tmp_path):
    report_payload = {
        "generated_at": "2026-02-20T10:00:00+00:00",
        "dataset": "data/test_benchmark.jsonl",
        "strategy": "hybrid",
        "top_k": 5,
        "policies": [
            {
                "policy": "standard",
                "aggregate": {
                    "samples": 10,
                    "retrieval_hit_rate": 0.7,
                    "id_hit_rate": 0.6,
                    "mean_expected_term_coverage": 0.65,
                    "mean_answer_term_coverage": 0.63,
                    "mean_confidence": 0.71,
                    "mean_latency_ms": 12.2,
                    "mean_result_count": 5.0,
                },
            }
        ],
    }

    report_path = tmp_path / "benchmark.json"
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    records = extract_trend_records(report_payload, report_path=str(report_path))
    history_path = tmp_path / "history.jsonl"
    markdown_path = tmp_path / "trends.md"

    appended = append_trend_records(history_path, records)
    assert appended == 1

    history = load_trend_history(history_path)
    assert len(history) == 1
    assert history[0].policy == "standard"

    write_trend_markdown(history_records=history, output_path=markdown_path, max_runs=10)
    content = markdown_path.read_text(encoding="utf-8")
    assert "Retrieval Policy Trends" in content
    assert "standard" in content
