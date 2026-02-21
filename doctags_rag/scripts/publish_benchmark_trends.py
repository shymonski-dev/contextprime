#!/usr/bin/env python3
"""
Append benchmark report metrics to trend history and publish markdown summary.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.benchmark_trends import (
    append_trend_records,
    extract_trend_records,
    load_benchmark_report,
    load_trend_history,
    write_trend_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish retrieval policy benchmark trends")
    parser.add_argument(
        "--report",
        required=True,
        help="Benchmark report json path from benchmark_retrieval_policies.py",
    )
    parser.add_argument(
        "--history",
        default="reports/retrieval_policy_trend_history.jsonl",
        help="Trend history json lines output path",
    )
    parser.add_argument(
        "--markdown",
        default="reports/retrieval_policy_trends.md",
        help="Trend summary markdown output path",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=30,
        help="Maximum recent runs to show in markdown summary",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        raise FileNotFoundError(f"Benchmark report not found: {report_path}")

    payload = load_benchmark_report(report_path)
    records = extract_trend_records(payload, report_path=str(report_path))
    appended = append_trend_records(Path(args.history), records)
    history = load_trend_history(Path(args.history))
    write_trend_markdown(
        history_records=history,
        output_path=Path(args.markdown),
        max_runs=max(1, int(args.max_runs)),
    )

    print(f"Trend records appended: {appended}")
    print(f"Trend history path: {Path(args.history)}")
    print(f"Trend markdown path: {Path(args.markdown)}")


if __name__ == "__main__":
    main()
