#!/usr/bin/env python3
"""
Build context selector training data from production feedback logs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.feedback_dataset import (
    build_selector_examples_from_events,
    load_jsonl_records,
    save_selector_examples,
)
from src.retrieval.feedback_capture_store import FeedbackCaptureStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Build selector dataset from retrieval feedback logs")
    parser.add_argument(
        "--feedback-db",
        default="data/feedback/retrieval_feedback.db",
        help="Path to feedback sqlite database (preferred when available)",
    )
    parser.add_argument(
        "--query-events",
        default="data/feedback/retrieval_query_events.jsonl",
        help="Path to retrieval query events json lines file",
    )
    parser.add_argument(
        "--feedback-events",
        default="data/feedback/retrieval_feedback_events.jsonl",
        help="Path to retrieval feedback events json lines file",
    )
    parser.add_argument(
        "--output",
        default="data/feedback/context_selector_feedback_dataset.jsonl",
        help="Output dataset path",
    )
    parser.add_argument(
        "--min-negative-per-feedback",
        type=int,
        default=1,
        help="Minimum number of negatives from unhelpful feedback events",
    )
    parser.add_argument(
        "--max-examples-per-feedback",
        type=int,
        default=20,
        help="Maximum generated examples from one feedback event",
    )
    args = parser.parse_args()

    query_events = []
    feedback_events = []
    db_path = Path(args.feedback_db)

    if db_path.exists():
        store = FeedbackCaptureStore(
            root_dir=db_path.parent,
            db_name=db_path.name,
            mirror_jsonl=False,
        )
        query_events = store.load_query_events()
        feedback_events = store.load_feedback_events()
    else:
        query_events = load_jsonl_records(Path(args.query_events))
        feedback_events = load_jsonl_records(Path(args.feedback_events))

    examples, summary = build_selector_examples_from_events(
        query_events=query_events,
        feedback_events=feedback_events,
        min_negative_per_feedback=max(1, int(args.min_negative_per_feedback)),
        max_examples_per_feedback=max(1, int(args.max_examples_per_feedback)),
    )
    save_selector_examples(Path(args.output), examples)

    print(f"Query events loaded: {len(query_events)}")
    print(f"Feedback events loaded: {len(feedback_events)}")
    print(f"Feedback events matched to queries: {summary.feedback_events}")
    print(f"Queries represented in dataset: {summary.query_count}")
    print(f"Examples written: {summary.total_examples}")
    print(f"Positive examples: {summary.positive_examples}")
    print(f"Negative examples: {summary.negative_examples}")
    print(f"Dataset path: {Path(args.output)}")


if __name__ == "__main__":
    main()
