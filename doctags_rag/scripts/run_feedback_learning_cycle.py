#!/usr/bin/env python3
"""
Run automated feedback learning cycle for context selector updates.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.context_selector import SelectorExample, TrainableContextSelector
from src.retrieval.feedback_dataset import (
    build_selector_examples_from_events,
    load_jsonl_records,
    save_selector_examples,
)


def _print_metrics(name: str, metrics) -> None:
    print(f"{name} metrics")
    print(f"  accuracy:  {metrics.accuracy:.4f}")
    print(f"  precision: {metrics.precision:.4f}")
    print(f"  recall:    {metrics.recall:.4f}")
    print(f"  samples:   {metrics.total_examples}")
    print(
        "  confusion: "
        f"tp={metrics.true_positive}, fp={metrics.false_positive}, "
        f"tn={metrics.true_negative}, fn={metrics.false_negative}"
    )


def _split_examples(
    examples: List[SelectorExample],
    holdout_ratio: float,
    seed: int,
) -> tuple[List[SelectorExample], List[SelectorExample]]:
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    holdout_size = max(1, int(len(shuffled) * holdout_ratio))
    holdout_examples = shuffled[:holdout_size]
    train_examples = shuffled[holdout_size:]
    if not train_examples:
        train_examples = shuffled
        holdout_examples = []
    return train_examples, holdout_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automated selector update from production feedback")
    parser.add_argument(
        "--query-events",
        default="data/feedback/retrieval_query_events.jsonl",
        help="Path to query events log",
    )
    parser.add_argument(
        "--feedback-events",
        default="data/feedback/retrieval_feedback_events.jsonl",
        help="Path to feedback events log",
    )
    parser.add_argument(
        "--dataset-out",
        default="data/feedback/context_selector_feedback_dataset.jsonl",
        help="Output path for built selector dataset",
    )
    parser.add_argument(
        "--model-path",
        default="models/context_selector.json",
        help="Selector model path to load and update",
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=20,
        help="Minimum examples required before updating model",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Holdout ratio for before and after evaluation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Evaluation decision threshold",
    )
    parser.add_argument(
        "--min-negative-per-feedback",
        type=int,
        default=1,
        help="Minimum negatives generated from unhelpful feedback",
    )
    parser.add_argument(
        "--max-examples-per-feedback",
        type=int,
        default=20,
        help="Maximum generated examples from one feedback event",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=19,
        help="Random seed for train and holdout split",
    )
    parser.add_argument(
        "--min-new-weight",
        type=float,
        default=0.2,
        help="Minimum blend weight for new feedback examples during partial fit",
    )
    args = parser.parse_args()

    query_events = load_jsonl_records(Path(args.query_events))
    feedback_events = load_jsonl_records(Path(args.feedback_events))
    examples, summary = build_selector_examples_from_events(
        query_events=query_events,
        feedback_events=feedback_events,
        min_negative_per_feedback=max(1, int(args.min_negative_per_feedback)),
        max_examples_per_feedback=max(1, int(args.max_examples_per_feedback)),
    )
    save_selector_examples(Path(args.dataset_out), examples)

    print(f"Query events loaded: {len(query_events)}")
    print(f"Feedback events loaded: {len(feedback_events)}")
    print(f"Feedback events matched to queries: {summary.feedback_events}")
    print(f"Examples generated: {summary.total_examples}")
    print(f"Positive examples: {summary.positive_examples}")
    print(f"Negative examples: {summary.negative_examples}")
    print(f"Dataset path: {Path(args.dataset_out)}")

    if summary.total_examples < max(1, int(args.min_examples)):
        print(
            f"Skipping model update because examples ({summary.total_examples}) "
            f"are below min-examples ({max(1, int(args.min_examples))})."
        )
        return

    model_path = Path(args.model_path)
    if model_path.exists():
        selector = TrainableContextSelector.load(model_path)
        print(f"Loaded existing selector model: {model_path}")
    else:
        selector = TrainableContextSelector()
        print(f"No existing model found. Initializing new selector model: {model_path}")

    holdout_ratio = max(0.0, min(0.8, float(args.holdout_ratio)))
    train_examples, holdout_examples = _split_examples(examples, holdout_ratio=holdout_ratio, seed=int(args.seed))

    if holdout_examples:
        before_metrics = selector.evaluate(holdout_examples, threshold=float(args.threshold))
        _print_metrics("Before update (holdout)", before_metrics)

    selector.partial_fit(
        train_examples,
        min_new_weight=max(0.0, min(1.0, float(args.min_new_weight))),
    )
    selector.save(model_path)
    print(f"Updated selector model saved to: {model_path}")
    print(f"Total training examples tracked: {selector.training_examples}")

    if holdout_examples:
        after_metrics = selector.evaluate(holdout_examples, threshold=float(args.threshold))
        _print_metrics("After update (holdout)", after_metrics)


if __name__ == "__main__":
    main()
