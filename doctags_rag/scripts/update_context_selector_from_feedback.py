#!/usr/bin/env python3
"""
Update the context selector model from labeled feedback samples.

Feedback dataset format (JSON lines):
{"query": "...", "content": "...", "label": 0 or 1}
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.context_selector import (
    SelectorExample,
    TrainableContextSelector,
)


def _load_examples(path: Path) -> List[SelectorExample]:
    samples: List[SelectorExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        samples.append(
            SelectorExample(
                query=str(payload.get("query", "")),
                content=str(payload.get("content", "")),
                label=int(payload.get("label", 0)),
            )
        )
    return samples


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Update context selector with labeled feedback")
    parser.add_argument("--feedback", required=True, help="Path to feedback JSON lines dataset")
    parser.add_argument(
        "--model-path",
        default="models/context_selector.json",
        help="Path to selector model (loaded and overwritten by default)",
    )
    parser.add_argument(
        "--output-model-path",
        default=None,
        help="Optional output path for updated model",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold used for evaluation",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Holdout ratio for before and after evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for holdout split",
    )
    parser.add_argument(
        "--max-terms",
        type=int,
        default=20000,
        help="Maximum retained token weights",
    )
    parser.add_argument(
        "--min-new-weight",
        type=float,
        default=0.2,
        help="Minimum weight given to new feedback during update",
    )
    args = parser.parse_args()

    feedback_path = Path(args.feedback)
    if not feedback_path.exists():
        raise FileNotFoundError(f"Feedback dataset not found: {feedback_path}")

    examples = _load_examples(feedback_path)
    if len(examples) < 5:
        raise ValueError("Feedback dataset must contain at least 5 examples")

    model_path = Path(args.model_path)
    if model_path.exists():
        selector = TrainableContextSelector.load(model_path)
        print(f"Loaded existing selector model: {model_path}")
    else:
        selector = TrainableContextSelector()
        print(f"No existing model found, starting from scratch: {model_path}")

    random.Random(args.seed).shuffle(examples)
    holdout_size = max(1, int(len(examples) * max(0.0, min(args.holdout_ratio, 0.8))))
    train_examples = examples[holdout_size:]
    holdout_examples = examples[:holdout_size]
    if len(train_examples) < 1:
        train_examples = examples
        holdout_examples = []

    if holdout_examples:
        before_metrics = selector.evaluate(holdout_examples, threshold=args.threshold)
        _print_metrics("Before update (holdout)", before_metrics)

    selector.partial_fit(
        train_examples,
        max_terms=max(100, int(args.max_terms)),
        min_new_weight=max(0.0, min(1.0, float(args.min_new_weight))),
    )

    output_model_path = Path(args.output_model_path) if args.output_model_path else model_path
    selector.save(output_model_path)
    print(f"Updated selector model saved to: {output_model_path}")
    print(f"Total training examples tracked: {selector.training_examples}")
    print(f"Feedback examples consumed this run: {len(train_examples)}")

    if holdout_examples:
        after_metrics = selector.evaluate(holdout_examples, threshold=args.threshold)
        _print_metrics("After update (holdout)", after_metrics)


if __name__ == "__main__":
    main()
