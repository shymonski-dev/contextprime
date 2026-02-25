#!/usr/bin/env python3
"""
Train and evaluate the trainable context selector model.

Dataset format:
- JSON lines file
- Each line: {"query": "...", "content": "...", "label": 0 or 1}
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

from contextprime.retrieval.context_selector import (
    SelectorExample,
    TrainableContextSelector,
)


def _load_examples(path: Path) -> List[SelectorExample]:
    examples: List[SelectorExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        payload = json.loads(stripped)
        examples.append(
            SelectorExample(
                query=str(payload.get("query", "")),
                content=str(payload.get("content", "")),
                label=int(payload.get("label", 0)),
            )
        )
    return examples


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
    parser = argparse.ArgumentParser(description="Train and evaluate context selector")
    parser.add_argument("--dataset", required=True, help="Path to JSON lines dataset")
    parser.add_argument(
        "--model-out",
        default="models/context_selector.json",
        help="Output path for trained selector model",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training split ratio",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for split",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    examples = _load_examples(dataset_path)
    if len(examples) < 10:
        raise ValueError("Dataset must contain at least 10 examples")

    random.Random(args.seed).shuffle(examples)
    split_index = max(1, min(len(examples) - 1, int(len(examples) * args.train_ratio)))
    train_examples = examples[:split_index]
    eval_examples = examples[split_index:]

    selector = TrainableContextSelector()
    selector.fit(train_examples)

    train_metrics = selector.evaluate(train_examples, threshold=args.threshold)
    eval_metrics = selector.evaluate(eval_examples, threshold=args.threshold)

    model_path = Path(args.model_out)
    selector.save(model_path)

    print(f"Model saved to: {model_path}")
    print(f"Training examples: {len(train_examples)}")
    print(f"Evaluation examples: {len(eval_examples)}")
    _print_metrics("Training", train_metrics)
    _print_metrics("Evaluation", eval_metrics)


if __name__ == "__main__":
    main()
