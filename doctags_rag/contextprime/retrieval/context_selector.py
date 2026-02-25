"""
Trainable context selector for retrieval results.

This module provides:
- Lightweight training from labeled query and passage pairs
- Passage scoring for a query
- Result filtering and ranking
- Model persistence
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import math
import re


@dataclass
class SelectorExample:
    """Training or evaluation sample for context selection."""

    query: str
    content: str
    label: int  # 1 for relevant, 0 for irrelevant


@dataclass
class SelectorMetrics:
    """Evaluation metrics for context selection."""

    accuracy: float
    precision: float
    recall: float
    total_examples: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int


class TrainableContextSelector:
    """
    Trainable lexical selector using token weight statistics.

    The model learns token weights from labeled data and uses them with
    query and content overlap features to score candidate passages.
    """

    _STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "if", "then", "else", "in",
        "on", "at", "to", "for", "of", "with", "by", "from", "as", "is",
        "are", "was", "were", "be", "been", "being", "do", "does", "did",
        "what", "who", "where", "when", "why", "how", "which", "that",
        "this", "these", "those", "it", "its", "their", "there", "can",
        "could", "would", "should", "may", "might", "will", "about",
    }

    def __init__(
        self,
        token_weights: Optional[Dict[str, float]] = None,
        bias: float = 0.0,
        training_examples: int = 0,
    ) -> None:
        self.token_weights = token_weights or {}
        self.bias = float(bias)
        self.training_examples = int(training_examples)

    def fit(
        self,
        examples: Iterable[SelectorExample],
        max_terms: int = 20000,
    ) -> None:
        """Train selector weights from labeled examples."""
        positive_count = 0
        negative_count = 0
        token_counts: Dict[str, List[int]] = {}
        total_examples = 0

        for example in examples:
            total_examples += 1
            label = 1 if int(example.label) > 0 else 0
            if label == 1:
                positive_count += 1
            else:
                negative_count += 1

            query_tokens = set(self._tokenize(example.query))
            content_tokens = set(self._tokenize(example.content))
            matched_tokens = query_tokens & content_tokens
            if not matched_tokens:
                matched_tokens = {"__no_overlap__"}

            for token in matched_tokens:
                stats = token_counts.setdefault(token, [0, 0])
                if label == 1:
                    stats[0] += 1
                else:
                    stats[1] += 1

        if total_examples == 0:
            self.token_weights = {}
            self.bias = 0.0
            self.training_examples = 0
            return

        # Prior log-odds with smoothing.
        self.bias = math.log((positive_count + 1.0) / (negative_count + 1.0))
        weighted_terms: List[Tuple[str, float]] = []
        for token, (token_positive, token_negative) in token_counts.items():
            weight = math.log((token_positive + 1.0) / (token_negative + 1.0))
            weighted_terms.append((token, weight))

        weighted_terms.sort(key=lambda item: abs(item[1]), reverse=True)
        self.token_weights = {term: weight for term, weight in weighted_terms[:max_terms]}
        self.training_examples = total_examples

    def partial_fit(
        self,
        examples: Iterable[SelectorExample],
        max_terms: int = 20000,
        min_new_weight: float = 0.2,
    ) -> None:
        """
        Update selector weights with additional labeled examples.

        This keeps existing learned weights while incorporating new feedback,
        then trims the vocabulary back to `max_terms`.
        """
        batch = list(examples)
        if not batch:
            return

        new_selector = TrainableContextSelector()
        new_selector.fit(batch, max_terms=max_terms)

        if self.training_examples <= 0:
            self.token_weights = dict(new_selector.token_weights)
            self.bias = float(new_selector.bias)
            self.training_examples = int(new_selector.training_examples)
            return

        old_examples = max(1, int(self.training_examples))
        new_examples = max(1, int(new_selector.training_examples))
        new_weight = max(float(min_new_weight), new_examples / float(old_examples + new_examples))
        new_weight = min(0.85, max(0.0, new_weight))
        old_weight = 1.0 - new_weight

        merged_weights: Dict[str, float] = {}
        all_terms = set(self.token_weights.keys()) | set(new_selector.token_weights.keys())
        for term in all_terms:
            merged = (
                old_weight * float(self.token_weights.get(term, 0.0))
                + new_weight * float(new_selector.token_weights.get(term, 0.0))
            )
            if abs(merged) >= 1e-8:
                merged_weights[term] = merged

        ordered_terms = sorted(
            merged_weights.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        self.token_weights = {term: weight for term, weight in ordered_terms[:max_terms]}
        self.bias = (old_weight * float(self.bias)) + (new_weight * float(new_selector.bias))
        self.training_examples = old_examples + new_examples

    def score(
        self,
        query: str,
        content: str,
    ) -> float:
        """Score a content passage for a query on a zero to one scale."""
        query_tokens = set(self._tokenize(query))
        content_tokens = set(self._tokenize(content))

        if not query_tokens or not content_tokens:
            return 0.0

        matched_tokens = query_tokens & content_tokens
        if not matched_tokens:
            matched_tokens = {"__no_overlap__"}

        token_weight_sum = 0.0
        for token in matched_tokens:
            token_weight_sum += self.token_weights.get(token, 0.0)
        token_weight_average = token_weight_sum / max(1, len(matched_tokens))

        overlap_ratio = len(query_tokens & content_tokens) / max(1, len(query_tokens))
        density = len(query_tokens & content_tokens) / max(1, len(content_tokens))

        raw_score = self.bias + token_weight_average + (1.25 * overlap_ratio) + (0.35 * density)
        return self._sigmoid(raw_score)

    def select(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
        min_score: float = 0.2,
        min_selected: int = 1,
    ) -> List[Dict[str, Any]]:
        """Select and rank result dictionaries by context selector score."""
        if not results:
            return []

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for result in results:
            updated = dict(result)
            content = str(updated.get("content", ""))
            selector_score = self.score(query, content)
            metadata = dict(updated.get("metadata") or {})
            metadata["context_selector_score"] = float(selector_score)
            updated["metadata"] = metadata
            scored.append((selector_score, updated))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [result for score, result in scored if score >= min_score]
        if len(selected) < min_selected:
            selected = [result for _, result in scored[:max(min_selected, 1)]]

        return selected[:top_k]

    def evaluate(
        self,
        examples: Iterable[SelectorExample],
        threshold: float = 0.5,
    ) -> SelectorMetrics:
        """Evaluate selector performance on labeled data."""
        true_positive = 0
        false_positive = 0
        true_negative = 0
        false_negative = 0
        total = 0

        for example in examples:
            total += 1
            label = 1 if int(example.label) > 0 else 0
            predicted = 1 if self.score(example.query, example.content) >= threshold else 0

            if predicted == 1 and label == 1:
                true_positive += 1
            elif predicted == 1 and label == 0:
                false_positive += 1
            elif predicted == 0 and label == 0:
                true_negative += 1
            else:
                false_negative += 1

        accuracy = (true_positive + true_negative) / max(1, total)
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        return SelectorMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            total_examples=total,
            true_positive=true_positive,
            false_positive=false_positive,
            true_negative=true_negative,
            false_negative=false_negative,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to dictionary."""
        return {
            "token_weights": self.token_weights,
            "bias": self.bias,
            "training_examples": self.training_examples,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainableContextSelector":
        """Create model from dictionary payload."""
        return cls(
            token_weights={str(k): float(v) for k, v in (payload.get("token_weights") or {}).items()},
            bias=float(payload.get("bias", 0.0)),
            training_examples=int(payload.get("training_examples", 0)),
        )

    def save(self, path: Path | str) -> None:
        """Save model to a file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "TrainableContextSelector":
        """Load model from a file."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"\b[a-zA-Z0-9]{2,}\b", text.lower())
        return [token for token in tokens if token not in self._STOPWORDS]

    def _sigmoid(self, value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)
