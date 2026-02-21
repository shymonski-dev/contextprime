"""
Build context selector training data from production retrieval feedback logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json

from .context_selector import SelectorExample


@dataclass
class FeedbackDatasetSummary:
    """Summary statistics for generated feedback training examples."""

    total_examples: int
    positive_examples: int
    negative_examples: int
    query_count: int
    feedback_events: int


def load_jsonl_records(path: str | Path) -> List[Dict[str, Any]]:
    """Load JSON lines records from path."""
    source = Path(path)
    if not source.exists():
        return []

    records: List[Dict[str, Any]] = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            records.append(dict(json.loads(line)))
        except json.JSONDecodeError:
            continue
    return records


def build_selector_examples_from_events(
    query_events: Sequence[Dict[str, Any]],
    feedback_events: Sequence[Dict[str, Any]],
    min_negative_per_feedback: int = 1,
    max_examples_per_feedback: int = 20,
) -> Tuple[List[SelectorExample], FeedbackDatasetSummary]:
    """
    Build labeled selector examples from query and feedback event records.

    Labeling rules:
    - explicit `result_labels` entries are used first
    - `selected_result_ids` are treated as positive examples
    - when `helpful` is false, unselected top results are treated as negatives
    """
    query_index: Dict[str, Dict[str, Any]] = {}
    for event in query_events:
        query_id = str(event.get("query_id", "")).strip()
        if not query_id:
            continue
        query_index[query_id] = event

    examples: List[SelectorExample] = []
    seen = set()
    feedback_count = 0
    touched_queries = set()

    for feedback in feedback_events:
        query_id = str(feedback.get("query_id", "")).strip()
        query_event = query_index.get(query_id)
        if not query_event:
            continue
        query_text = str(query_event.get("query", "")).strip()
        if not query_text:
            continue

        result_map = _build_result_content_index(query_event.get("results", []))
        if not result_map:
            continue

        touched_queries.add(query_id)
        feedback_count += 1

        explicit_labels = _normalize_labels(feedback.get("result_labels", []))
        selected_ids = _normalize_id_list(feedback.get("selected_result_ids", []))
        helpful = feedback.get("helpful")
        helpful_bool = bool(helpful) if helpful is not None else None

        per_feedback_count = 0
        used_ids = set()

        for result_id, label in explicit_labels:
            content = result_map.get(result_id)
            if not content:
                continue
            key = (query_text, content, label)
            if key in seen:
                continue
            examples.append(SelectorExample(query=query_text, content=content, label=label))
            seen.add(key)
            used_ids.add(result_id)
            per_feedback_count += 1
            if per_feedback_count >= max_examples_per_feedback:
                break

        if per_feedback_count < max_examples_per_feedback:
            for result_id in selected_ids:
                if result_id in used_ids:
                    continue
                content = result_map.get(result_id)
                if not content:
                    continue
                key = (query_text, content, 1)
                if key in seen:
                    continue
                examples.append(SelectorExample(query=query_text, content=content, label=1))
                seen.add(key)
                used_ids.add(result_id)
                per_feedback_count += 1
                if per_feedback_count >= max_examples_per_feedback:
                    break

        if helpful_bool is False and per_feedback_count < max_examples_per_feedback:
            negatives = 0
            for result_id, content in result_map.items():
                if result_id in used_ids or result_id in selected_ids:
                    continue
                key = (query_text, content, 0)
                if key in seen:
                    continue
                examples.append(SelectorExample(query=query_text, content=content, label=0))
                seen.add(key)
                used_ids.add(result_id)
                per_feedback_count += 1
                negatives += 1
                if negatives >= max(1, int(min_negative_per_feedback)):
                    break
                if per_feedback_count >= max_examples_per_feedback:
                    break

    positive = sum(1 for example in examples if int(example.label) > 0)
    negative = len(examples) - positive
    summary = FeedbackDatasetSummary(
        total_examples=len(examples),
        positive_examples=positive,
        negative_examples=negative,
        query_count=len(touched_queries),
        feedback_events=feedback_count,
    )
    return examples, summary


def save_selector_examples(path: str | Path, examples: Iterable[SelectorExample]) -> None:
    """Persist selector examples to JSON lines file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for example in examples:
            payload = {
                "query": example.query,
                "content": example.content,
                "label": int(example.label),
            }
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _build_result_content_index(results: Any) -> Dict[str, str]:
    result_map: Dict[str, str] = {}
    if not isinstance(results, list):
        return result_map
    for row in results:
        payload = dict(row or {})
        result_id = str(payload.get("id", "")).strip()
        content = str(payload.get("content", "")).strip()
        if not result_id or not content:
            continue
        result_map[result_id] = content
    return result_map


def _normalize_id_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    normalized: List[str] = []
    seen = set()
    for value in values:
        token = str(value).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _normalize_labels(values: Any) -> List[Tuple[str, int]]:
    if not isinstance(values, list):
        return []
    labels: List[Tuple[str, int]] = []
    for row in values:
        payload = dict(row or {})
        result_id = str(payload.get("result_id", "")).strip()
        if not result_id:
            continue
        label = int(payload.get("label", 0))
        if label not in (0, 1):
            continue
        labels.append((result_id, label))
    return labels
