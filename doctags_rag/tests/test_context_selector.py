from pathlib import Path

from contextprime.retrieval.context_selector import (
    SelectorExample,
    TrainableContextSelector,
)


def test_context_selector_training_and_scoring():
    selector = TrainableContextSelector()
    examples = [
        SelectorExample(query="machine learning retrieval", content="retrieval for machine learning", label=1),
        SelectorExample(query="machine learning retrieval", content="cooking recipe and ingredients", label=0),
        SelectorExample(query="graph search", content="graph search neighborhood traversal", label=1),
        SelectorExample(query="graph search", content="sports schedule for local teams", label=0),
    ]

    selector.fit(examples)

    positive_score = selector.score("machine learning retrieval", "retrieval with machine learning evidence")
    negative_score = selector.score("machine learning retrieval", "restaurant menu and travel guides")

    assert positive_score > negative_score
    assert selector.training_examples == len(examples)


def test_context_selector_select_and_persistence(tmp_path):
    selector = TrainableContextSelector()
    selector.fit(
        [
            SelectorExample(query="retrieval grounding", content="grounding evidence retrieval", label=1),
            SelectorExample(query="retrieval grounding", content="music festivals and events", label=0),
        ]
    )

    results = [
        {"id": "a", "content": "grounding evidence retrieval", "metadata": {}},
        {"id": "b", "content": "music festivals and events", "metadata": {}},
    ]
    selected = selector.select(
        query="retrieval grounding",
        results=results,
        top_k=2,
        min_score=0.2,
        min_selected=1,
    )
    assert selected
    assert "context_selector_score" in selected[0]["metadata"]

    model_path = tmp_path / "selector.json"
    selector.save(model_path)
    loaded = TrainableContextSelector.load(model_path)

    assert loaded.training_examples == selector.training_examples
    assert loaded.score("retrieval grounding", "grounding evidence retrieval") > 0


def test_context_selector_partial_fit_updates_model():
    selector = TrainableContextSelector()
    selector.fit(
        [
            SelectorExample(query="graph retrieval", content="graph retrieval with evidence", label=1),
            SelectorExample(query="graph retrieval", content="sports news and weather", label=0),
        ]
    )
    initial_examples = selector.training_examples
    initial_score = selector.score("graph retrieval", "graph retrieval with evidence")

    selector.partial_fit(
        [
            SelectorExample(query="graph retrieval", content="community graph summaries", label=1),
            SelectorExample(query="graph retrieval", content="cooking instructions", label=0),
        ],
        min_new_weight=0.3,
    )

    updated_score = selector.score("graph retrieval", "community graph summaries")
    assert selector.training_examples > initial_examples
    assert updated_score > 0.45
    assert initial_score > 0.45
