from src.retrieval.feedback_dataset import build_selector_examples_from_events


def test_build_selector_examples_from_events():
    query_events = [
        {
            "query_id": "q1",
            "query": "how does retrieval grounding improve quality",
            "results": [
                {"id": "r1", "content": "Grounding improves quality with evidence."},
                {"id": "r2", "content": "Unrelated text about sports scores."},
            ],
        }
    ]
    feedback_events = [
        {
            "query_id": "q1",
            "helpful": False,
            "selected_result_ids": [],
            "result_labels": [{"result_id": "r1", "label": 1}],
        }
    ]

    examples, summary = build_selector_examples_from_events(query_events, feedback_events)

    assert summary.total_examples >= 2
    assert summary.positive_examples >= 1
    assert summary.negative_examples >= 1
    assert any(example.label == 1 for example in examples)
    assert any(example.label == 0 for example in examples)
