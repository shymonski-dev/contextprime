from pathlib import Path

from src.retrieval.feedback_capture_store import FeedbackCaptureStore


def test_feedback_capture_store_records_query_and_feedback(tmp_path):
    store = FeedbackCaptureStore(root_dir=tmp_path)

    query_id = store.record_query_event(
        query="how does community retrieval work",
        request_payload={"strategy": "hybrid"},
        results=[
            {
                "id": "chunk-1",
                "content": "Community retrieval uses summary anchors.",
                "score": 0.9,
                "confidence": 0.8,
                "source": "graph",
                "metadata": {"policy": "community"},
            }
        ],
        metadata={"graph_policy": "community"},
    )

    assert query_id.startswith("qry_")
    assert store.query_exists(query_id) is True

    feedback_id = store.record_feedback_event(
        query_id=query_id,
        helpful=True,
        selected_result_ids=["chunk-1"],
        result_labels=[{"result_id": "chunk-1", "label": 1}],
    )
    assert feedback_id.startswith("fbk_")

    stats = store.get_statistics()
    assert stats["query_events"] == 1
    assert stats["feedback_events"] == 1
    assert Path(stats["root_dir"]) == tmp_path


def test_feedback_capture_store_rejects_unknown_query_id(tmp_path):
    store = FeedbackCaptureStore(root_dir=tmp_path)
    try:
        store.record_feedback_event(
            query_id="missing-query",
            helpful=False,
        )
        raise AssertionError("Expected ValueError")
    except ValueError:
        pass
