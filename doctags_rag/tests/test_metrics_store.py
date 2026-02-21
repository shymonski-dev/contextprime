from src.api.services.metrics_store import OperationalMetricsStore


def test_metrics_store_persists_counters_across_instances(tmp_path):
    db_path = tmp_path / "metrics.db"
    store_a = OperationalMetricsStore(db_path=db_path)

    assert store_a.get("total_uploads") == 0
    assert store_a.get("total_queries") == 0

    store_a.increment_uploads()
    store_a.increment_uploads()
    store_a.increment_queries()

    store_b = OperationalMetricsStore(db_path=db_path)
    snapshot = store_b.get_snapshot()

    assert snapshot["total_uploads"] == 2
    assert snapshot["total_queries"] == 1
