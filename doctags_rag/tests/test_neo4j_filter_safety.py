import pytest

from src.knowledge_graph.neo4j_manager import Neo4jManager


def test_build_safe_property_filters_uses_parameterized_values():
    manager = Neo4jManager.__new__(Neo4jManager)

    clause, params = manager._build_safe_property_filters(
        {
            "doc_id": "doc-1",
            "chunk_index": 3,
            "tags": ["engine", "transmission"],
            "archived": None,
        }
    )

    assert "n.`doc_id` = $filter_value_0" in clause
    assert "n.`chunk_index` = $filter_value_1" in clause
    assert "n.`tags` IN $filter_value_2" in clause
    assert "n.`archived` IS NULL" in clause
    assert params["filter_value_0"] == "doc-1"
    assert params["filter_value_1"] == 3
    assert params["filter_value_2"] == ["engine", "transmission"]


def test_build_safe_property_filters_rejects_injection_key():
    manager = Neo4jManager.__new__(Neo4jManager)

    with pytest.raises(ValueError):
        manager._build_safe_property_filters({"name OR 1=1": "value"})
