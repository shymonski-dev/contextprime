from pydantic import ValidationError
import pytest

from contextprime.api.models import AdvancedQueryRequest


def test_advanced_query_filters_accept_scalar_and_list_values():
    request = AdvancedQueryRequest(
        query="filter validation",
        filters={
            "source_id": "manual",
            "priority": 2,
            "active": True,
            "tags": ["engine", "transmission"],
        },
    )

    assert request.filters == {
        "source_id": "manual",
        "priority": 2,
        "active": True,
        "tags": ["engine", "transmission"],
    }


def test_advanced_query_filters_reject_invalid_filter_key():
    with pytest.raises(ValidationError):
        AdvancedQueryRequest(
            query="filter validation",
            filters={"bad-key; DROP": "x"},
        )


def test_advanced_query_filters_reject_nested_filter_value():
    with pytest.raises(ValidationError):
        AdvancedQueryRequest(
            query="filter validation",
            filters={"source": {"nested": "value"}},
        )
