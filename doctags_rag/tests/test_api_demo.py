import json
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from contextprime.api.main import app
from contextprime.core.config import get_settings


@pytest.fixture(scope="module")
def api_client():
    settings = get_settings()
    original_require_access_token = settings.security.require_access_token

    settings.security.require_access_token = False
    try:
        with TestClient(app) as client:
            yield client
    finally:
        settings.security.require_access_token = original_require_access_token


def test_status_endpoint_reports_semantic_flag(api_client):
    response = api_client.get("/api/status")
    assert response.status_code == 200

    data = response.json()
    processing_details = data["services"]["processing"]["details"]

    assert "documents_processed" in processing_details
    assert "semantic_chunking" in processing_details
    semantic = processing_details["semantic_chunking"]
    assert "available" in semantic


def test_upload_and_retrieve_document(api_client):
    repo_root = Path(__file__).resolve().parents[1]
    sample_file = repo_root / "data" / "samples" / "sample_text.txt"
    assert sample_file.exists(), "Sample document missing"

    settings = {
        "enable_ocr": False,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "chunking_method": "structure",
        "extract_entities": False,
        "build_raptor": False,
    }

    with sample_file.open("rb") as fh:
        response = api_client.post(
            "/api/documents",
            files={"file": (sample_file.name, fh, "text/plain")},
            data={"settings": json.dumps(settings)},
        )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["success"] is True
    document = payload["document"]
    assert document["num_chunks"] > 0

    doc_id = document["id"]

    list_response = api_client.get("/api/documents")
    assert list_response.status_code == 200
    list_payload = list_response.json()
    assert any(doc["id"] == doc_id for doc in list_payload["documents"])

    detail_response = api_client.get(f"/api/documents/{doc_id}")
    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["success"] is True
    assert detail_payload["document"]["metadata"]["chunking_method"] in {"structure", "semantic"}
    assert len(detail_payload["chunks"]) == detail_payload["document"]["num_chunks"]


def test_upload_rejects_file_above_limit(api_client):
    settings = get_settings()
    original_limit = settings.document_processing.max_file_size_mb
    settings.document_processing.max_file_size_mb = 1

    payload = {
        "enable_ocr": False,
        "chunk_size": 500,
        "chunk_overlap": 50,
        "chunking_method": "structure",
        "extract_entities": False,
        "build_raptor": False,
    }

    file_bytes = b"a" * ((1024 * 1024) + 1)
    try:
        response = api_client.post(
            "/api/documents",
            files={"file": ("too-large.txt", io.BytesIO(file_bytes), "text/plain")},
            data={"settings": json.dumps(payload)},
        )
    finally:
        settings.document_processing.max_file_size_mb = original_limit

    assert response.status_code == 413
