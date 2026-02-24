"""
Tier 2 end-to-end integration tests for web ingestion.

Requirements:
  - Qdrant running at localhost:6333
  - Neo4j running at bolt://localhost:7687
  - OPENAI_API_KEY set (used for embeddings)
  - crawl4ai + playwright chromium installed

Run with:
  venv/bin/python -m pytest tests/integration/test_web_e2e.py -v -m integration
"""

import pytest

from .conftest import requires_services

from src.pipelines.document_ingestion import DocumentIngestionPipeline, DocumentIngestionConfig
from src.pipelines.web_ingestion import WebIngestionPipeline, WebIngestionConfig
from src.agents.planning_agent import StepType


def _make_pipeline(test_collection_name: str) -> WebIngestionPipeline:
    """Helper: build a WebIngestionPipeline wired to the test Qdrant collection."""
    ingestion_cfg = DocumentIngestionConfig(
        qdrant_collection=test_collection_name,
        create_qdrant_collection=True,
    )
    storage_pipeline = DocumentIngestionPipeline(config=ingestion_cfg)
    return WebIngestionPipeline(
        config=WebIngestionConfig(),
        document_ingestion_pipeline=storage_pipeline,
    )


# ── Marker ─────────────────────────────────────────────────────────────────────
pytestmark = pytest.mark.integration


# ── Test 1: Full ingest → Qdrant verify ───────────────────────────────────────

@requires_services
@pytest.mark.asyncio
async def test_ingest_url_stores_chunks_in_qdrant(test_page_url, test_collection_name, cleanup_test_collection):
    """
    ingest_url(test_page_url) → IngestionReport.chunks_ingested > 0
    AND the vectors are findable in Qdrant under the test collection.
    """
    pipeline = _make_pipeline(test_collection_name)
    report = await pipeline.ingest_url(test_page_url)

    assert report.chunks_ingested > 0, (
        f"Expected at least 1 chunk ingested, got {report.chunks_ingested}. "
        f"Failed docs: {report.failed_documents}"
    )
    assert test_page_url not in report.failed_documents

    # Verify vectors exist in Qdrant
    from qdrant_client import QdrantClient
    client = QdrantClient(host="localhost", port=6333, check_compatibility=False)
    count_result = client.count(collection_name=test_collection_name)
    assert count_result.count > 0, "No vectors found in Qdrant test collection after ingestion"


@requires_services
@pytest.mark.asyncio
async def test_ingest_url_content_is_retrievable(test_page_url, test_collection_name, cleanup_test_collection):
    """
    ingest_url → then scroll Qdrant and verify page content is in stored payloads.
    This test does its own ingest so it is order-independent.
    """
    from qdrant_client import QdrantClient

    # Ingest first (order-independent)
    pipeline = _make_pipeline(test_collection_name)
    report = await pipeline.ingest_url(test_page_url)
    assert report.chunks_ingested > 0, f"Ingest failed: {report.failed_documents}"

    # Now scroll Qdrant
    client = QdrantClient(host="localhost", port=6333, check_compatibility=False)
    points, _ = client.scroll(
        collection_name=test_collection_name,
        with_payload=True,
        limit=50,
    )
    assert points, "No points found in test collection after ingestion"

    all_text = " ".join(
        str(p.payload.get("text", "") or p.payload.get("content", ""))
        for p in points
    ).lower()

    # The test page contains "Acme Widget" — verify it was stored
    assert "acme" in all_text, (
        "Expected 'acme' in stored chunk text — content from test_page.html was not persisted"
    )


# ── Test 2: AgenticPipeline detects URL and triggers WEB_INGESTION step ───────

@requires_services
@pytest.mark.asyncio
async def test_agentic_pipeline_generates_web_ingestion_plan(test_page_url, test_collection_name, cleanup_test_collection):
    """
    AgenticPipeline.process_query("summarise {url}") builds a plan that
    includes a WEB_INGESTION step, executes it, and returns a non-empty answer.
    """
    from src.agents.agentic_pipeline import AgenticPipeline

    # Wire a real WebIngestionPipeline pointing at the test collection
    web_pipeline = _make_pipeline(test_collection_name)

    pipeline = AgenticPipeline(
        web_pipeline=web_pipeline,
        # retrieval_pipeline intentionally omitted — will fall back gracefully
    )

    query = f"summarise {test_page_url}"
    result = await pipeline.process_query(query)

    # The plan must contain a WEB_INGESTION step
    web_steps = [s for s in result.plan.steps if s.step_type == StepType.WEB_INGESTION]
    assert web_steps, "Expected a WEB_INGESTION step in the plan for a URL-bearing query"
    assert web_steps[0].parameters["url"] == test_page_url

    # The result must have a non-empty answer
    assert result.answer, "Expected a non-empty answer from AgenticPipeline"
    assert len(result.answer) > 10

    # The WEB_INGESTION execution result must show success
    web_exec = next(
        (r for r in result.execution_results if "web" in r.step_id or
         (r.results and r.results[0].get("url") == test_page_url)),
        None,
    )
    if web_exec is not None:
        assert web_exec.success, f"WEB_INGESTION step failed: {web_exec.error_message}"
