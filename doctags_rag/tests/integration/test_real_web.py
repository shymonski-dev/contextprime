"""
Tier 3 — live real-web smoke test against worldwidecloud.io.

Exercises the full pipeline against a real public URL:
  crawl4ai (live network) → DocTags → chunks → OpenAI embeddings → Qdrant
  → HybridRetriever (vector-only) → AgenticPipeline (synthesis=True)
  → grounded answers asserting on stable, factual site content

Requirements:
  - Live internet access
  - Qdrant running at localhost:6333
  - Neo4j running at bolt://localhost:7687
  - OPENAI_API_KEY set

Run with:
  venv/bin/python -m pytest tests/integration/test_real_web.py -v -m real_web
"""

import os
import uuid
import pytest

from .conftest import requires_services, _qdrant_reachable
from src.agents.agentic_pipeline import AgenticPipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.qdrant_manager import QdrantManager
from src.pipelines.web_ingestion import WebIngestionPipeline
from src.pipelines.document_ingestion import DocumentIngestionPipeline, DocumentIngestionConfig
from src.core.config import QdrantConfig


pytestmark = pytest.mark.real_web

_SITE_URL = "https://worldwidecloud.io"

_requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping real-web synthesis test",
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_collection_name():
    """Unique Qdrant collection for this test module."""
    return f"test_real_web_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def cleanup_real_collection(real_collection_name):
    """Delete the real-web test collection after the module."""
    yield
    if not _qdrant_reachable():
        return
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host="localhost", port=6333, check_compatibility=False)
        cols = [c.name for c in client.get_collections().collections]
        if real_collection_name in cols:
            client.delete_collection(real_collection_name)
    except Exception:
        pass


@pytest.fixture(scope="module")
def ingested_collection(real_collection_name, cleanup_real_collection):
    """
    Module-scoped fixture: crawl worldwidecloud.io once and store vectors.
    All query tests in this module share the same ingested collection.
    """
    import asyncio

    ingestion_cfg = DocumentIngestionConfig(
        qdrant_collection=real_collection_name,
        create_qdrant_collection=True,
    )
    storage_pipeline = DocumentIngestionPipeline(config=ingestion_cfg)
    web_pipeline = WebIngestionPipeline(
        document_ingestion_pipeline=storage_pipeline,
    )

    report = asyncio.get_event_loop().run_until_complete(
        web_pipeline.ingest_url(_SITE_URL)
    )

    assert report.chunks_ingested > 0, (
        f"Ingestion of {_SITE_URL} produced no chunks. "
        f"Failed docs: {report.failed_documents}"
    )
    return real_collection_name


def _build_pipeline(collection_name: str) -> AgenticPipeline:
    """Wire a HybridRetriever + AgenticPipeline for the given collection."""
    qdrant_cfg = QdrantConfig(
        host="localhost",
        port=6333,
        collection_name=collection_name,
    )
    retriever = HybridRetriever(
        qdrant_manager=QdrantManager(config=qdrant_cfg),
        vector_weight=1.0,
        graph_weight=0.0,
    )
    return AgenticPipeline(
        retrieval_pipeline=retriever,
        enable_synthesis=True,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

@requires_services
@_requires_openai
@pytest.mark.asyncio
async def test_service_area_query(ingested_collection):
    """
    Query: London boroughs served.
    The site lists Westminster, Camden, Southwark, Lambeth, Wandsworth etc.
    """
    pipeline = _build_pipeline(ingested_collection)
    result = await pipeline.process_query(
        "Which London boroughs and areas does World Wide Cloud serve?"
    )

    assert result.answer and len(result.answer) > 30
    assert "Retrieved content" not in result.answer

    answer_lower = result.answer.lower()
    london_boroughs = {"westminster", "camden", "southwark", "lambeth",
                       "wandsworth", "london", "islington"}
    matched = london_boroughs & set(answer_lower.split())
    assert matched, (
        f"Answer does not mention any London boroughs.\n"
        f"Expected one of: {london_boroughs}\nAnswer: {result.answer[:400]}"
    )


@requires_services
@_requires_openai
@pytest.mark.asyncio
async def test_technology_history_query(ingested_collection):
    """
    Query: Amazon developer history.
    The site states 'Since 2007 — Amazon Developer'.
    2007 is a stable, unlikely-to-change fact.
    """
    pipeline = _build_pipeline(ingested_collection)
    result = await pipeline.process_query(
        "When did World Wide Cloud become an Amazon developer?"
    )

    assert result.answer and len(result.answer) > 20
    assert "Retrieved content" not in result.answer

    assert "2007" in result.answer, (
        f"Answer should contain the founding year '2007'.\n"
        f"Answer: {result.answer[:400]}"
    )


@requires_services
@_requires_openai
@pytest.mark.asyncio
async def test_faq_differentiator_query(ingested_collection):
    """
    Query: What makes WWC different (mirrors an FAQ entry verbatim).
    Tests that FAQ Q&A pairs are retrievable and that the answer
    references multiple technology partners from the page.
    """
    pipeline = _build_pipeline(ingested_collection)
    result = await pipeline.process_query(
        "What makes World Wide Cloud's AI consulting different from other providers?"
    )

    assert result.answer and len(result.answer) > 30
    assert "Retrieved content" not in result.answer

    answer_lower = result.answer.lower()
    differentiators = {"2008", "ibm", "quantum", "amazon", "apple", "automation",
                       "small", "enterprise", "london"}
    matched = differentiators & set(answer_lower.split())
    assert len(matched) >= 2, (
        f"Answer should reference at least 2 differentiating terms from the site.\n"
        f"Expected 2+ of: {differentiators}\nMatched: {matched}\n"
        f"Answer: {result.answer[:400]}"
    )
