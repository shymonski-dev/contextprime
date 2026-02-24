"""
Tier 2 — full end-to-end pipeline test.

Exercises the unbroken path:
  crawl4ai → markdown → DocTags → chunks → OpenAI embeddings → Qdrant
                                                                    ↓
                                        query → embed → vector search
                                                                    ↓
                                        HybridRetriever → results
                                                                    ↓
                                        gpt-4o-mini synthesis → grounded answer

Requirements:
  - Qdrant running at localhost:6333
  - Neo4j running at bolt://localhost:7687
  - OPENAI_API_KEY set
  - crawl4ai + playwright chromium installed

Run with:
  venv/bin/python -m pytest tests/integration/test_full_e2e.py -v -m integration
"""

import os
import pytest

from .conftest import requires_services
from src.agents.agentic_pipeline import AgenticPipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.qdrant_manager import QdrantManager
from src.core.config import QdrantConfig


pytestmark = pytest.mark.integration

_requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping synthesis test",
)


def _make_web_pipeline(test_collection_name: str):
    """Re-use the helper from test_web_e2e to avoid duplication."""
    from tests.integration.test_web_e2e import _make_pipeline
    return _make_pipeline(test_collection_name)


def _make_retriever(collection_name: str) -> HybridRetriever:
    """Build a HybridRetriever aimed at a specific Qdrant collection.

    Uses pure vector search (graph_weight=0) because the graph executor is
    still a stub — this keeps the test deterministic and honest.
    """
    qdrant_cfg = QdrantConfig(
        host="localhost",
        port=6333,
        collection_name=collection_name,
    )
    return HybridRetriever(
        qdrant_manager=QdrantManager(config=qdrant_cfg),
        vector_weight=1.0,
        graph_weight=0.0,
    )


# ── Test ──────────────────────────────────────────────────────────────────────

@requires_services
@_requires_openai
@pytest.mark.asyncio
async def test_ingest_then_query_returns_grounded_answer(
    test_page_url, test_collection_name, cleanup_test_collection
):
    """
    Full pipeline: crawl URL → Qdrant → HybridRetriever → AgenticPipeline
    → answer that references actual page content.

    Assertions:
    1. Ingest succeeds and writes at least one chunk.
    2. Retrieval returns results from Qdrant (not the old simulation strings).
    3. The synthesised answer contains terms that appear on the test page.
    4. No simulated "Retrieved content N" strings leaked into the answer.
    """
    # ── Phase 1: Ingest ───────────────────────────────────────────────
    web_pipeline = _make_web_pipeline(test_collection_name)
    report = await web_pipeline.ingest_url(test_page_url)
    assert report.chunks_ingested > 0, (
        f"Ingest failed — no chunks stored. Failed docs: {report.failed_documents}"
    )

    # ── Phase 2: Wire retriever to the same test collection ───────────
    retriever = _make_retriever(test_collection_name)

    # ── Phase 3: Build AgenticPipeline with synthesis enabled ─────────
    pipeline = AgenticPipeline(
        retrieval_pipeline=retriever,
        web_pipeline=web_pipeline,
        enable_synthesis=True,
    )

    # ── Phase 4: Query for content we know is on the page ─────────────
    # The test page (test_page.html) contains:
    #   "Do not operate the Acme Widget near open water."
    #   "Always wear protective gloves when handling the cutting blade."
    #   "In case of emergency, press the red STOP button immediately."
    result = await pipeline.process_query(
        "What safety precautions are required when using the Acme Widget?"
    )

    # ── Phase 5: Structural assertions ───────────────────────────────
    assert result.answer, "Expected a non-empty answer"
    assert len(result.answer) > 30, f"Answer suspiciously short: {result.answer!r}"

    # ── Phase 6: Canary — simulation must not have fired ─────────────
    assert "Retrieved content" not in result.answer, (
        "Answer contains simulated retrieval content — real retrieval failed silently.\n"
        f"Answer: {result.answer[:300]}"
    )
    for r in result.results:
        assert "Retrieved content" not in r.get("content", ""), (
            f"Simulated content leaked into retrieval results: {r}"
        )

    # ── Phase 7: Grounding — answer must reference actual page content ─
    # At least one of these terms must appear; they are all in test_page.html.
    page_keywords = {"acme", "safety", "protective", "glove", "stop", "blade",
                     "water", "emergency", "widget"}
    answer_words = set(result.answer.lower().split())
    matched = page_keywords & answer_words
    assert matched, (
        f"Answer does not reference any known page content.\n"
        f"Expected one of: {page_keywords}\n"
        f"Answer: {result.answer[:400]}"
    )

    # ── Phase 8: Retrieval results came from Qdrant ───────────────────
    assert result.results, "Expected non-empty retrieval results from Qdrant"
