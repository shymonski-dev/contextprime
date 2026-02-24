"""
CrawlPrimePipeline — the central orchestrator for CrawlPrime web RAG.

Composes ContextPrime shared utilities:
  WebIngestionPipeline  — crawl + DocTags + chunk + embed + store
  HybridRetriever       — vector + lexical retrieval with RRF fusion
  AgenticPipeline       — multi-agent query processing with LLM synthesis

Usage::

    import asyncio
    from src.crawl_prime.pipeline import CrawlPrimePipeline

    async def main():
        cp = CrawlPrimePipeline(collection="my_web_kb")
        await cp.ingest("https://example.com")
        result = await cp.query("What services does the site offer?")
        print(result.answer)

    asyncio.run(main())
"""

import sys
from pathlib import Path

# Ensure ContextPrime is importable
_DOCTAGS_ROOT = Path(__file__).resolve().parents[3] / "doctags_rag"
if str(_DOCTAGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_DOCTAGS_ROOT))

from typing import Optional, Any
from loguru import logger

from src.pipelines.web_ingestion import WebIngestionPipeline, WebIngestionReport
from src.pipelines.document_ingestion import DocumentIngestionPipeline, DocumentIngestionConfig
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.qdrant_manager import QdrantManager
from src.agents.agentic_pipeline import AgenticPipeline, AgenticResult
from src.core.config import QdrantConfig


class CrawlPrimePipeline:
    """
    End-to-end web RAG pipeline.

    Wraps ContextPrime's shared utilities into a single, easy-to-use
    interface for crawling, indexing, and querying web content.
    """

    def __init__(
        self,
        collection: str = "crawlprime_default",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        enable_synthesis: bool = True,
    ):
        """
        Args:
            collection:       Qdrant collection name for web content.
            qdrant_host:      Qdrant host.
            qdrant_port:      Qdrant port.
            enable_synthesis: Enable LLM answer synthesis (requires OPENAI_API_KEY).
        """
        self.collection = collection

        # Ingestion pipeline
        ingestion_cfg = DocumentIngestionConfig(
            qdrant_collection=collection,
            create_qdrant_collection=True,
        )
        self._storage = DocumentIngestionPipeline(config=ingestion_cfg)
        self._web_ingestion = WebIngestionPipeline(
            document_ingestion_pipeline=self._storage,
        )

        # Retrieval pipeline
        qdrant_cfg = QdrantConfig(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection,
        )
        self._retriever = HybridRetriever(
            qdrant_manager=QdrantManager(config=qdrant_cfg),
            vector_weight=1.0,
            graph_weight=0.0,
        )

        # Agentic query pipeline (document RAG, no URL detection)
        self._agentic = AgenticPipeline(
            retrieval_pipeline=self._retriever,
            enable_synthesis=enable_synthesis,
        )

        logger.info(
            "CrawlPrimePipeline initialised (collection=%s, synthesis=%s)",
            collection,
            enable_synthesis,
        )

    async def ingest(self, url: str) -> WebIngestionReport:
        """
        Crawl a URL and index its content into Qdrant.

        Args:
            url: The URL to crawl and ingest.

        Returns:
            WebIngestionReport with chunks_ingested count and any failures.
        """
        logger.info("CrawlPrime ingesting: %s", url)
        report = await self._web_ingestion.ingest_url(url)
        logger.info(
            "Ingestion complete — %d chunks stored, %d failures",
            report.chunks_ingested,
            len(report.failed_documents),
        )
        return report

    async def query(
        self,
        text: str,
        max_iterations: int = 2,
        min_quality_threshold: float = 0.5,
    ) -> AgenticResult:
        """
        Query the indexed web content.

        Args:
            text:                  Natural language question.
            max_iterations:        Max agentic improvement iterations.
            min_quality_threshold: Minimum quality score to accept answer.

        Returns:
            AgenticResult with .answer and .results.
        """
        return await self._agentic.process_query(
            text,
            max_iterations=max_iterations,
            min_quality_threshold=min_quality_threshold,
        )

    def close(self) -> None:
        """Release resources."""
        try:
            self._storage.close()
        except Exception:
            pass
