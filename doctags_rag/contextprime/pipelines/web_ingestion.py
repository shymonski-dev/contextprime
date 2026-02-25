"""
Web Ingestion Pipeline.

Orchestrates the live crawling, processing, and indexing of web content.
Reuses the core DocumentIngestionPipeline logic for storage (Qdrant/Neo4j).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

from loguru import logger

from ..core.config import LegalMetadataConfig
from ..processing.web.crawler import WebCrawler
from ..processing.web.mapper import WebDocTagsMapper
from ..processing.chunker import StructurePreservingChunker
from ..processing.pipeline import ProcessingResult, ProcessingStage
from ..processing.document_parser import ParsedDocument
from ..pipelines.document_ingestion import DocumentIngestionPipeline, IngestionReport, DocumentIngestionConfig


@dataclass
class WebIngestionConfig:
    """Configuration for web ingestion."""
    headless: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    ingestion_config: Optional[DocumentIngestionConfig] = None


class WebIngestionPipeline:
    """
    Pipeline for crawling URLs and ingesting them into the RAG system.
    """

    def __init__(
        self,
        config: Optional[WebIngestionConfig] = None,
        document_ingestion_pipeline: Optional[DocumentIngestionPipeline] = None
    ):
        self.config = config or WebIngestionConfig()
        
        # Components
        self.crawler = WebCrawler(headless=self.config.headless)
        self.mapper = WebDocTagsMapper()
        self.chunker = StructurePreservingChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        # Reuse existing storage pipeline
        if document_ingestion_pipeline:
            self.storage_pipeline = document_ingestion_pipeline
        else:
            self.storage_pipeline = DocumentIngestionPipeline(
                config=self.config.ingestion_config
            )

    async def ingest_url(
        self, 
        url: str, 
        legal_metadata: Optional[LegalMetadataConfig] = None
    ) -> IngestionReport:
        """
        Crawl, process, and ingest a single URL.

        Args:
            url: The URL to ingest.
            legal_metadata: Optional legal metadata overrides.

        Returns:
            IngestionReport summary.
        """
        if not self.crawler.is_available:
            logger.error("Web crawling is not available (crawl4ai missing).")
            return IngestionReport(failed_documents=[url])

        start_time = time.time()
        logger.info(f"Ingesting URL: {url}")

        # 1. Crawl
        crawl_result = await self.crawler.crawl_url(url)
        if not crawl_result.success:
            logger.error(f"Crawl failed for {url}: {crawl_result.error}")
            return IngestionReport(failed_documents=[url])

        # 2. Map to DocTags
        try:
            doctags_doc = self.mapper.map_to_doctags(crawl_result)
            
            # Construct a synthetic ParsedDocument for compatibility
            parsed_doc = ParsedDocument(
                text=crawl_result.markdown,
                elements=[], # Elements are already in doctags_doc
                metadata={
                    "source": "web",
                    "url": url,
                    "title": crawl_result.title,
                    "crawled_at": crawl_result.crawled_at
                },
                structure={}
            )
        except Exception as e:
            logger.error(f"Mapping failed for {url}: {e}")
            return IngestionReport(failed_documents=[url])

        # 3. Chunk
        chunks = self.chunker.chunk_document(doctags_doc)
        logger.info(f"Created {len(chunks)} chunks from {url}")

        # 4. Construct ProcessingResult
        processing_result = ProcessingResult(
            file_path=Path(f"web://{url}"), # Virtual path
            success=True,
            stage=ProcessingStage.COMPLETED,
            parsed_doc=parsed_doc,
            doctags_doc=doctags_doc,
            chunks=chunks,
            processing_time=time.time() - start_time,
            metadata={
                "source_type": "web",
                "http_status": 200
            }
        )

        # 5. Ingest into Storage (Neo4j/Qdrant)
        # We delegate this to the synchronous storage pipeline.
        # Since ingest_processing_results is CPU-bound (embeddings) or I/O bound (DB),
        # and not async, we call it directly. Ideally, we'd run this in a thread if blocking.
        report = await asyncio.to_thread(
            self.storage_pipeline.ingest_processing_results,
            [processing_result],
            legal_metadata=legal_metadata,
        )
        
        return report

    def close(self):
        """Close resources."""
        self.storage_pipeline.close()
