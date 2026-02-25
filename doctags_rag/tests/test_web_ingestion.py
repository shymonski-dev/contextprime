import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path

from contextprime.pipelines.web_ingestion import WebIngestionPipeline, WebIngestionConfig
from contextprime.processing.web.crawler import WebCrawlResult
from contextprime.pipelines.document_ingestion import IngestionReport

class TestWebIngestionPipeline:
    """Tests for WebIngestionPipeline."""

    @pytest.fixture
    def mock_crawler(self):
        crawler = MagicMock()
        crawler.is_available = True
        crawler.crawl_url = AsyncMock()
        return crawler

    @pytest.fixture
    def pipeline(self, mock_crawler):
        # We need to mock the storage pipeline to avoid DB dependencies
        mock_storage = MagicMock()
        mock_storage.ingest_processing_results.return_value = IngestionReport(
            processed_documents=1,
            chunks_ingested=5
        )
        
        pipeline = WebIngestionPipeline(document_ingestion_pipeline=mock_storage)
        pipeline.crawler = mock_crawler
        return pipeline

    @pytest.mark.asyncio
    async def test_ingest_url_success(self, pipeline, mock_crawler):
        # Setup mock crawl result
        mock_crawler.crawl_url.return_value = WebCrawlResult(
            url="https://example.com",
            title="Example Title",
            markdown="# Example Title\n\nThis is a paragraph.",
            html="<html>...</html>",
            crawled_at="2026-02-23T12:00:00",
            success=True
        )
        
        report = await pipeline.ingest_url("https://example.com")
        
        assert report.processed_documents == 1
        assert report.chunks_ingested > 0
        assert len(report.failed_documents) == 0
        
        # Verify crawler was called
        mock_crawler.crawl_url.assert_called_once_with("https://example.com")

    @pytest.mark.asyncio
    async def test_ingest_url_crawl_failure(self, pipeline, mock_crawler):
        # Setup mock crawl failure
        mock_crawler.crawl_url.return_value = WebCrawlResult(
            url="https://example.com",
            title="",
            markdown="",
            html="",
            crawled_at="",
            success=False,
            error="404 Not Found"
        )
        
        report = await pipeline.ingest_url("https://example.com")
        
        assert len(report.failed_documents) == 1
        assert report.failed_documents[0] == "https://example.com"

    @pytest.mark.asyncio
    async def test_ingest_url_not_available(self, pipeline, mock_crawler):
        mock_crawler.is_available = False
        
        report = await pipeline.ingest_url("https://example.com")
        
        assert len(report.failed_documents) == 1
        assert "https://example.com" in report.failed_documents
