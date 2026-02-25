"""
Web Crawler Engine wrapping crawl4ai.

This module provides the asynchronous crawler adapter for Contextprime's
Web Ingestion Pipeline. It handles:
- URL fetching with headless browser (Playwright)
- Dynamic content rendering (JS)
- Metadata and link extraction
- Error handling and result standardization
"""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    try:
        from crawl4ai.extraction_strategy import LLMExtractionStrategy
    except ImportError:
        LLMExtractionStrategy = None  # type: ignore
    _CRAWL4AI_AVAILABLE = True
except ImportError:
    AsyncWebCrawler = None  # type: ignore
    BrowserConfig = None  # type: ignore
    CrawlerRunConfig = None  # type: ignore
    LLMExtractionStrategy = None  # type: ignore
    _CRAWL4AI_AVAILABLE = False
    logger.warning("crawl4ai not installed. Web crawling features will be disabled.")


@dataclass
class WebCrawlResult:
    """Standardized result from a web crawl."""
    url: str
    title: str
    markdown: str
    html: str
    crawled_at: str
    links: List[str] = field(default_factory=list)
    media: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None


class WebCrawler:
    """
    Engine for crawling websites and extracting structured content.
    """

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._browser_config = BrowserConfig(headless=headless) if BrowserConfig is not None else None
        if not _CRAWL4AI_AVAILABLE:
            logger.error("Attempted to initialize WebCrawler but crawl4ai is missing.")

    @property
    def is_available(self) -> bool:
        """Check if crawling dependencies are available."""
        return _CRAWL4AI_AVAILABLE

    async def crawl_url(self, url: str) -> WebCrawlResult:
        """
        Crawl a single URL and return structured result.

        Args:
            url: The URL to crawl.

        Returns:
            WebCrawlResult containing markdown and metadata.
        """
        if not self.is_available:
            return WebCrawlResult(
                url=url, title="", markdown="", html="", crawled_at="", 
                success=False, error="crawl4ai not installed"
            )

        logger.info(f"Starting crawl for: {url}")
        
        try:
            # crawl4ai 0.8.x pattern: BrowserConfig + CrawlerRunConfig
            run_cfg = CrawlerRunConfig()
            async with AsyncWebCrawler(config=self._browser_config) as crawler:
                result = await crawler.arun(url=url, config=run_cfg)
                
                if not result.success:
                    logger.error(f"Crawl failed for {url}: {result.error_message}")
                    return WebCrawlResult(
                        url=url,
                        title="",
                        markdown="",
                        html="",
                        crawled_at=datetime.now().isoformat(),
                        success=False,
                        error=result.error_message
                    )

                # Extract links robustly
                links = []
                if hasattr(result, "links"):
                    if isinstance(result.links, dict):
                        # Handle {'internal': [...], 'external': [...]}
                        for link_list in result.links.values():
                            if isinstance(link_list, list):
                                for link_obj in link_list:
                                    if isinstance(link_obj, dict) and link_obj.get("href"):
                                        links.append(link_obj.get("href"))
                    elif isinstance(result.links, list):
                        # Handle flat list [link_dict, ...]
                        for link_obj in result.links:
                            if isinstance(link_obj, dict) and link_obj.get("href"):
                                links.append(link_obj.get("href"))
                
                return WebCrawlResult(
                    url=url,
                    title=result.metadata.get("title", "Untitled"),
                    markdown=result.markdown or "",
                    html=result.html or "",
                    crawled_at=datetime.now().isoformat(),
                    links=links,
                    media=result.media if hasattr(result, "media") else [],
                    metadata=result.metadata or {},
                    success=True
                )

        except Exception as e:
            logger.exception(f"Exception during crawl of {url}")
            return WebCrawlResult(
                url=url,
                title="",
                markdown="",
                html="",
                crawled_at=datetime.now().isoformat(),
                success=False,
                error=str(e)
            )
