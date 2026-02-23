"""
Core Web Crawler Engine wrapping crawl4ai.
"""

import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

# Conditional import to allow structure setup without installing heavy deps immediately
try:
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.extraction_strategy import LLMExtractionStrategy
except ImportError:
    AsyncWebCrawler = None  # type: ignore
    logger.warning("crawl4ai not installed. Run 'pip install crawl4ai'")


@dataclass
class CrawlResult:
    """Standardized result from a web crawl."""
    url: str
    title: str
    markdown: str
    html: str
    crawled_at: str
    links: List[str]
    media: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class WebCrawlerEngine:
    """
    Engine for crawling websites and extracting structured content.
    """

    def __init__(self, headless: bool = True):
        self.headless = headless
        if AsyncWebCrawler is None:
            raise ImportError("crawl4ai dependency is missing")

    async def crawl_url(self, url: str) -> CrawlResult:
        """
        Crawl a single URL and return structured result.
        """
        logger.info(f"Starting crawl for: {url}")
        
        try:
            async with AsyncWebCrawler(verbose=True, headless=self.headless) as crawler:
                result = await crawler.arun(url=url)
                
                if not result.success:
                    logger.error(f"Crawl failed for {url}: {result.error_message}")
                    return CrawlResult(
                        url=url,
                        title="",
                        markdown="",
                        html="",
                        crawled_at=datetime.now().isoformat(),
                        links=[],
                        media=[],
                        metadata={},
                        success=False,
                        error=result.error_message
                    )

                # Extract useful metadata
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
                        # Handle old format [link_dict, ...]
                        for link_obj in result.links:
                            if isinstance(link_obj, dict) and link_obj.get("href"):
                                links.append(link_obj.get("href"))
                
                return CrawlResult(
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
            return CrawlResult(
                url=url,
                title="",
                markdown="",
                html="",
                crawled_at=datetime.now().isoformat(),
                links=[],
                media=[],
                metadata={},
                success=False,
                error=str(e)
            )
