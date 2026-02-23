"""
CLI Entry point for CrawlPrime.
"""

import asyncio
import argparse
import json
import os
from pathlib import Path
from loguru import logger

from .crawler.engine import WebCrawlerEngine
from .processing.mapper import DocTagsMapper

async def main():
    parser = argparse.ArgumentParser(description="CrawlPrime: Web to DocTags Crawler")
    parser.add_argument("--url", required=True, help="URL to crawl")
    parser.add_argument("--output", default="data/output", help="Output directory")
    args = parser.parse_args()

    # Ensure output dir exists
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Crawl
    logger.info(f"Crawling {args.url}...")
    crawler = WebCrawlerEngine()
    result = await crawler.crawl_url(args.url)

    if not result.success:
        logger.error(f"Crawl failed: {result.error}")
        return

    # 2. Map
    logger.info("Mapping to DocTags format...")
    mapper = DocTagsMapper()
    doctags = mapper.map_to_doctags(result)

    # 3. Save
    safe_filename = "".join(x for x in result.title if x.isalnum() or x in "._- ") or "doc"
    filename = f"{safe_filename[:50]}.json"
    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doctags, f, indent=2, ensure_ascii=False)

    logger.success(f"Saved structured output to {output_path}")
    logger.info(f"Total Tags: {len(doctags['tags'])}")

if __name__ == "__main__":
    asyncio.run(main())
