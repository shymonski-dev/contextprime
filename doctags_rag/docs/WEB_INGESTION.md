# Web Ingestion System (ContextWeb)

## Overview

The Web Ingestion System, also known as **ContextWeb**, enables Contextprime to crawl, process, and index live web content. It treats "The Web as a Structured Document," leveraging `crawl4ai` to preserve the semantic hierarchy of web pages and integrate them directly into the RAG knowledge graph.

## Features

- **Dynamic Crawling**: Integration with `crawl4ai` for headless browser rendering, handling JavaScript-heavy sites.
- **Structure Preservation**: Maps HTML semantic markers (Headers `#`, `##`, Lists, Tables) directly into the `DocTags` hierarchy.
- **Web-to-Graph mapping**: Automatically extracts internal and external links, persisting them as `(:Page)-[:LINKS_TO]->(:Page)` relationships in Neo4j.
- **Agentic Integration**: The `ExecutionAgent` can now actively "browse" URLs on-demand to acquire fresh knowledge.

## Architecture

```
URL ───► WebCrawler (Playwright) ───► Clean Markdown 
                                          │
                                          ▼
Neo4j ◄─── WebDocTagsMapper ◄─────── DocTags Hierarchy
  │               │                       │
  │               ▼                       ▼
  └──────► Ingestion Pipeline ──────► Qdrant (Vectors)
```

### Components

1. **WebCrawler (`src/processing/web/crawler.py`)**: Wraps `crawl4ai`. Manages browser sessions and fetches raw content.
2. **WebDocTagsMapper (`src/processing/web/mapper.py`)**: Translates Markdown into Contextprime's native `DocTagsDocument`.
3. **WebIngestionPipeline (`src/pipelines/web_ingestion.py`)**: Orchestrates the flow from URL to database storage.

## Usage

### Direct Ingestion

```python
import asyncio
from src.pipelines.web_ingestion import WebIngestionPipeline

async def main():
    pipeline = WebIngestionPipeline()
    report = await pipeline.ingest_url("https://fastapi.tiangolo.com/")
    
    print(f"Ingested {report.chunks_ingested} chunks.")
    print(f"Neo4j Documents: {report.neo4j_documents}")
    pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())
```

### Agent-Driven Research

Agents can now request web ingestion as part of a multi-step research plan. The `ExecutionAgent` identifies the `web_ingestion` capability and triggers the pipeline.

```json
{
  "step_id": "step_1",
  "step_type": "web_ingestion",
  "parameters": {
    "url": "https://example.com/news/latest-regulation"
  }
}
```

## Configuration

Add the following to your `config/config.yaml` or environment:

```yaml
# Web Ingestion Settings
web_ingestion:
  headless: true
  chunk_size: 1000
  chunk_overlap: 200
```

Environment variables:
- `CRAWL4AI_HEADLESS=true`
- `PLAYWRIGHT_BROWSER_TYPE=chromium`

## Dependencies

The web module requires additional dependencies:
- `crawl4ai`
- `playwright`
- `python-slugify`

Run `playwright install chromium` after installing requirements.

## Troubleshooting

- **"Just a moment..." (Cloudflare)**: Some sites may block the headless crawler. Enable `stealth_mode` in the `WebCrawler` configuration if available in `crawl4ai`.
- **Link Extraction Errors**: Ensure the target site uses standard `<a>` tags for navigation.
