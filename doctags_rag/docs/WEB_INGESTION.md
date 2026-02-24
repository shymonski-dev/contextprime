# Web Ingestion System (ContextWeb)

## Overview

The Web Ingestion System, also known as **ContextWeb**, enables Contextprime to crawl, process, and index live web content. It treats "The Web as a Structured Document," leveraging `crawl4ai` to preserve the semantic hierarchy of web pages and integrate them directly into the RAG knowledge graph.

## Features

- **Dynamic Crawling**: Integration with `crawl4ai` (0.8.x) for headless browser rendering, handling JavaScript-heavy sites.
- **Structure Preservation**: Maps HTML semantic markers (Headers `#`, `##`, Lists, Tables) directly into the `DocTags` hierarchy.
- **Web-to-Graph mapping**: Automatically extracts internal and external links, persisting them as `(:Page)-[:LINKS_TO]->(:Page)` relationships in Neo4j.
- **Agentic Integration**: The `ExecutionAgent` can actively "browse" URLs on-demand to acquire fresh knowledge, with web ingestion steps correctly sequenced before retrieval in the dependency graph.

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

1. **WebCrawler (`src/processing/web/crawler.py`)**: Wraps `crawl4ai` 0.8.x. Manages browser sessions using `BrowserConfig` and `CrawlerRunConfig`.
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

The `PlanningAgent` automatically detects URLs in queries and inserts a `WEB_INGESTION` step that is sequenced *before* retrieval steps. The `ExecutionAgent` routes this step to `WebIngestionPipeline.ingest_url()`.

```json
{
  "step_id": "step_0",
  "step_type": "web_ingestion",
  "parameters": {
    "url": "https://example.com/news/latest-regulation"
  },
  "dependencies": []
}
```

Retrieval steps automatically declare a dependency on the web ingestion step, ensuring content is available before any vector search runs:

```json
{
  "step_id": "step_1",
  "step_type": "retrieval",
  "dependencies": ["step_0"]
}
```

### Agentic Pipeline with Synthesis

```python
import asyncio
from src.agents.agentic_pipeline import AgenticPipeline
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.qdrant_manager import QdrantManager
from src.core.config import QdrantConfig

async def main():
    qdrant_cfg = QdrantConfig(host="localhost", port=6333, collection_name="my_collection")
    retriever = HybridRetriever(
        qdrant_manager=QdrantManager(config=qdrant_cfg),
        vector_weight=1.0,
        graph_weight=0.0,
    )
    pipeline = AgenticPipeline(
        retrieval_pipeline=retriever,
        enable_synthesis=True,   # Forces LLM synthesis on; reads OPENAI_API_KEY
    )
    result = await pipeline.process_query(
        "Summarise https://example.com/report"
    )
    print(result.answer)

asyncio.run(main())
```

## Configuration

Add the following to your environment:

```bash
# crawl4ai headless mode (default: true)
CRAWL4AI_HEADLESS=true

# LLM synthesis (required for answer generation)
OPENAI_API_KEY=sk-...

# Optional: route synthesis through OpenRouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

## Dependencies

The web module requires:
- `crawl4ai>=0.8.0,<0.9.0`
- `playwright`
- `python-slugify`

After installing requirements, install the Playwright browser:

```bash
pip install -r requirements.txt
playwright install chromium
```

### crawl4ai 0.8.x API Notes

The 0.8.x release changed the `AsyncWebCrawler` constructor:

```python
# crawl4ai 0.8.x
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

browser_cfg = BrowserConfig(headless=True)
run_cfg = CrawlerRunConfig()

async with AsyncWebCrawler(config=browser_cfg) as crawler:   # <-- config=, not browser_config=
    result = await crawler.arun(url=url, config=run_cfg)
```

The old `verbose=True, headless=True` kwargs to `AsyncWebCrawler` are no longer accepted.

## Testing

The web ingestion stack is covered across three test tiers:

### Tier 1 — Unit / wiring (always runs, no Docker)

```bash
python -m pytest tests/test_agentic_web_wiring.py tests/test_web_ingestion.py -v
```

Tests: `WebIngestionPipeline` constructor, `ExecutionAgent` step routing, `PlanningAgent` URL detection and dependency ordering (WEB_INGESTION → RETRIEVAL → GRAPH_QUERY).

### Tier 2 — Integration (requires Docker: Qdrant + Neo4j, and Playwright)

```bash
python -m pytest tests/integration/test_web_e2e.py tests/integration/test_full_e2e.py \
    -v -m integration
```

Tests: live crawl of a local test page → Qdrant → `HybridRetriever` → `AgenticPipeline` → grounded answer.

### Tier 3 — Real-web smoke test (requires live internet + Docker + OPENAI_API_KEY)

```bash
python -m pytest tests/integration/test_real_web.py -v -m real_web
```

Tests: crawls a live public website (worldwidecloud.io), verifies that factual content (service areas, technology history, FAQ differentiators) is retrievable and synthesised correctly by GPT.

## Troubleshooting

- **"Just a moment..." (Cloudflare)**: Some sites block headless crawlers. The `stealth_mode` option in `crawl4ai` may help on supported versions.
- **`TypeError: got multiple values for keyword argument 'browser_config'`**: You are passing `browser_config=` to `AsyncWebCrawler`. The correct kwarg in 0.8.x is `config=`.
- **Link Extraction Errors**: Ensure the target site uses standard `<a>` tags for navigation.
- **`POST /api/documents/web` returns 503**: `crawl4ai` is not installed in the current environment. Install it and run `playwright install chromium`.
