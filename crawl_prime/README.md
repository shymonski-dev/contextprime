# CrawlPrime

**Web RAG — crawl, index, and query live web content.**

CrawlPrime is a standalone web RAG built on top of **ContextPrime** shared utilities.
It transforms dynamic websites into structured, queryable knowledge — using the same
DocTags hierarchy, vector storage, and hybrid retrieval that powers ContextPrime
document RAG.

## Architecture

```
URL ──► WebCrawler (crawl4ai + Playwright)
              │
              ▼
        WebDocTagsMapper          ← ContextPrime shared utility
              │
              ▼
        WebIngestionPipeline      ← ContextPrime shared utility
         │            │
         ▼            ▼
      Neo4j        Qdrant (vectors)
                      │
                      ▼
              HybridRetriever     ← ContextPrime shared utility
                      │
                      ▼
              AgenticPipeline     ← ContextPrime shared utility
                      │
                      ▼
               LLM synthesis → Answer
```

CrawlPrime owns: `pipeline.py` (orchestrator), `planner.py` (URL-aware step planning),
`api.py` (FastAPI app), `main.py` (CLI).

ContextPrime provides: WebCrawler, WebDocTagsMapper, WebIngestionPipeline,
HybridRetriever, AgenticPipeline, QdrantManager, Neo4jManager.

## Installation

```bash
# 1. Ensure ContextPrime (doctags_rag) is in the parent directory
# 2. Install CrawlPrime dependencies (includes crawl4ai 0.8.x)
pip install -r requirements.txt
playwright install chromium

# 3. Start services (Qdrant + Neo4j)
docker-compose -f ../docker-compose.yml up -d
```

## Quick Start

### Python API

```python
import asyncio
from src.crawl_prime.pipeline import CrawlPrimePipeline

async def main():
    cp = CrawlPrimePipeline(collection="my_web_kb", enable_synthesis=True)

    # Crawl and index a site
    report = await cp.ingest("https://example.com")
    print(f"Indexed {report.chunks_ingested} chunks")

    # Query it
    result = await cp.query("What services does the site offer?")
    print(result.answer)
    cp.close()

asyncio.run(main())
```

### CLI

```bash
python -m src.crawl_prime.main --url "https://example.com" --output data/output
```

### REST API

```bash
uvicorn src.crawl_prime.api:app --reload --port 8001
```

```bash
# Ingest a URL
curl -X POST http://localhost:8001/ingest \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'

# Query
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What services does the site offer?"}'
```

## Environment Variables

```bash
OPENAI_API_KEY=sk-...             # Required for LLM synthesis
QDRANT_HOST=localhost              # Qdrant host (default: localhost)
QDRANT_PORT=6333                   # Qdrant port (default: 6333)

# Optional: route LLM calls through OpenRouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1
```

## crawl4ai 0.8.x API

CrawlPrime requires `crawl4ai>=0.8.0`. The 0.8.x release changed the
`AsyncWebCrawler` constructor — the `BrowserConfig` object must be passed
as `config=`, not `browser_config=`:

```python
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

async with AsyncWebCrawler(config=BrowserConfig(headless=True)) as crawler:
    result = await crawler.arun(url=url, config=CrawlerRunConfig())
```

## Testing

```bash
# Unit tests (always — no Docker needed)
pytest tests/test_processing.py -v

# Integration tests (requires Docker + Playwright + OPENAI_API_KEY)
pytest tests/integration/test_pipeline_e2e.py -v -m integration

# Real-web smoke test (requires live internet + Docker + OPENAI_API_KEY)
pytest tests/integration/test_real_web.py -v -m real_web
```

## Relationship to ContextPrime

| Concern | ContextPrime | CrawlPrime |
|---|---|---|
| PDF / DOCX / HTML file ingestion | ✓ | — |
| Document query pipeline | ✓ | — |
| Web crawling + indexing | utility only | ✓ (orchestrates) |
| Web query pipeline | — | ✓ |
| URL detection in planner | — | ✓ |
| WebCrawler, WebDocTagsMapper | ✓ shared | imports |
| WebIngestionPipeline | ✓ shared | imports |
| HybridRetriever, AgenticPipeline | ✓ owns | imports |
