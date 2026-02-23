# CrawlPrime (ContextWeb)

**Structured Web Intelligence for RAG Systems**

CrawlPrime is a standalone module designed to transform dynamic websites into structured, LLM-ready "DocTags" documents. It leverages `crawl4ai` to preserve the semantic hierarchy of web pages (Headers, Sections, Tables) and exports them in a format compatible with the Contextprime RAG system.

## 🚀 Features

*   **Structure Preservation:** Maps HTML headers (#, ##) to semantic Sections and Articles.
*   **Dynamic Crawling:** Handles JavaScript-heavy sites using `crawl4ai` (Playwright).
*   **Graph-Ready:** Extracts internal/external links for knowledge graph construction.
*   **Legal/Compliance Focus:** Detects legal document structures (Articles, Clauses) in web content.

## 📦 Output Format

CrawlPrime outputs JSON files that match the **DocTags** schema:

```json
{
  "doc_id": "web_openai_tos_2024",
  "title": "Terms of Service",
  "source_url": "https://openai.com/policies/terms-of-use",
  "tags": [
    {"type": "title", "content": "Terms of Service"},
    {"type": "section", "content": "1. Registration"},
    {"type": "paragraph", "content": "You must be 13 years or older..."}
  ]
}
```

## 🛠️ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt
playwright install

# 2. Run the crawler
python -m src.crawl_prime.main --url "https://example.com"
```
