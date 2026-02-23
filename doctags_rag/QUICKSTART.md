# DocTags Processing Pipeline - Quick Start Guide

## Installation

```bash
cd doctags_rag
pip install -r requirements.txt
```

## Basic Usage

### Process a Single Document

```python
from src.processing import create_pipeline

# Create pipeline
pipeline = create_pipeline(
    chunk_size=1000,
    chunk_overlap=200
)

# Process document
result = pipeline.process_file("document.pdf")

if result.success:
    print(f"✓ Created {len(result.chunks)} chunks")

    # Access chunks
    for chunk in result.chunks:
        print(chunk.content)
else:
    print(f"✗ Error: {result.error}")
```

### Process Multiple Documents

```python
from pathlib import Path

# Find documents
files = list(Path("documents/").glob("*.pdf"))

# Process in batch
results = pipeline.process_batch(files)

# Check results
successful = sum(1 for r in results if r.success)
print(f"Processed {successful}/{len(files)} files")
```

### Save Outputs

```python
from src.processing import PipelineConfig, DocumentProcessingPipeline
from pathlib import Path

# Configure with output saving
config = PipelineConfig(
    save_intermediate=True,
    save_json=True,
    save_markdown=True,
    output_dir=Path("output/")
)

pipeline = DocumentProcessingPipeline(config)
result = pipeline.process_file("document.pdf")
```

## Supported Formats

- ✓ PDF (text-based and scanned)
- ✓ DOCX / DOC
- ✓ HTML
- ✓ TXT / MD
- ✓ Images (PNG, JPG, JPEG)

## Output Formats

```python
from src.processing import DocTagsConverter

# Get DocTags document
doctags = result.doctags_doc

# Convert to different formats
markdown = DocTagsConverter.to_markdown(doctags)
html = DocTagsConverter.to_html(doctags)
text = DocTagsConverter.to_text(doctags)

# Save as JSON
doctags.save_json(Path("output.json"))
```

## Configuration Options

```python
from src.processing import PipelineConfig

config = PipelineConfig(
    # Chunking
    chunk_size=1000,              # Target chunk size in chars
    chunk_overlap=200,            # Overlap between chunks
    chunking_method='structure',  # or 'semantic'

    # OCR
    ocr_engine='paddleocr',       # or 'tesseract'
    ocr_lang='en',                # Language code

    # Processing
    max_file_size_mb=100.0,       # Max file size
    max_workers=4,                # Parallel workers

    # Output
    save_intermediate=True,
    save_json=True,
    save_markdown=True,
    output_dir=Path("output/")
)
```

### LLM-backed query decomposition

By default, complex queries are split using fast heuristics (conjunction splitting, comparison
detection). To enable an LLM-backed fallback for ambiguous queries set:

```bash
DOCTAGS_LLM_DECOMPOSITION=true
OPENAI_API_KEY=sk-...
```

When enabled, the planner runs heuristics first. If they produce no split, an LLM generates
2–4 focused sub-questions (2s timeout, silent fallback on failure). Each sub-question becomes
a parallel retrieval step, improving coverage on multi-part legal and analytical queries.

### Semantic Chunking

Semantic chunking requires a local embedding model. Set the environment variable `DOCTAGS_SEMANTIC_MODEL`
to the name of a Sentence Transformers model (for example `sentence-transformers/all-MiniLM-L6-v2`) before
launching the API or pipeline. When the model is not configured, the system automatically falls back to
structure-aware chunking and the web UI disables the semantic option.

## Access Structured Data

### Parsed Elements

```python
for element in result.parsed_doc.elements:
    print(f"{element.type}: {element.content[:50]}...")
```

### DocTags Hierarchy

```python
for tag in result.doctags_doc.tags:
    print(f"[{tag.tag_type.value}] {tag.content[:50]}...")
```

### Chunks with Context

```python
for chunk in result.chunks:
    print(f"Chunk {chunk.chunk_index}:")
    print(f"  Breadcrumbs: {chunk.context['breadcrumbs']}")
    print(f"  Content: {chunk.content[:100]}...")
```

## Progress Tracking

```python
def progress_callback(stage, percent):
    print(f"{stage}: {percent:.0%}")

result = pipeline.process_file(
    "document.pdf",
    progress_callback=progress_callback
)
```

## Error Handling

```python
result = pipeline.process_file("document.pdf")

if result.success:
    print(f"✓ Processed successfully")
    print(f"  - Stage: {result.stage.value}")
    print(f"  - Time: {result.processing_time:.2f}s")
else:
    print(f"✗ Processing failed")
    print(f"  - Stage: {result.stage.value}")
    print(f"  - Error: {result.error}")
```

## Statistics

```python
# Process multiple files
results = pipeline.process_batch(files)

# Get statistics
stats = pipeline.get_statistics(results)

print(f"Total: {stats['total']}")
print(f"Success: {stats['successful']}")
print(f"Failed: {stats['failed']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Total chunks: {stats['total_chunks']}")
print(f"Avg time: {stats['avg_processing_time']:.2f}s")
```

## Testing

```bash
# Run all tests
pytest tests/test_processing.py -v

# Run specific test
pytest tests/test_processing.py::TestDocumentParser -v

# Run with coverage
pytest tests/test_processing.py --cov=src.processing
```

## Demo Scripts

```bash
# Full demo with all features
python scripts/demo_processing.py

# Quick functionality test
python scripts/quick_test_processing.py
```

## Common Patterns

### Process Directory Recursively

```python
results = pipeline.process_directory(
    directory="documents/",
    recursive=True,
    progress_callback=lambda done, total: print(f"{done}/{total}")
)
```

### Streaming Large Files

```python
from src.processing import StreamingPipeline

streaming = StreamingPipeline(config)

def handle_chunk(chunk):
    # Process chunk immediately
    print(f"Received chunk: {chunk.chunk_id}")

streaming.process_file_streaming(
    "large_document.pdf",
    chunk_callback=handle_chunk
)
```

### Custom Chunking

```python
from src.processing import StructurePreservingChunker

# Custom chunker
chunker = StructurePreservingChunker(
    chunk_size=500,
    chunk_overlap=100,
    min_chunk_size=50,
    respect_boundaries=True,
    include_context=True
)

# Use in pipeline
pipeline.chunker = chunker
```

## Troubleshooting

### OCR Not Working

```python
# Try different OCR engine
config.ocr_engine = 'tesseract'

# Or check if PaddleOCR is installed
pip install paddleocr paddlepaddle
```

### PDF Parsing Issues

```python
# Force OCR for scanned PDFs
config.use_ocr_for_pdf = True

# Or install pdfplumber for better parsing
pip install pdfplumber
```

### Memory Issues

```python
# Reduce workers
config.max_workers = 2

# Use streaming for large files
from src.processing import StreamingPipeline
streaming = StreamingPipeline(config)
```

## Legal Document Processing

Contextprime includes domain-aware processing for UK/EU legal documents (regulations, statutes, contracts). When the processor detects ≥3 legal patterns in a document it activates a set of specialised `DocTagType` values that improve chunking and retrieval precision.

### Automatic domain detection

No configuration is needed. The processor inspects element text for patterns such as `"Article \d+"`, `"Schedule \d+"`, `"Regulation (EU)"`, `"pursuant to"`, etc. If at least 3 match, the document is tagged as `"legal"` and the following type mappings activate:

| Heading text | DocTagType |
|---|---|
| `"Article 6"`, `"Article 17"` | `ARTICLE` |
| `"Schedule 1"`, `"Annex I"` | `SCHEDULE` |
| `"Definitions"`, `"Interpretation"` | `DEFINITION` |

| Paragraph pattern | DocTagType |
|---|---|
| Starts with `"..."` (quoted definition) | `DEFINITION` |
| Contains `"except where"`, `"unless"` | `EXCEPTION` |
| Contains `"subject to Article"`, `"pursuant to Article"` | `CROSS_REFERENCE` |

ARTICLE and SCHEDULE tags also trigger chunk flush boundaries so that each article / schedule starts a new chunk, preserving retrieval precision.

### Legal metadata

Attach regulatory lifecycle metadata at ingestion time. The fields are stored in both Qdrant payloads and Neo4j document properties, enabling date-range filtering at retrieval time.

```python
from pathlib import Path
from src.core.config import LegalMetadataConfig
from src.pipelines import DocumentIngestionPipeline

pipeline = DocumentIngestionPipeline()

legal_meta = LegalMetadataConfig(
    in_force_from="2018-05-25",          # ISO-8601 date
    in_force_until=None,                  # still in force
    amended_by="Regulation (EU) 2021/1119",
    supersedes="Directive 95/46/EC",
)

report = pipeline.process_files(
    [Path("data/gdpr.pdf")],
    legal_metadata=legal_meta,
)

print(f"Chunks ingested:          {report.chunks_ingested}")
print(f"Cross-references stored:  {report.cross_references_stored}")
pipeline.close()
```

### Cross-reference edges

During ingestion the pipeline automatically extracts legal cross-references from chunk text (e.g. `"pursuant to Article 6"`, `"subject to Schedule 2"`) and stores them as `(:Chunk)-[:REFERENCES]->(:Chunk)` edges in Neo4j. These edges can be traversed in graph queries to follow regulatory citations across a document corpus.

```python
from src.processing.cross_reference_extractor import extract_cross_references

refs = extract_cross_references(
    chunk_id="gdpr_chunk_0042",
    content="The controller shall comply pursuant to Article 17(3).",
    doc_id="gdpr",
)
# refs → [CrossRef(ref_type='article', target_label='article_17(3)', ...)]
```

## Next Steps

1. ✓ Install dependencies: `pip install -r requirements.txt`
2. ✓ Run quick test: `python scripts/quick_test_processing.py`
3. ✓ Try demo: `python scripts/demo_processing.py`
4. ✓ Process your documents: Use examples above
5. ✓ Integrate with RAG: Feed chunks to Qdrant + Neo4j

## Full Documentation

See `/PROCESSING_IMPLEMENTATION.md` for complete technical documentation.

### Web Interface Demo

You can test the pipeline through the bundled FastAPI web UI without writing code:

```bash
cd doctags_rag
uvicorn src.api.main:app --reload
```

Open http://127.0.0.1:8000/ in your browser to access the interface. Upload a document, adjust chunking settings, and inspect the generated DocTags metadata, chunk previews, and markdown reconstruction on the right-hand side. Processed documents are stored in memory for the current session only.
