# Contextprime

**Ultimate RAG combining IBM structure preservation with Microsoft cross-document intelligence for advanced agentic reasoning**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Code](https://img.shields.io/badge/Code-35,680%20lines-green)](FINAL_SYSTEM_REPORT.md)
[![Status](https://img.shields.io/badge/Status-Release%20Validation%20Pending-yellow)](RELEASE_READINESS_REPORT_2026-02-20.md)
[![Tests](https://img.shields.io/badge/Tests-200%2B-brightgreen)](doctags_rag/tests/)

---

## GitHub Marketplace Action

This repository now includes a publish-ready GitHub action at `/Volumes/SSD2/SUPER_RAG/action.yml`.

Quick usage:

```yaml
name: Contextprime Gates
on:
  workflow_dispatch:

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ./
        with:
          run-security-gate: "true"
          run-readiness-gate: "true"
          run-full-tests: "false"
```

Action inputs:

- `run-security-gate` (default `true`)
- `run-readiness-gate` (default `true`)
- `run-full-tests` (default `false`)
- `full-tests-skip-build` (default `true`)
- `install-tooling` (default `true`)
- `install-project-dependencies` (default `true`)
- `create-env-from-example` (default `true`)

Marketplace release steps:

1. Push the committed changes to your public repository.
2. Create a version tag such as `v1.0.0`.
3. Create a GitHub release from that tag.
4. Publish the action to GitHub Marketplace from repository settings.

---

## 🎯 Overview

Contextprime is a **production-ready, enterprise-grade retrieval system** that implements:

- **IBM DocTags**: Structure-preserving document processing with layout analysis
- **Microsoft GraphRAG**: Cross-document intelligence with community detection
- **RAPTOR**: Hierarchical recursive summarization for multi-level retrieval
- **Agentic Architecture**: Self-improving system with reinforcement learning

Current checkpoint (2026-02-20):
- Development is paused for end of day.
- Release validation resumes with:
  - `./run_security_release_gate.sh`
  - `./run_european_union_artificial_intelligence_act_readiness.sh`
  - `./run_full_tests_stable.sh`

### System Stats
- **35,680** lines of production Python code
- **67** modules across 9 major components
- **200+** comprehensive tests
- **10** working demo scripts
- **0** mocked functionality

---

## 🚀 Quick Start

```bash
# 1. Navigate to project
cd doctags_rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download models
python -m spacy download en_core_web_lg
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# 4. Start databases (optional)
cd ..
docker-compose up -d neo4j qdrant

# 5. Run tests
cd doctags_rag
pytest tests/ -v

# 6. Try demos
python scripts/demo_processing.py
python scripts/demo_advanced_retrieval.py
python scripts/demo_agentic.py
```

---

## 📋 System Architecture

```
Documents → Processing (DocTags) → Dual Indexing (Neo4j + Qdrant)
                ↓
     Knowledge Graph Construction
                ↓
      Advanced Retrieval (CRAG)
                ↓
  RAPTOR + Community Detection
                ↓
      Agentic Feedback Loop
                ↓
            Results
```

---

## 📦 Components

### Phase 1: Dual Indexing Infrastructure (5,335 lines)
- Neo4j graph database with HNSW indexes
- Qdrant vector database
- Hybrid retrieval with RRF fusion
- **Lazy initialization with graceful degradation** ✨

```python
from src.retrieval import HybridRetriever

retriever = HybridRetriever()  # Auto-initializes databases
results, metrics = retriever.search(
    query_vector=embedding,
    query_text="What is machine learning?",
    top_k=10
)
```

### Phase 2: Document Processing (4,028 lines)
- Multi-format support (PDF, DOCX, HTML, images)
- PaddleOCR with layout analysis
- DocTags structure preservation
- Context-aware chunking

```python
from src.processing import DocumentProcessingPipeline

pipeline = DocumentProcessingPipeline()
result = pipeline.process_file("document.pdf")
print(f"Created {len(result.chunks)} chunks")
```

### Phase 3: Knowledge Graph (4,613 lines)
- Entity extraction (15+ types)
- Relationship detection (20+ types)
- Entity resolution with fuzzy matching
- Cross-document linking
- **Full-text index on Neo4j** — keyword search uses `CALL db.index.fulltext.queryNodes` for sub-millisecond lookups with automatic fallback to Python scan if the index is not yet created ✨

```python
from src.knowledge_graph import KnowledgeGraphPipeline

kg_pipeline = KnowledgeGraphPipeline()
result = kg_pipeline.process_documents_batch(documents)
```

### Phase 4: Advanced Retrieval (5,335 lines)
- CRAG-style confidence scoring
- Query routing and expansion
- Iterative refinement
- Cross-encoder reranking
- **Flexible embedding function injection** ✨

```python
from src.retrieval import AdvancedRetrievalPipeline

pipeline = AdvancedRetrievalPipeline(
    hybrid_retriever=retriever,
    embedding_function=your_embedding_fn  # Custom embeddings
)
result = pipeline.retrieve("query", top_k=10)
```

### Phase 5: RAPTOR Summarization (4,355 lines)
- Hierarchical tree construction
- Multi-level summarization
- Tree-based retrieval strategies

```python
from src.summarization import RAPTORPipeline

raptor = RAPTORPipeline()
result = raptor.process_document(document)
results = raptor.query("main findings", result.tree_id)
```

### Phase 6: Community Detection (4,373 lines)
- Multiple algorithms (Louvain, Leiden)
- Community summarization
- Global query handling

```python
from src.community import CommunityPipeline

pipeline = CommunityPipeline()
result = pipeline.run(graph=my_graph)
response = pipeline.answer_global_query("What are the main themes?")
```

### Phase 7: Agentic System (5,017 lines)
- Multi-agent coordination
- Reinforcement learning
- Self-improvement loops
- Memory systems

```python
from src.agents import AgenticPipeline, AgenticMode

pipeline = AgenticPipeline(mode=AgenticMode.STANDARD)
result = await pipeline.process_query(
    "complex question",
    max_iterations=2
)
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Stable full suite from repository root
cd ..
./run_full_tests_stable.sh

# Skip image build if you already built the latest app image
./run_full_tests_stable.sh --skip-build

# Security release gate (dependency scan + config checks + secret pattern checks)
./run_security_release_gate.sh

# Run specific test suite
pytest tests/test_agents.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

Test coverage:
- `test_indexing.py` - 45+ tests for dual indexing
- `test_processing.py` - 30+ tests for document processing
- `test_knowledge_graph.py` - 35+ tests for KG construction
- `test_advanced_retrieval.py` - 30+ tests for retrieval
- `test_summarization.py` - 20+ tests for RAPTOR
- `test_community.py` - 30+ tests for community detection
- `test_agents.py` - 70+ tests for agentic system

**Total: 200+ comprehensive tests**

---

## 📚 Documentation

### Key Documents
- **[FINAL_SYSTEM_REPORT.md](FINAL_SYSTEM_REPORT.md)** - Complete system report and code review
- **[system_report.json](system_report.json)** - Detailed metrics in JSON format
- **[CLAUDE.md](CLAUDE.md)** - Development principles and instructions

### Component Documentation
- [Dual Indexing Guide](DUAL_INDEXING_SETUP.md)
- [Knowledge Graph Guide](doctags_rag/docs/KNOWLEDGE_GRAPH.md)
- [Advanced Retrieval Guide](doctags_rag/docs/ADVANCED_RETRIEVAL.md)
- [Agentic System Guide](doctags_rag/docs/AGENTIC_SYSTEM.md)
- [Component READMEs](doctags_rag/src/)

---

## 🎯 Key Features

### IBM DocTags Approach
✅ Layout analysis with structure preservation
✅ Hierarchical document representation
✅ Context-aware chunking
✅ Multi-format support (PDF, DOCX, HTML, images)

### Microsoft GraphRAG
✅ Cross-document entity linking
✅ Community detection (Louvain, Leiden)
✅ Global query answering
✅ Knowledge synthesis

### RAPTOR Summarization
✅ Hierarchical tree construction
✅ Multi-level retrieval
✅ Adaptive strategies
✅ UMAP + HDBSCAN clustering

### Agentic Architecture
✅ Multi-agent coordination
✅ Reinforcement learning (Q-learning, MAB)
✅ Self-evaluation and improvement
✅ Memory systems (STM, LTM, episodic)

### Production Features
✅ Lazy initialization with fallbacks
✅ Flexible embedding injection
✅ Comprehensive error handling
✅ Extensive logging and monitoring
✅ Thread-safe singleton initialization (double-checked locking)
✅ Graceful degradation
✅ Admin-role-gated management endpoints
✅ JWT expiry enforcement (configurable)
✅ Distributed rate limiting with per-request cost weights
✅ OCR engine health logged at startup
✅ Versioned SQLite schema migrations
✅ Partial backend failures surfaced in `search_errors` metadata

---

## 🛠️ Configuration

Edit `doctags_rag/config/config.yaml`:

```yaml
# Neo4j Configuration
neo4j:
  uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "password"

# Qdrant Configuration
qdrant:
  host: "localhost"
  port: 6333

# LLM Configuration
llm:
  provider: "openai"
  model: "gpt-4-turbo-preview"
  api_key: null  # Set via environment

# Document Processing
document_processing:
  ocr_engine: "paddleocr"
  chunk_size: 1000
  chunk_overlap: 200
  preserve_structure: true

# Knowledge Graph
knowledge_graph:
  entity_extraction:
    confidence_threshold: 0.7
  entity_resolution:
    similarity_threshold: 0.85

# Advanced Retrieval
retrieval:
  hybrid_search:
    enable: true
    vector_weight: 0.7
    graph_weight: 0.3
  confidence_scoring:
    enable: true
    min_confidence: 0.5

# Agentic Features
agents:
  enable_feedback_loop: true
  self_critique: true
  max_iterations: 3
```

---

## 📊 Performance

### Latency (typical)
- Document processing: 1-5s per page
- Entity extraction: ~100 entities/sec
- Graph queries: <100ms
- Vector search: <50ms
- Hybrid retrieval: 100-500ms
- Agentic pipeline: 1-5s

### Scalability
- Documents: 10K+ tested
- Entities: 100K+ supported
- Graph: Millions of nodes/edges
- Vectors: Billions (via Qdrant)

---

## 🎓 Use Cases

1. **Enterprise Search**: Intelligent document retrieval with structure awareness
2. **Research Assistant**: Cross-document analysis and synthesis
3. **Compliance Analysis**: Policy and regulation checking
4. **Knowledge Management**: Organizational intelligence
5. **Due Diligence**: Document review and analysis
6. **Scientific Literature Review**: Multi-document understanding
7. **Legal Document Analysis**: Case law and precedent analysis

---

## 🏗️ System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- 10GB disk space

### Recommended
- Python 3.10+
- 16GB+ RAM
- 50GB+ disk space
- GPU for OCR/embeddings (optional)

### External Services (Optional)
- Neo4j 5.x (or Docker)
- Qdrant (or Docker)
- OpenAI API (for LLM features)

---

## 🔧 Verification Scripts

### Interactive Launch
```bash
python launch_system.py
```

### Non-Interactive Verification
```bash
python verify_system.py
```

Both scripts provide comprehensive system diagnostics, dependency checking, and test execution options.

---

## 📦 Installation

### Standard Installation
```bash
pip install -r doctags_rag/requirements.txt
```

### With Development Tools
```bash
pip install -r doctags_rag/requirements.txt
pip install pytest pytest-cov black flake8 mypy
```

### Download Models
```bash
# SpaCy
python -m spacy download en_core_web_lg

# NLTK
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## 🐳 Docker Deployment

The compose file ships with services for Neo4j, Qdrant, and the Contextprime API.
The application image builds from `doctags_rag/Dockerfile` and mounts the
repo-local `models/` directory so large Hugging Face checkpoints remain outside
the container.

> **Note**: On ARM64 builds the Docker image skips heavy optional packages that
> lack wheels (`paddleocr`, `paddlepaddle`, `hdbscan`). OCR and RAPTOR features
> will be disabled in that environment unless you install those dependencies
> manually.

> **Tip**: `python-multipart` and `langdetect` are now bundled in the base
> requirements so FastAPI form uploads and language detection work out of the
> box.
>
> **MonoT5 reminder**: Install `sentencepiece` (already pinned in
> `requirements.txt`) before enabling the reranker, otherwise the tokenizer
> cannot initialise.
>
> **Embedding reminder**: All bundled collections use OpenAI
> `text-embedding-3-small` (1536‑d). When running ad-hoc retrieval tests, prefer
> `src.embeddings.openai_embeddings.OpenAIEmbeddingModel` so your query vectors
> match the stored dimensionality.

```bash
# 1) Build the Contextprime application image
docker compose --env-file doctags_rag/.env build app

# 2) (Optional) Download MonoT5 weights into ./doctags_rag/models
docker compose --env-file doctags_rag/.env run --rm app python scripts/download_models.py

# 3) Launch the full stack
docker compose --env-file doctags_rag/.env up -d neo4j qdrant app

# 4) Follow application logs
docker compose --env-file doctags_rag/.env logs -f app

# 5) Stop all services
docker compose --env-file doctags_rag/.env down

# 6) Tear down services and remove persistent volumes
docker compose --env-file doctags_rag/.env down -v
```

Access points:
- Contextprime API and web interface: http://localhost:8000
- Neo4j Browser: http://localhost:7474
- Neo4j Bolt: bolt://localhost:7687
- Qdrant API: http://localhost:6333
- Qdrant Dashboard: http://localhost:6333/dashboard

Open http://localhost:8000 to launch the **Contextprime Studio** console. The interface covers
document ingestion, hybrid search (with optional MonoT5 reranking), and FAST-mode agentic queries, and
includes a dark-mode toggle that remembers your preference.

### ARM64 OCR Support

PaddleOCR/PaddlePaddle are skipped in the default ARM64 image to avoid lengthy builds. To enable OCR in an
ARM64 container:

```bash
# Install Paddle deps inside the running app container
docker compose --env-file doctags_rag/.env exec app pip install "paddlepaddle==3.2.0" "paddleocr==2.7.0"

# Restart the app service afterward
docker compose --env-file doctags_rag/.env restart app
```

Alternatively, switch the pipeline to the Tesseract engine (`enable_ocr=false` or `ocr_engine='tesseract'`)
when PaddleOCR is unavailable.

The app container reads environment variables from `doctags_rag/.env`. Set
your `OPENAI_API_KEY` and strong infrastructure secrets there before booting the
stack.

Minimum secure environment values:

```bash
NEO4J_PASSWORD=<strong_neo4j_password>
SECURITY__REQUIRE_ACCESS_TOKEN=true
SECURITY__AUTH_MODE=jwt
SECURITY__JWT_SECRET=<long_random_jwt_secret_min_32_chars>
SECURITY__JWT_REQUIRE_EXPIRY=true
```

The API enforces startup checks in docker mode. The app service now fails fast
if required secrets are missing or default-valued.

### Security Hardening (2026-02-22)

The following security and reliability improvements are included:

**Admin endpoint protection** — `GET /api/admin/neo4j/connectivity` and
`POST /api/admin/neo4j/recover-password` now require the caller's JWT to carry
an `admin` or `owner` role. Tokens with only `api:write` scope are rejected
with 403. In `token` auth mode the admin endpoints are always blocked (use JWT
mode with roles for production admin access).

**JWT expiry enforcement** — By default, tokens without an `exp` claim are
rejected. Set `SECURITY__JWT_REQUIRE_EXPIRY=false` only for internal service
accounts that intentionally use non-expiring tokens.

**Token budget rate limiting** — The Redis-backed rate limiter now correctly
applies per-request cost weights (for LLM token budgeting) through a Lua
script, so cost accounting is consistent across distributed workers.

**Partial backend failures surfaced** — When the vector or graph backend is
temporarily unavailable, the search response now includes a `search_errors`
list in the metadata rather than silently returning an empty result set. Use
this field to distinguish "no matches" from "backend error".

Runtime probes:

- Liveness: `GET /api/health`
- Readiness: `GET /api/readiness`

Live smoke gate command:

```bash
./run_live_smoke_gate.sh
```

Real-world stable command:

```bash
./run_realworld_stable.sh
```

European Union Artificial Intelligence Act readiness gate:

```bash
./run_european_union_artificial_intelligence_act_readiness.sh
```

Compliance profile and evidence files are under `doctags_rag/compliance/`.

When access control is enabled, the web console includes a token field in the
header and sends that value on all protected requests. In signed token mode,
use a signed token.

The `./doctags_rag/models` and `./doctags_rag/data` directories are mounted into
the container; any ingested artifacts or downloaded reranker weights persist
across rebuilds.

> **Container networking**: when you target services from inside Docker, use the
> compose service names (e.g. `qdrant`, `neo4j`). The bundled `.env` already sets
> `QDRANT_HOST=qdrant` for you.

---

## 🎬 Demo Scripts

```bash
# Document Processing
python scripts/demo_processing.py

# Advanced Retrieval
python scripts/demo_advanced_retrieval.py

# Knowledge Graph
python scripts/build_sample_kg.py

# RAPTOR Summarization
python scripts/demo_raptor.py

# Community Detection
python scripts/demo_community.py

# Agentic System
python scripts/demo_agentic.py

# Full Integration (coming soon)
python scripts/demo_full_system.py
```

---

## 🤝 Contributing

This is a research implementation. For production use:

1. Configure appropriate API keys
2. Set up production databases
3. Adjust performance parameters
4. Enable monitoring and logging
5. Set signed token secrets and permissions
6. Run release security and compliance gates
7. Set up backups

---

## 📄 License

Contextprime is dual licensed:

1. GNU Affero General Public License, version 3 or any later version
2. Contextprime Commercial License (separate signed agreement)

See:

- `DUAL_LICENSE.md`
- `LICENSE`
- `LICENSES/CONTEXTPRIME_COMMERCIAL_LICENSE.md`

---

## 🎉 Status

**Hardening complete, release validation pending final gate re run**

- All 6 phases fully implemented
- 35,680 lines of production code
- 67 modules, 200+ tests
- Zero mocked functionality
- Comprehensive documentation
- Release checklist and security gate are in place

See [FINAL_SYSTEM_REPORT.md](FINAL_SYSTEM_REPORT.md) for detailed analysis.

---

## 📞 Support

For issues or questions:
1. Check documentation in `docs/`
2. Review demo scripts in `scripts/`
3. Run verification: `python verify_system.py`
4. Consult [FINAL_SYSTEM_REPORT.md](FINAL_SYSTEM_REPORT.md)

---

Built following IBM DocTags, Microsoft GraphRAG, and RAPTOR principles.

Last Updated: February 22, 2026
