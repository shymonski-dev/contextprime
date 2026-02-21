# Retrieval-Augmented Generation Methods Review (February 20, 2026)

This note captures high-value methods from primary sources and maps them to this code base.

## Primary sources reviewed

1. Self-Reflective Retrieval-Augmented Generation (Self-RAG, ICLR 2024): https://arxiv.org/abs/2310.11511
2. Corrective Retrieval-Augmented Generation (CRAG, arXiv 2024): https://arxiv.org/abs/2401.15884
3. Graph Retrieval-Augmented Generation for local to global search (Microsoft Research, 2024): https://arxiv.org/abs/2404.16130
4. Graph Retrieval-Augmented Generation project updates (LazyGraphRAG and Dynamic community selection): https://www.microsoft.com/en-us/research/project/graphrag/
5. Retrieval-Augmented generation with tree-organized retrieval (RAPTOR, ICLR 2024): https://arxiv.org/abs/2401.18059
6. Instruction-following retrieval denoising and reasoning (InstructRAG, ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/1f7f8986012740f81f6898f41f86684d-Abstract-Conference.html
7. Sparse retrieval with lightweight reasoning step (SparseRAG, ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/ee44ca11ac174922b06f43f947a752f8-Abstract-Conference.html
8. Efficient context pruning for long context retrieval (PROVENCE, ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/2f98f1ea8e2f7f3038d195bbf05f7603-Abstract-Conference.html
9. Zero cost context pruning extension (XPROVENCE, arXiv January 2026): https://arxiv.org/abs/2601.18886
10. Efficient adaptive graph retrieval for generation (EA-GraphRAG, arXiv February 2026): https://arxiv.org/abs/2602.13022

## Current status in this repository

Implemented in retrieval path:

1. Multi-query retrieval with reciprocal rank fusion.
2. Dense plus lexical hybrid retrieval.
3. Corrective retrieval pass (optional, config driven):
   - trigger on low confidence or low result count
   - run recovery variants with broader retrieval
4. Sentence-level context pruning (optional, config driven):
   - keeps high-overlap evidence sentences
   - reduces low-value context before generation
5. Graph retrieval policy modes (optional, config driven):
   - standard, local neighborhood expansion, global keyword mode, drift mixed mode, community summary mode
6. Generator-driven retrieval feedback loop in agent pipeline:
   - asks for focused extra evidence when quality is weak
   - keeps enriched evidence set if quality improves
7. Trainable context selector:
   - model class in retrieval module
   - offline training and evaluation script
8. Incremental context selector updates from labeled feedback:
   - `partial_fit` update method in selector
   - update script for feedback ingestion and holdout checks
9. Retrieval policy benchmark suite:
   - reusable metric helpers for retrieval and answer coverage
   - offline script to compare standard, local, global, drift, and community modes
10. Production feedback capture and continuous selector updates:
   - query event capture with query identifier in search metadata
   - feedback endpoint for helpfulness and result-level labels
   - automated feedback learning cycle script for selector updates
11. Benchmark trend publishing:
   - append benchmark outputs to trend history file
   - publish markdown summary of latest and recent policy metrics
12. Adaptive graph policy option:
   - request-level `adaptive` policy that routes between standard, local, global, drift, and community graph search modes
   - keeps default behavior stable while enabling adaptive retrieval when requested
13. Strict request budget controls:
   - hard caps for result count, variant count, total retrieval passes, and request time budget
   - budget telemetry in search response metadata for production tuning

## New configuration keys

In `retrieval.hybrid_search`:

1. `corrective.*`
2. `context_pruning.*`
3. `graph_policy.*`

Both are disabled by default to avoid behavior changes until you choose to enable them.

## Recommended next implementation steps

1. Add automated feedback capture pipeline from production requests into selector training data.
2. Add benchmark dashboard publishing so policy metrics trend over time and across datasets.
3. Add scheduled jobs for nightly feedback learning cycle and weekly benchmark trend refresh.
