from types import SimpleNamespace
from pathlib import Path

import numpy as np

from src.pipelines.document_ingestion import (
    DocumentIngestionConfig,
    DocumentIngestionPipeline,
)
from src.knowledge_graph.graph_ingestor import GraphIngestionStats
from src.processing.chunker import Chunk
from src.processing.doctags_processor import DocTagsDocument
from src.processing.pipeline import ProcessingResult, ProcessingStage


class DummyEmbedder:
    def encode(self, texts, show_progress_bar=False):
        return [[float(len(text)), 0.0, 1.0] for text in texts]


class FakeQdrantManager:
    def __init__(self):
        self.created = []
        self.inserted = []

    def create_collection(self, **kwargs):
        self.created.append(kwargs)
        return True

    def insert_vectors_batch(self, vectors, collection_name=None, batch_size=100):
        self.inserted.append(
            {
                "vectors": vectors,
                "collection_name": collection_name,
                "batch_size": batch_size,
            }
        )
        return len(vectors)

    def close(self):  # pragma: no cover - nothing to cleanup in tests
        pass


class FakeGraphIngestor:
    def __init__(self):
        self.calls = []

    def ingest_document(self, doc_props, chunks):
        for chunk in chunks:
            assert "embedding" in chunk
            assert isinstance(chunk["embedding"], list)
            assert len(chunk["embedding"]) == 3
        self.calls.append((doc_props, chunks))
        return GraphIngestionStats(
            doc_id=doc_props["doc_id"],
            chunks_processed=len(chunks),
            sections_linked=len({c.get("context", {}).get("section") for c in chunks if c.get("context")}),
            subsections_linked=0,
        )

    def close(self):  # pragma: no cover - nothing to cleanup in tests
        pass


def build_processing_result(chunk_text: str = "Example text") -> ProcessingResult:
    chunk = Chunk(
        chunk_id="doc1_chunk_0000",
        content=chunk_text,
        doc_id="doc1",
        chunk_index=0,
        char_start=0,
        char_end=len(chunk_text),
        metadata={"tag_type": "paragraph"},
        context={"section": "Intro", "breadcrumbs": "Doc > Intro"},
    )

    doctags_doc = DocTagsDocument(
        doc_id="doc1",
        title="Doc 1",
        tags=[],
        metadata={},
        hierarchy={"title": "Doc 1"},
    )

    parsed_doc = SimpleNamespace(metadata={"extension": "txt"})

    return ProcessingResult(
        file_path=Path("doc1.txt"),
        success=True,
        stage=ProcessingStage.COMPLETED,
        parsed_doc=parsed_doc,
        doctags_doc=doctags_doc,
        chunks=[chunk],
        processing_time=0.1,
        metadata={"num_elements": 1, "file_type": "txt", "num_chunks": 1},
    )


def test_document_ingestion_pipeline_ingests_document():
    qdrant = FakeQdrantManager()
    graph = FakeGraphIngestor()
    embedder = DummyEmbedder()
    pipeline = DocumentIngestionPipeline(
        embeddings_model=embedder,
        processing_pipeline=SimpleNamespace(),
        qdrant_manager=qdrant,
        graph_ingestor=graph,
        config=DocumentIngestionConfig(qdrant_batch_size=10),
    )

    report = pipeline.ingest_processing_results([build_processing_result()])

    assert report.processed_documents == 1
    assert report.chunks_ingested == 1
    assert report.qdrant_vectors == 1
    assert len(qdrant.created) == 1  # collection ensured
    assert len(qdrant.inserted) == 1

    vector_point = qdrant.inserted[0]["vectors"][0]
    assert vector_point.id == "doc1_chunk_0000"
    assert vector_point.metadata["section"] == "Intro"
    assert "text" in vector_point.metadata

    assert len(graph.calls) == 1
    doc_props, chunks = graph.calls[0]
    assert doc_props["doc_id"] == "doc1"
    assert chunks[0]["chunk_id"] == "doc1_chunk_0000"


def test_chunk_text_truncation():
    qdrant = FakeQdrantManager()
    graph = FakeGraphIngestor()
    embedder = DummyEmbedder()
    config = DocumentIngestionConfig(chunk_text_truncate=5)
    pipeline = DocumentIngestionPipeline(
        embeddings_model=embedder,
        processing_pipeline=SimpleNamespace(),
        qdrant_manager=qdrant,
        graph_ingestor=graph,
        config=config,
    )

    long_text = "abcdefghij"
    report = pipeline.ingest_processing_results([build_processing_result(long_text)])

    assert report.processed_documents == 1
    vector_point = qdrant.inserted[0]["vectors"][0]
    assert vector_point.metadata["text"] == long_text[:5]

    _, chunks = graph.calls[0]
    assert chunks[0]["content"] == long_text[:5]
