from types import SimpleNamespace
from pathlib import Path
from contextprime.core.config import LegalMetadataConfig

import numpy as np

from contextprime.pipelines.document_ingestion import (
    DocumentIngestionConfig,
    DocumentIngestionPipeline,
)
from contextprime.knowledge_graph.graph_ingestor import GraphIngestionStats
from contextprime.processing.chunker import Chunk
from contextprime.processing.doctags_processor import DocTagsDocument
from contextprime.processing.pipeline import ProcessingResult, ProcessingStage


class DummyEmbedder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, show_progress_bar=False):
        self.calls.append(list(texts))
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


def test_embedding_text_includes_context_by_default():
    qdrant = FakeQdrantManager()
    graph = FakeGraphIngestor()
    embedder = DummyEmbedder()
    pipeline = DocumentIngestionPipeline(
        embeddings_model=embedder,
        processing_pipeline=SimpleNamespace(),
        qdrant_manager=qdrant,
        graph_ingestor=graph,
        config=DocumentIngestionConfig(),
    )

    report = pipeline.ingest_processing_results([build_processing_result("Context body text")])
    assert report.processed_documents == 1
    assert embedder.calls

    embedded_text = embedder.calls[0][0]
    assert "Document title: Doc 1" in embedded_text
    assert "Section: Intro" in embedded_text
    assert "Content:" in embedded_text
    assert "Context body text" in embedded_text


class FakeNeo4j:
    """Fake Neo4j manager that records stored cross-references."""
    def __init__(self):
        self.stored_refs = []

    def store_cross_references(self, refs, **kwargs):
        self.stored_refs.extend(refs)
        return len(refs)


class FakeGraphIngestorWithNeo4j(FakeGraphIngestor):
    """FakeGraphIngestor extended with a neo4j property for cross-ref tests."""
    def __init__(self):
        super().__init__()
        self.neo4j = FakeNeo4j()


LEGAL_CHUNK_TEXT = (
    "This obligation applies pursuant to Article 6. "
    "The controller must comply subject to Article 17(3)."
)


def test_embedding_text_can_disable_context():
    qdrant = FakeQdrantManager()
    graph = FakeGraphIngestor()
    embedder = DummyEmbedder()
    pipeline = DocumentIngestionPipeline(
        embeddings_model=embedder,
        processing_pipeline=SimpleNamespace(),
        qdrant_manager=qdrant,
        graph_ingestor=graph,
        config=DocumentIngestionConfig(contextualize_embeddings=False),
    )

    raw_text = "Raw chunk content only"
    report = pipeline.ingest_processing_results([build_processing_result(raw_text)])
    assert report.processed_documents == 1
    assert embedder.calls[0][0] == raw_text


def test_cross_references_stored_for_legal_content():
    qdrant = FakeQdrantManager()
    graph = FakeGraphIngestorWithNeo4j()
    embedder = DummyEmbedder()
    pipeline = DocumentIngestionPipeline(
        embeddings_model=embedder,
        processing_pipeline=SimpleNamespace(),
        qdrant_manager=qdrant,
        graph_ingestor=graph,
        config=DocumentIngestionConfig(),
    )

    report = pipeline.ingest_processing_results([build_processing_result(LEGAL_CHUNK_TEXT)])

    assert report.cross_references_stored >= 1
    assert len(graph.neo4j.stored_refs) >= 1


def test_cross_references_zero_for_plain_text():
    qdrant = FakeQdrantManager()
    graph = FakeGraphIngestorWithNeo4j()
    embedder = DummyEmbedder()
    pipeline = DocumentIngestionPipeline(
        embeddings_model=embedder,
        processing_pipeline=SimpleNamespace(),
        qdrant_manager=qdrant,
        graph_ingestor=graph,
        config=DocumentIngestionConfig(),
    )

    plain_text = "The controller shall implement appropriate technical measures."
    report = pipeline.ingest_processing_results([build_processing_result(plain_text)])

    assert report.cross_references_stored == 0


def test_legal_metadata_stored_in_graph_doc_props():
    qdrant = FakeQdrantManager()
    graph = FakeGraphIngestorWithNeo4j()
    embedder = DummyEmbedder()
    pipeline = DocumentIngestionPipeline(
        embeddings_model=embedder,
        processing_pipeline=SimpleNamespace(),
        qdrant_manager=qdrant,
        graph_ingestor=graph,
        config=DocumentIngestionConfig(),
    )

    legal_meta = LegalMetadataConfig(in_force_from="2018-05-25")
    pipeline.ingest_processing_results(
        [build_processing_result()],
        legal_metadata=legal_meta,
    )

    doc_props = graph.calls[0][0]
    assert doc_props["in_force_from"] == "2018-05-25"


def test_legal_metadata_stored_in_qdrant_payload():
    qdrant = FakeQdrantManager()
    graph = FakeGraphIngestorWithNeo4j()
    embedder = DummyEmbedder()
    pipeline = DocumentIngestionPipeline(
        embeddings_model=embedder,
        processing_pipeline=SimpleNamespace(),
        qdrant_manager=qdrant,
        graph_ingestor=graph,
        config=DocumentIngestionConfig(),
    )

    legal_meta = LegalMetadataConfig(in_force_from="2018-05-25")
    pipeline.ingest_processing_results(
        [build_processing_result()],
        legal_metadata=legal_meta,
    )

    vector_metadata = qdrant.inserted[0]["vectors"][0].metadata
    assert vector_metadata["in_force_from"] == "2018-05-25"


def test_no_legal_keys_without_legal_metadata():
    qdrant = FakeQdrantManager()
    graph = FakeGraphIngestorWithNeo4j()
    embedder = DummyEmbedder()
    pipeline = DocumentIngestionPipeline(
        embeddings_model=embedder,
        processing_pipeline=SimpleNamespace(),
        qdrant_manager=qdrant,
        graph_ingestor=graph,
        config=DocumentIngestionConfig(),
    )

    pipeline.ingest_processing_results([build_processing_result()])

    for inserted_batch in qdrant.inserted:
        for vector_point in inserted_batch["vectors"]:
            assert "in_force_from" not in vector_point.metadata
