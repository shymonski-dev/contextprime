"""
Document Processing Module for DocTags RAG System.

Provides comprehensive document processing capabilities:
- Multi-format parsing (PDF, DOCX, HTML, TXT, images)
- OCR for scanned documents
- Structure preservation using DocTags
- Intelligent chunking
- Complete processing pipeline

Main Components:
- DocumentParser: Parse documents from various formats
- OCREngine: Extract text from images and scanned documents
- DocTagsProcessor: Convert to structured DocTags format
- Chunker: Create context-aware chunks
- Pipeline: Orchestrate the complete workflow
"""

from .document_parser import (
    DocumentParser,
    ParsedDocument,
    DocumentElement,
    PDFParser,
    DOCXParser,
    HTMLParser,
    TextParser,
    ImageParser,
)

from .ocr_engine import (
    OCREngineFactory,
    PaddleOCREngine,
    TesseractOCREngine,
    OCRResult,
    OCRBox,
)

from .doctags_processor import (
    DocTagsProcessor,
    DocTagsDocument,
    DocTag,
    DocTagType,
    DocTagsConverter,
)

from .chunker import (
    StructurePreservingChunker,
    SemanticChunker,
    Chunk,
)

from .pipeline import (
    DocumentProcessingPipeline,
    ProcessingResult,
    ProcessingStage,
    PipelineConfig,
    StreamingPipeline,
    create_pipeline,
)

from .utils import (
    FileTypeDetector,
    TextCleaner,
    LanguageDetector,
    EncodingDetector,
    ImagePreprocessor,
    TableExtractor,
    ContentHasher,
    DocumentMetadataExtractor,
)

__all__ = [
    # Document Parsing
    'DocumentParser',
    'ParsedDocument',
    'DocumentElement',
    'PDFParser',
    'DOCXParser',
    'HTMLParser',
    'TextParser',
    'ImageParser',

    # OCR
    'OCREngineFactory',
    'PaddleOCREngine',
    'TesseractOCREngine',
    'OCRResult',
    'OCRBox',

    # DocTags
    'DocTagsProcessor',
    'DocTagsDocument',
    'DocTag',
    'DocTagType',
    'DocTagsConverter',

    # Chunking
    'StructurePreservingChunker',
    'SemanticChunker',
    'Chunk',

    # Pipeline
    'DocumentProcessingPipeline',
    'ProcessingResult',
    'ProcessingStage',
    'PipelineConfig',
    'StreamingPipeline',
    'create_pipeline',

    # Utils
    'FileTypeDetector',
    'TextCleaner',
    'LanguageDetector',
    'EncodingDetector',
    'ImagePreprocessor',
    'TableExtractor',
    'ContentHasher',
    'DocumentMetadataExtractor',
]

__version__ = '1.0.0'
