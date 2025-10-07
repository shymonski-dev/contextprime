"""
Document Processing Pipeline.

Orchestrates the complete document processing workflow:
1. File type detection
2. Raw content extraction
3. OCR if needed
4. Structure analysis
5. DocTags generation
6. Chunking
7. Embedding preparation

Supports batch processing, error handling, and progress tracking.
"""

from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import os

from loguru import logger

from .utils import FileTypeDetector, DocumentMetadataExtractor
from .document_parser import DocumentParser, ParsedDocument
from .doctags_processor import DocTagsProcessor, DocTagsDocument, DocTagsConverter
from .chunker import StructurePreservingChunker, SemanticChunker, Chunk
from .ocr_engine import OCREngineFactory


class ProcessingStage(Enum):
    """Processing pipeline stages."""
    FILE_DETECTION = "file_detection"
    PARSING = "parsing"
    DOCTAGS_GENERATION = "doctags_generation"
    CHUNKING = "chunking"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingResult:
    """Result of document processing."""
    file_path: Path
    success: bool
    stage: ProcessingStage
    parsed_doc: Optional[ParsedDocument] = None
    doctags_doc: Optional[DocTagsDocument] = None
    chunks: Optional[List[Chunk]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'file_path': str(self.file_path),
            'success': self.success,
            'stage': self.stage.value,
            'error': self.error,
            'processing_time': self.processing_time,
            'metadata': self.metadata,
            'num_chunks': len(self.chunks) if self.chunks else 0,
        }


@dataclass
class PipelineConfig:
    """Configuration for processing pipeline."""
    # OCR settings
    enable_ocr: bool = True
    ocr_engine: str = 'paddleocr'
    ocr_lang: str = 'en'
    use_ocr_for_pdf: bool = False  # Force OCR for PDFs

    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_method: str = 'structure'  # 'structure' or 'semantic'
    semantic_model: Optional[str] = None

    # Processing settings
    max_file_size_mb: float = 100.0
    supported_formats: List[str] = field(default_factory=lambda: [
        'pdf', 'docx', 'html', 'txt', 'md', 'png', 'jpg', 'jpeg'
    ])

    # Output settings
    save_intermediate: bool = False
    output_dir: Optional[Path] = None
    save_markdown: bool = False
    save_json: bool = False

    # Performance settings
    batch_size: int = 10
    max_workers: int = 4


class DocumentProcessingPipeline:
    """
    Main document processing pipeline.

    Orchestrates all processing stages with error handling,
    progress tracking, and batch processing support.
    """

    _SEMANTIC_MODEL_CACHE: Dict[str, Any] = {}
    _SEMANTIC_MODEL_ERRORS: Dict[str, str] = {}

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize processing pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()

        # Initialize components
        logger.info("Initializing processing pipeline...")

        # OCR engine (optional)
        self.ocr_engine = None
        self.ocr_enabled = self.config.enable_ocr

        if self.config.enable_ocr:
            ocr_type = (self.config.ocr_engine or '').lower()

            if ocr_type in {'', 'none', 'disabled'}:
                self.ocr_enabled = False
                logger.info("OCR disabled via configuration")
            else:
                try:
                    self.ocr_engine = OCREngineFactory.create_engine(
                        engine_type=self.config.ocr_engine,
                        lang=self.config.ocr_lang
                    )
                    if self.ocr_engine is None:
                        self.ocr_enabled = False
                        logger.info("OCR engine factory returned None; continuing without OCR")
                except Exception as ocr_err:
                    self.ocr_enabled = False
                    logger.warning(
                        f"Failed to initialize OCR engine '{self.config.ocr_engine}': {ocr_err}. "
                        "Proceeding without OCR support."
                    )
        else:
            logger.info("OCR support disabled")

        if not self.ocr_enabled:
            if self.config.use_ocr_for_pdf:
                logger.warning("OCR disabled but use_ocr_for_pdf=True; disabling forced OCR for PDFs")
                self.config.use_ocr_for_pdf = False

            image_formats = {'png', 'jpg', 'jpeg', 'tiff', 'bmp'}
            self.config.supported_formats = [
                fmt for fmt in self.config.supported_formats if fmt.lower() not in image_formats
            ]

        # Document parser
        self.parser = DocumentParser(
            ocr_engine=self.ocr_engine,
            enable_ocr=self.ocr_enabled
        )

        # DocTags processor
        self.doctags_processor = DocTagsProcessor()

        # Chunker
        self.actual_chunking_method = 'structure'
        self.semantic_chunking_error: Optional[str] = None
        self.semantic_model_name: Optional[str] = None

        if self.config.chunking_method == 'semantic':
            model_name = (
                self.config.semantic_model
                or os.getenv('DOCTAGS_SEMANTIC_MODEL')
            )
            self.semantic_model_name = model_name

            if not model_name:
                self.semantic_chunking_error = (
                    "Semantic chunking requested but no semantic model configured. "
                    "Set PipelineConfig.semantic_model or DOCTAGS_SEMANTIC_MODEL."
                )
                logger.warning(self.semantic_chunking_error)
                self.chunker = StructurePreservingChunker(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                )
            else:
                embeddings_model, error = self._get_semantic_model(model_name)
                if embeddings_model is None:
                    self.semantic_chunking_error = error or (
                        "Semantic chunking unavailable for unknown reason"
                    )
                    logger.warning(
                        f"Semantic chunker unavailable ({self.semantic_chunking_error}); "
                        "falling back to structure-preserving chunks"
                    )
                    self.chunker = StructurePreservingChunker(
                        chunk_size=self.config.chunk_size,
                        chunk_overlap=self.config.chunk_overlap
                    )
                else:
                    self.chunker = SemanticChunker(
                        chunk_size=self.config.chunk_size,
                        embeddings_model=embeddings_model
                    )
                    self.actual_chunking_method = 'semantic'
        else:
            self.chunker = StructurePreservingChunker(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )

        if self.actual_chunking_method != 'semantic':
            self.actual_chunking_method = 'structure'

        logger.info("Pipeline initialized successfully")

    def process_file(
        self,
        file_path: Union[str, Path],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> ProcessingResult:
        """
        Process a single document file.

        Args:
            file_path: Path to the document
            progress_callback: Optional callback for progress updates

        Returns:
            ProcessingResult with all processed data
        """
        file_path = Path(file_path)
        start_time = time.time()

        logger.info(f"Processing: {file_path}")

        try:
            # Stage 1: File detection and validation
            if progress_callback:
                progress_callback("file_detection", 0.1)

            if not self._validate_file(file_path):
                return ProcessingResult(
                    file_path=file_path,
                    success=False,
                    stage=ProcessingStage.FILE_DETECTION,
                    error="File validation failed"
                )

            # Stage 2: Parse document
            if progress_callback:
                progress_callback("parsing", 0.3)

            parsed_doc = self.parser.parse(file_path)

            if not parsed_doc or not parsed_doc.text:
                return ProcessingResult(
                    file_path=file_path,
                    success=False,
                    stage=ProcessingStage.PARSING,
                    error="Failed to extract text from document"
                )

            logger.info(f"Parsed: {len(parsed_doc.elements)} elements, {len(parsed_doc.text)} chars")

            # Stage 3: Generate DocTags
            if progress_callback:
                progress_callback("doctags_generation", 0.5)

            doctags_doc = self.doctags_processor.process(parsed_doc)

            logger.info(f"Generated DocTags: {len(doctags_doc.tags)} tags")

            # Stage 4: Chunk document
            if progress_callback:
                progress_callback("chunking", 0.7)

            chunks = self.chunker.chunk_document(doctags_doc)

            logger.info(f"Created chunks: {len(chunks)} chunks")

            # Stage 5: Save intermediate results if configured
            if self.config.save_intermediate and self.config.output_dir:
                self._save_intermediate_results(
                    file_path, parsed_doc, doctags_doc, chunks
                )

            # Complete
            if progress_callback:
                progress_callback("completed", 1.0)

            processing_time = time.time() - start_time

            result = ProcessingResult(
                file_path=file_path,
                success=True,
                stage=ProcessingStage.COMPLETED,
                parsed_doc=parsed_doc,
                doctags_doc=doctags_doc,
                chunks=chunks,
                processing_time=processing_time,
                metadata={
                    'num_elements': len(parsed_doc.elements),
                    'num_tags': len(doctags_doc.tags),
                    'num_chunks': len(chunks),
                    'file_type': parsed_doc.metadata.get('extension', 'unknown'),
                    'chunking_method': self.actual_chunking_method,
                    'chunking_method_requested': self.config.chunking_method,
                    'semantic_chunking_error': self.semantic_chunking_error,
                    'semantic_model': self.semantic_model_name,
                }
            )

            logger.info(
                f"Completed {file_path.name}: "
                f"{len(chunks)} chunks in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Processing failed for {file_path}: {e}")
            logger.debug(traceback.format_exc())

            return ProcessingResult(
                file_path=file_path,
                success=False,
                stage=ProcessingStage.FAILED,
                error=str(e),
                processing_time=time.time() - start_time
            )

    def process_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[ProcessingResult]:
        """
        Process all documents in a directory.

        Args:
            directory: Directory path
            recursive: Process subdirectories
            progress_callback: Optional callback(processed, total)

        Returns:
            List of processing results
        """
        directory = Path(directory)

        if not directory.exists() or not directory.is_dir():
            logger.error(f"Invalid directory: {directory}")
            return []

        # Find all supported files
        files = self._find_supported_files(directory, recursive)

        logger.info(f"Found {len(files)} supported files in {directory}")

        return self.process_batch(files, progress_callback)

    def process_batch(
        self,
        file_paths: List[Union[str, Path]],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple documents in batch.

        Args:
            file_paths: List of file paths
            progress_callback: Optional callback(processed, total)

        Returns:
            List of processing results
        """
        results = []
        total = len(file_paths)

        logger.info(f"Processing batch of {total} files")

        # Process in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_file, file_path): file_path
                for file_path in file_paths
            }

            # Collect results as they complete
            for idx, future in enumerate(as_completed(future_to_file)):
                file_path = future_to_file[future]

                try:
                    result = future.result()
                    results.append(result)

                    if progress_callback:
                        progress_callback(idx + 1, total)

                except Exception as e:
                    logger.error(f"Batch processing error for {file_path}: {e}")
                    results.append(ProcessingResult(
                        file_path=Path(file_path),
                        success=False,
                        stage=ProcessingStage.FAILED,
                        error=str(e)
                    ))

        # Log summary
        successful = sum(1 for r in results if r.success)
        logger.info(
            f"Batch processing complete: "
            f"{successful}/{total} successful"
        )

        return results

    def _validate_file(self, file_path: Path) -> bool:
        """
        Validate file before processing.

        Args:
            file_path: Path to file

        Returns:
            True if file is valid
        """
        # Check existence
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        # Check file size
        size_mb = FileTypeDetector.get_file_size_mb(file_path)
        if size_mb > self.config.max_file_size_mb:
            logger.error(
                f"File too large: {size_mb:.2f}MB > "
                f"{self.config.max_file_size_mb}MB"
            )
            return False

        # Check file type
        if not FileTypeDetector.is_supported(
            file_path, self.config.supported_formats
        ):
            file_type = FileTypeDetector.detect_file_type(file_path)
            logger.error(f"Unsupported file type: {file_type}")
            return False

        return True

    def _find_supported_files(
        self,
        directory: Path,
        recursive: bool
    ) -> List[Path]:
        """
        Find all supported files in directory.

        Args:
            directory: Directory to search
            recursive: Search subdirectories

        Returns:
            List of file paths
        """
        files: List[Path] = []

        candidates = directory.rglob('*') if recursive else directory.glob('*')
        supported = {ext.lower() for ext in self.config.supported_formats}

        for candidate in candidates:
            if not candidate.is_file():
                continue

            ext = candidate.suffix.lstrip('.').lower()
            if ext in supported:
                files.append(candidate)

        return sorted(files)

    def _save_intermediate_results(
        self,
        file_path: Path,
        parsed_doc: ParsedDocument,
        doctags_doc: DocTagsDocument,
        chunks: List[Chunk]
    ) -> None:
        """
        Save intermediate processing results.

        Args:
            file_path: Original file path
            parsed_doc: Parsed document
            doctags_doc: DocTags document
            chunks: List of chunks
        """
        if not self.config.output_dir:
            return

        output_dir = self.config.output_dir / file_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save DocTags JSON
        if self.config.save_json:
            json_path = output_dir / 'doctags.json'
            doctags_doc.save_json(json_path)

        # Save Markdown
        if self.config.save_markdown:
            md_path = output_dir / 'document.md'
            markdown = DocTagsConverter.to_markdown(doctags_doc)
            md_path.write_text(markdown, encoding='utf-8')

        # Save chunks JSON
        chunks_path = output_dir / 'chunks.json'
        import json
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump(
                [chunk.to_dict() for chunk in chunks],
                f,
                indent=2,
                ensure_ascii=False
            )

        logger.info(f"Saved intermediate results to {output_dir}")

    def get_statistics(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """
        Get statistics from processing results.

        Args:
            results: List of processing results

        Returns:
            Statistics dictionary
        """
        if not results:
            return {}

        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        stats = {
            'total': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(results) if results else 0,
        }

        if successful:
            stats.update({
                'total_chunks': sum(len(r.chunks) for r in successful if r.chunks),
                'avg_chunks_per_doc': sum(len(r.chunks) for r in successful if r.chunks) / len(successful),
                'total_processing_time': sum(r.processing_time for r in successful),
                'avg_processing_time': sum(r.processing_time for r in successful) / len(successful),
            })

        # File type breakdown
        file_types = {}
        for result in results:
            file_type = result.metadata.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1

        stats['file_types'] = file_types

        # Error breakdown
        if failed:
            error_types = {}
            for result in failed:
                error = result.error or 'unknown'
                error_types[error] = error_types.get(error, 0) + 1
            stats['error_types'] = error_types

        return stats


class StreamingPipeline:
    """
    Streaming version of pipeline for processing large files.

    Processes documents in chunks to reduce memory usage.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize streaming pipeline."""
        self.config = config or PipelineConfig()
        self.pipeline = DocumentProcessingPipeline(config)

    def process_file_streaming(
        self,
        file_path: Path,
        chunk_callback: Callable[[Chunk], None]
    ) -> ProcessingResult:
        """
        Process file in streaming mode.

        Args:
            file_path: Path to file
            chunk_callback: Callback function for each chunk

        Returns:
            Processing result
        """
        result = self.pipeline.process_file(file_path)

        if result.success and result.chunks:
            for chunk in result.chunks:
                chunk_callback(chunk)

        return result


def create_pipeline(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    ocr_engine: str = 'paddleocr',
    **kwargs
) -> DocumentProcessingPipeline:
    """
    Factory function to create processing pipeline.

    Args:
        chunk_size: Chunk size in characters
        chunk_overlap: Overlap between chunks
        ocr_engine: OCR engine to use
        **kwargs: Additional config parameters

    Returns:
        Configured pipeline instance
    """
    config = PipelineConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        ocr_engine=ocr_engine,
        **kwargs
    )

    return DocumentProcessingPipeline(config)

    @classmethod
    def _get_semantic_model(cls, model_name: str) -> Tuple[Optional[Any], Optional[str]]:
        """Return cached semantic model or load it lazily."""
        if model_name in cls._SEMANTIC_MODEL_CACHE:
            return cls._SEMANTIC_MODEL_CACHE[model_name], None

        if model_name in cls._SEMANTIC_MODEL_ERRORS:
            return None, cls._SEMANTIC_MODEL_ERRORS[model_name]

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            error = "sentence-transformers package is not installed"
            cls._SEMANTIC_MODEL_ERRORS[model_name] = error
            return None, error

        try:
            model = SentenceTransformer(model_name)
        except Exception as err:  # pragma: no cover - depends on environment
            error = str(err)
            cls._SEMANTIC_MODEL_ERRORS[model_name] = error
            return None, error

        cls._SEMANTIC_MODEL_CACHE[model_name] = model
        return model, None

    @classmethod
    def semantic_support_status(cls, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Report semantic chunking availability for diagnostics."""
        configured_name = model_name or os.getenv('DOCTAGS_SEMANTIC_MODEL')
        if not configured_name:
            return {
                'available': False,
                'reason': 'Set DOCTAGS_SEMANTIC_MODEL or PipelineConfig.semantic_model to enable semantic chunking.'
            }

        model, error = cls._get_semantic_model(configured_name)
        if model is None:
            return {
                'available': False,
                'reason': error
            }

        return {
            'available': True,
            'model': configured_name
        }
