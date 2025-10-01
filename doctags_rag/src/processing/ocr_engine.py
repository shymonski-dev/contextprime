"""
OCR Engine for document processing using PaddleOCR.

Provides comprehensive OCR capabilities including:
- Text detection and recognition
- Layout analysis
- Table structure recognition
- Multi-language support
- Confidence scoring
- Batch processing
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
from PIL import Image
from loguru import logger

from .utils import ImagePreprocessor


@dataclass
class OCRBox:
    """Represents a detected text box."""
    text: str
    bbox: List[List[float]]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float
    line_num: Optional[int] = None


@dataclass
class OCRResult:
    """Complete OCR result for a document."""
    text: str
    boxes: List[OCRBox]
    layout: Optional[Dict[str, Any]] = None
    tables: Optional[List[Dict[str, Any]]] = None
    confidence: float = 0.0
    processing_time: float = 0.0
    language: str = 'en'


class PaddleOCREngine:
    """
    PaddleOCR-based OCR engine with advanced features.

    Supports:
    - Multi-language text recognition
    - Layout analysis
    - Table detection and structure recognition
    - Angle classification for rotation correction
    - Batch processing
    """

    def __init__(
        self,
        lang: str = 'en',
        use_gpu: bool = False,
        use_angle_cls: bool = True,
        det_db_thresh: float = 0.3,
        rec_batch_num: int = 6,
        show_log: bool = False
    ):
        """
        Initialize PaddleOCR engine.

        Args:
            lang: Language code (en, ch, fr, etc.)
            use_gpu: Use GPU acceleration
            use_angle_cls: Enable angle classification
            det_db_thresh: Detection threshold
            rec_batch_num: Recognition batch size
            show_log: Show PaddleOCR logs
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.use_angle_cls = use_angle_cls

        try:
            from paddleocr import PaddleOCR

            self.ocr = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang,
                use_gpu=use_gpu,
                det_db_thresh=det_db_thresh,
                rec_batch_num=rec_batch_num,
                show_log=show_log,
                use_dilation=True,  # Better for dense text
            )

            logger.info(f"PaddleOCR initialized (lang={lang}, gpu={use_gpu})")
        except ImportError:
            logger.error("PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    def process_image(
        self,
        image_path: Path,
        preprocess: bool = True,
        detect_layout: bool = True
    ) -> OCRResult:
        """
        Process image with OCR.

        Args:
            image_path: Path to the image
            preprocess: Apply image preprocessing
            detect_layout: Detect document layout

        Returns:
            OCR result with detected text and metadata
        """
        start_time = time.time()

        try:
            # Load image
            image = ImagePreprocessor.load_image(image_path)

            # Preprocess if requested
            if preprocess:
                image = ImagePreprocessor.preprocess_for_ocr(
                    image,
                    resize_factor=1.5,
                    denoise=True,
                    sharpen=True,
                    binarize=False
                )

            # Run OCR
            result = self.ocr.ocr(image, cls=self.use_angle_cls)

            # Parse results
            boxes = []
            full_text_lines = []
            total_confidence = 0.0
            box_count = 0

            if result and result[0]:
                for line_num, detection in enumerate(result[0]):
                    if detection:
                        bbox = detection[0]  # Bounding box coordinates
                        text_info = detection[1]  # (text, confidence)

                        if isinstance(text_info, tuple) and len(text_info) == 2:
                            text, confidence = text_info
                        else:
                            text = str(text_info)
                            confidence = 1.0

                        boxes.append(OCRBox(
                            text=text,
                            bbox=bbox,
                            confidence=confidence,
                            line_num=line_num
                        ))

                        full_text_lines.append(text)
                        total_confidence += confidence
                        box_count += 1

            # Combine text
            full_text = '\n'.join(full_text_lines)
            avg_confidence = total_confidence / box_count if box_count > 0 else 0.0

            # Detect layout if requested
            layout = None
            if detect_layout:
                layout = self._analyze_layout(boxes, image.shape)

            processing_time = time.time() - start_time

            return OCRResult(
                text=full_text,
                boxes=boxes,
                layout=layout,
                confidence=avg_confidence,
                processing_time=processing_time,
                language=self.lang
            )

        except Exception as e:
            logger.error(f"OCR processing failed for {image_path}: {e}")
            raise

    def process_pdf_page(
        self,
        page_image: np.ndarray,
        page_num: int,
        detect_layout: bool = True
    ) -> OCRResult:
        """
        Process a PDF page image.

        Args:
            page_image: Page image as numpy array
            page_num: Page number
            detect_layout: Detect document layout

        Returns:
            OCR result for the page
        """
        start_time = time.time()

        try:
            # Run OCR
            result = self.ocr.ocr(page_image, cls=self.use_angle_cls)

            # Parse results
            boxes = []
            full_text_lines = []
            total_confidence = 0.0
            box_count = 0

            if result and result[0]:
                for line_num, detection in enumerate(result[0]):
                    if detection:
                        bbox = detection[0]
                        text_info = detection[1]

                        if isinstance(text_info, tuple) and len(text_info) == 2:
                            text, confidence = text_info
                        else:
                            text = str(text_info)
                            confidence = 1.0

                        boxes.append(OCRBox(
                            text=text,
                            bbox=bbox,
                            confidence=confidence,
                            line_num=line_num
                        ))

                        full_text_lines.append(text)
                        total_confidence += confidence
                        box_count += 1

            full_text = '\n'.join(full_text_lines)
            avg_confidence = total_confidence / box_count if box_count > 0 else 0.0

            # Detect layout
            layout = None
            if detect_layout:
                layout = self._analyze_layout(boxes, page_image.shape)

            processing_time = time.time() - start_time

            return OCRResult(
                text=full_text,
                boxes=boxes,
                layout=layout,
                confidence=avg_confidence,
                processing_time=processing_time,
                language=self.lang
            )

        except Exception as e:
            logger.error(f"OCR processing failed for page {page_num}: {e}")
            raise

    def _analyze_layout(
        self,
        boxes: List[OCRBox],
        image_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """
        Analyze document layout from OCR boxes.

        Args:
            boxes: List of detected text boxes
            image_shape: Image dimensions

        Returns:
            Layout analysis dictionary
        """
        if not boxes:
            return {
                'reading_order': [],
                'columns': 1,
                'text_regions': [],
            }

        height, width = image_shape[:2]

        # Sort boxes by vertical position (top to bottom)
        sorted_boxes = sorted(boxes, key=lambda b: b.bbox[0][1])

        # Detect columns (simple heuristic)
        x_positions = [b.bbox[0][0] for b in sorted_boxes]
        x_avg = sum(x_positions) / len(x_positions)

        # Count boxes on left vs right
        left_count = sum(1 for x in x_positions if x < width / 2)
        right_count = len(x_positions) - left_count

        # Simple column detection
        num_columns = 2 if min(left_count, right_count) > len(boxes) * 0.3 else 1

        # Determine reading order
        if num_columns == 1:
            reading_order = list(range(len(sorted_boxes)))
        else:
            # Two-column layout: sort by columns
            left_boxes = [(i, b) for i, b in enumerate(sorted_boxes) if b.bbox[0][0] < width / 2]
            right_boxes = [(i, b) for i, b in enumerate(sorted_boxes) if b.bbox[0][0] >= width / 2]

            # Sort each column by y position
            left_boxes.sort(key=lambda x: x[1].bbox[0][1])
            right_boxes.sort(key=lambda x: x[1].bbox[0][1])

            reading_order = [i for i, _ in left_boxes] + [i for i, _ in right_boxes]

        # Identify text regions (headers, body, footer)
        text_regions = self._identify_text_regions(sorted_boxes, height)

        return {
            'reading_order': reading_order,
            'columns': num_columns,
            'text_regions': text_regions,
            'image_width': width,
            'image_height': height,
        }

    def _identify_text_regions(
        self,
        boxes: List[OCRBox],
        image_height: int
    ) -> List[Dict[str, Any]]:
        """
        Identify text regions (header, body, footer).

        Args:
            boxes: List of OCR boxes
            image_height: Image height

        Returns:
            List of text regions
        """
        if not boxes:
            return []

        regions = []

        # Define region boundaries
        header_threshold = image_height * 0.15
        footer_threshold = image_height * 0.85

        for box in boxes:
            y_pos = box.bbox[0][1]

            if y_pos < header_threshold:
                region_type = 'header'
            elif y_pos > footer_threshold:
                region_type = 'footer'
            else:
                region_type = 'body'

            regions.append({
                'type': region_type,
                'text': box.text,
                'bbox': box.bbox,
                'confidence': box.confidence,
            })

        return regions

    def detect_tables(
        self,
        image_path: Path,
        confidence_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Detect and extract table structures.

        Args:
            image_path: Path to the image
            confidence_threshold: Minimum confidence for detection

        Returns:
            List of detected tables with structure
        """
        try:
            # Try to use PaddleOCR's table recognition if available
            from paddleocr import PPStructure

            table_engine = PPStructure(
                show_log=False,
                use_gpu=self.use_gpu,
                lang=self.lang
            )

            image = str(image_path)
            result = table_engine(image)

            tables = []
            for item in result:
                if item['type'] == 'table':
                    tables.append({
                        'bbox': item.get('bbox', []),
                        'html': item.get('res', {}).get('html', ''),
                        'confidence': item.get('score', 0.0),
                    })

            logger.info(f"Detected {len(tables)} tables in {image_path}")
            return tables

        except ImportError:
            logger.warning("PPStructure not available for table detection")
            return []
        except Exception as e:
            logger.error(f"Table detection failed: {e}")
            return []

    def batch_process(
        self,
        image_paths: List[Path],
        preprocess: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[OCRResult]:
        """
        Process multiple images in batch.

        Args:
            image_paths: List of image paths
            preprocess: Apply preprocessing
            progress_callback: Optional callback for progress updates

        Returns:
            List of OCR results
        """
        results = []

        for idx, image_path in enumerate(image_paths):
            try:
                result = self.process_image(image_path, preprocess=preprocess)
                results.append(result)

                if progress_callback:
                    progress_callback(idx + 1, len(image_paths))

                logger.info(f"Processed {idx + 1}/{len(image_paths)}: {image_path.name}")

            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                # Add empty result for failed processing
                results.append(OCRResult(
                    text="",
                    boxes=[],
                    confidence=0.0,
                    processing_time=0.0,
                    language=self.lang
                ))

        return results


class TesseractOCREngine:
    """
    Fallback OCR engine using Tesseract.

    Used when PaddleOCR is not available or fails.
    """

    def __init__(self, lang: str = 'eng'):
        """
        Initialize Tesseract OCR engine.

        Args:
            lang: Language code (eng, fra, deu, etc.)
        """
        self.lang = lang

        try:
            import pytesseract
            self.pytesseract = pytesseract

            # Test if Tesseract is installed
            pytesseract.get_tesseract_version()

            logger.info(f"Tesseract OCR initialized (lang={lang})")
        except ImportError:
            logger.error("pytesseract not installed. Install with: pip install pytesseract")
            raise
        except Exception as e:
            logger.error(f"Tesseract not found. Please install Tesseract OCR: {e}")
            raise

    def process_image(
        self,
        image_path: Path,
        preprocess: bool = True
    ) -> OCRResult:
        """
        Process image with Tesseract OCR.

        Args:
            image_path: Path to the image
            preprocess: Apply image preprocessing

        Returns:
            OCR result
        """
        start_time = time.time()

        try:
            # Load image
            image = Image.open(image_path)

            # Preprocess if requested
            if preprocess:
                image_np = np.array(image)
                image_np = ImagePreprocessor.preprocess_for_ocr(
                    image_np,
                    resize_factor=2.0,
                    denoise=True,
                    sharpen=True,
                    binarize=True
                )
                image = Image.fromarray(image_np)

            # Extract text with detailed data
            data = self.pytesseract.image_to_data(
                image,
                lang=self.lang,
                output_type=self.pytesseract.Output.DICT
            )

            # Parse results
            boxes = []
            full_text_lines = []
            total_confidence = 0.0
            box_count = 0

            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                if text:
                    conf = float(data['conf'][i])
                    if conf > 0:  # Filter out low confidence
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                        # Convert to bbox format
                        bbox = [
                            [x, y],
                            [x + w, y],
                            [x + w, y + h],
                            [x, y + h]
                        ]

                        boxes.append(OCRBox(
                            text=text,
                            bbox=bbox,
                            confidence=conf / 100.0,  # Normalize to 0-1
                            line_num=data['line_num'][i]
                        ))

                        full_text_lines.append(text)
                        total_confidence += conf / 100.0
                        box_count += 1

            # Get full text
            full_text = self.pytesseract.image_to_string(image, lang=self.lang)
            avg_confidence = total_confidence / box_count if box_count > 0 else 0.0

            processing_time = time.time() - start_time

            return OCRResult(
                text=full_text,
                boxes=boxes,
                confidence=avg_confidence,
                processing_time=processing_time,
                language=self.lang
            )

        except Exception as e:
            logger.error(f"Tesseract OCR failed for {image_path}: {e}")
            raise


class OCREngineFactory:
    """Factory for creating OCR engines with fallback support."""

    @staticmethod
    def create_engine(
        engine_type: Optional[str] = 'paddleocr',
        lang: str = 'en',
        **kwargs
    ) -> Optional[Union['PaddleOCREngine', 'TesseractOCREngine']]:
        """
        Create OCR engine with automatic fallback.

        Args:
            engine_type: Engine type ('paddleocr' or 'tesseract')
            lang: Language code
            **kwargs: Additional engine-specific parameters

        Returns:
            OCR engine instance
        """
        if not engine_type or engine_type.lower() in {'none', 'disabled'}:
            logger.info("OCR engine creation skipped by configuration")
            return None

        if engine_type == 'paddleocr':
            try:
                return PaddleOCREngine(lang=lang, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to initialize PaddleOCR: {e}, falling back to Tesseract")
                return TesseractOCREngine(lang=lang if lang == 'en' else 'eng')

        elif engine_type == 'tesseract':
            # Map language codes
            lang_map = {'en': 'eng', 'zh': 'chi_sim', 'fr': 'fra', 'de': 'deu'}
            tesseract_lang = lang_map.get(lang, 'eng')
            return TesseractOCREngine(lang=tesseract_lang)

        else:
            raise ValueError(f"Unknown OCR engine type: {engine_type}")
