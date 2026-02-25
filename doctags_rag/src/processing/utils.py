"""
Processing utilities for document handling.

Provides helper functions for:
- File type detection
- Text cleaning and normalization
- Language detection
- Encoding detection
- Image preprocessing
- Table extraction
"""

import os
import re
import mimetypes
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import hashlib

import numpy as np
from loguru import logger

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PIL_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # Ensure consistent results
    LANGDETECT_AVAILABLE = True
except ImportError:
    logger.warning("langdetect not installed, language detection disabled")
    LANGDETECT_AVAILABLE = False


class FileTypeDetector:
    """Detects file types using multiple methods."""

    # MIME type mappings
    MIME_TO_EXT = {
        'application/pdf': 'pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'application/msword': 'doc',
        'text/html': 'html',
        'text/plain': 'txt',
        'text/markdown': 'md',
        'image/png': 'png',
        'image/jpeg': 'jpg',
        'image/jpg': 'jpg',
        'image/tiff': 'tiff',
    }

    # Magic bytes for file type detection
    MAGIC_BYTES = {
        b'%PDF': 'pdf',
        b'PK\x03\x04': 'docx',  # DOCX is a ZIP archive
        b'\x89PNG': 'png',
        b'\xff\xd8\xff': 'jpg',
        b'GIF89a': 'gif',
        b'GIF87a': 'gif',
    }

    @classmethod
    def detect_file_type(cls, file_path: Path) -> str:
        """
        Detect file type using multiple methods.

        Args:
            file_path: Path to the file

        Returns:
            File extension (e.g., 'pdf', 'docx')
        """
        # Try extension first
        ext = file_path.suffix.lower().lstrip('.')
        if ext:
            return ext

        # Try MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type and mime_type in cls.MIME_TO_EXT:
            return cls.MIME_TO_EXT[mime_type]

        # Try magic bytes
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
                for magic, file_type in cls.MAGIC_BYTES.items():
                    if header.startswith(magic):
                        return file_type
        except Exception as e:
            logger.warning(f"Failed to read file header: {e}")

        # Default to extension or 'unknown'
        return ext or 'unknown'

    @classmethod
    def is_supported(cls, file_path: Path, supported_formats: List[str]) -> bool:
        """
        Check if file type is supported.

        Args:
            file_path: Path to the file
            supported_formats: List of supported extensions

        Returns:
            True if file type is supported
        """
        file_type = cls.detect_file_type(file_path).lower()
        supported = {fmt.lower() for fmt in supported_formats}
        return file_type in supported

    @classmethod
    def get_file_size_mb(cls, file_path: Path) -> float:
        """Get file size in megabytes."""
        return file_path.stat().st_size / (1024 * 1024)


class TextCleaner:
    """Clean and normalize text content."""

    @staticmethod
    def clean_text(text: str, aggressive: bool = False) -> str:
        """
        Clean and normalize text.

        Args:
            text: Input text
            aggressive: If True, apply more aggressive cleaning

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove null bytes
        text = text.replace('\x00', '')

        # Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)

        # Remove excessive blank lines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))

        if aggressive:
            # Remove control characters except newlines and tabs
            text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)

            # Normalize unicode whitespace
            text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)

            # Remove excessive spaces
            text = re.sub(r' {2,}', ' ', text)

        return text.strip()

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters."""
        import unicodedata
        return unicodedata.normalize('NFKC', text)

    @staticmethod
    def remove_urls(text: str) -> str:
        """Remove URLs from text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.sub(url_pattern, '', text)

    @staticmethod
    def remove_emails(text: str) -> str:
        """Remove email addresses from text."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.sub(email_pattern, '', text)

    @staticmethod
    def extract_numbers(text: str) -> List[float]:
        """Extract all numbers from text."""
        return [float(num) for num in re.findall(r'-?\d+\.?\d*', text)]


class LanguageDetector:
    """Detect language of text."""

    @staticmethod
    def detect_language(text: str, min_length: int = 50) -> Optional[str]:
        """
        Detect language of text.

        Args:
            text: Input text
            min_length: Minimum text length for reliable detection

        Returns:
            ISO 639-1 language code (e.g., 'en', 'es') or None
        """
        if not LANGDETECT_AVAILABLE:
            return None

        if not text or len(text) < min_length:
            return None

        try:
            return detect(text)
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
            return None

    @staticmethod
    def is_english(text: str) -> bool:
        """Check if text is in English."""
        lang = LanguageDetector.detect_language(text)
        return lang == 'en' if lang else False


class EncodingDetector:
    """Detect text encoding."""

    @staticmethod
    def detect_encoding(file_path: Path) -> str:
        """
        Detect file encoding.

        Args:
            file_path: Path to the file

        Returns:
            Encoding name (e.g., 'utf-8', 'latin-1')
        """
        try:
            import chardet

            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except ImportError:
            logger.warning("chardet not installed, defaulting to utf-8")
            return 'utf-8'
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, defaulting to utf-8")
            return 'utf-8'

    @staticmethod
    def read_text_file(file_path: Path, encoding: Optional[str] = None) -> str:
        """
        Read text file with automatic encoding detection.

        Args:
            file_path: Path to the file
            encoding: Explicit encoding (auto-detect if None)

        Returns:
            File content as string
        """
        if encoding is None:
            encoding = EncodingDetector.detect_encoding(file_path)

        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 which accepts all byte values
            logger.warning(f"Failed to decode with {encoding}, trying latin-1")
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()


class ImagePreprocessor:
    """Preprocess images for OCR."""

    @staticmethod
    def load_image(image_path: Path) -> np.ndarray:
        """
        Load image as numpy array.

        Args:
            image_path: Path to the image

        Returns:
            Image as numpy array
        """
        img = Image.open(image_path)

        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return np.array(img)

    @staticmethod
    def preprocess_for_ocr(
        image: np.ndarray,
        resize_factor: float = 2.0,
        denoise: bool = True,
        sharpen: bool = True,
        binarize: bool = False
    ) -> np.ndarray:
        """
        Preprocess image for better OCR results.

        Args:
            image: Input image as numpy array
            resize_factor: Factor to resize image (larger = better OCR)
            denoise: Apply denoising
            sharpen: Apply sharpening
            binarize: Apply binarization (threshold)

        Returns:
            Preprocessed image
        """
        # Resize for better OCR
        if resize_factor != 1.0:
            height, width = image.shape[:2]
            new_size = (int(width * resize_factor), int(height * resize_factor))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Denoise
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=10)

        # Sharpen
        if sharpen:
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            gray = cv2.filter2D(gray, -1, kernel)

        # Binarize
        if binarize:
            gray = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11, 2
            )

        return gray

    @staticmethod
    def correct_orientation(image: np.ndarray) -> np.ndarray:
        """
        Detect and correct image orientation.

        Args:
            image: Input image

        Returns:
            Corrected image
        """
        # This is a simplified version
        # In production, you might want to use a more sophisticated method
        try:
            from PIL import Image as PILImage
            from PIL import ExifTags

            pil_image = PILImage.fromarray(image)

            # Get EXIF orientation
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break

            exif = pil_image._getexif()
            if exif is not None:
                orientation_value = exif.get(orientation)

                if orientation_value == 3:
                    pil_image = pil_image.rotate(180, expand=True)
                elif orientation_value == 6:
                    pil_image = pil_image.rotate(270, expand=True)
                elif orientation_value == 8:
                    pil_image = pil_image.rotate(90, expand=True)

            return np.array(pil_image)
        except Exception as e:
            logger.debug(f"Could not correct orientation: {e}")
            return image

    @staticmethod
    def get_image_info(image_path: Path) -> Dict[str, Any]:
        """
        Get image metadata.

        Args:
            image_path: Path to the image

        Returns:
            Dictionary with image information
        """
        img = Image.open(image_path)

        return {
            'width': img.width,
            'height': img.height,
            'format': img.format,
            'mode': img.mode,
            'size_mb': image_path.stat().st_size / (1024 * 1024),
        }


class TableExtractor:
    """Extract and structure table data."""

    @staticmethod
    def detect_table_structure(cells: List[List[str]]) -> Dict[str, Any]:
        """
        Analyze table structure.

        Args:
            cells: 2D list of cell contents

        Returns:
            Table structure information
        """
        if not cells:
            return {
                'rows': 0,
                'cols': 0,
                'has_header': False,
            }

        num_rows = len(cells)
        num_cols = max(len(row) for row in cells) if cells else 0

        # Simple heuristic: if first row has different formatting or all bold
        has_header = False
        if num_rows > 1:
            # Check if first row is likely a header
            first_row_empty = all(not cell.strip() for cell in cells[0])
            if not first_row_empty and num_rows > 1:
                has_header = True  # Assume header if non-empty

        return {
            'rows': num_rows,
            'cols': num_cols,
            'has_header': has_header,
        }

    @staticmethod
    def table_to_markdown(cells: List[List[str]], has_header: bool = True) -> str:
        """
        Convert table to markdown format.

        Args:
            cells: 2D list of cell contents
            has_header: Whether first row is header

        Returns:
            Markdown table string
        """
        if not cells:
            return ""

        # Ensure all rows have same number of columns
        max_cols = max(len(row) for row in cells)
        normalized_cells = [
            row + [''] * (max_cols - len(row))
            for row in cells
        ]

        lines = []

        # Add header
        if has_header and len(normalized_cells) > 0:
            header = '| ' + ' | '.join(normalized_cells[0]) + ' |'
            lines.append(header)
            separator = '|' + '|'.join(['---'] * max_cols) + '|'
            lines.append(separator)
            data_rows = normalized_cells[1:]
        else:
            data_rows = normalized_cells

        # Add data rows
        for row in data_rows:
            row_str = '| ' + ' | '.join(row) + ' |'
            lines.append(row_str)

        return '\n'.join(lines)

    @staticmethod
    def table_to_html(cells: List[List[str]], has_header: bool = True) -> str:
        """
        Convert table to HTML format.

        Args:
            cells: 2D list of cell contents
            has_header: Whether first row is header

        Returns:
            HTML table string
        """
        if not cells:
            return ""

        lines = ['<table>']

        if has_header and len(cells) > 0:
            lines.append('  <thead>')
            lines.append('    <tr>')
            for cell in cells[0]:
                lines.append(f'      <th>{cell}</th>')
            lines.append('    </tr>')
            lines.append('  </thead>')
            data_rows = cells[1:]
        else:
            data_rows = cells

        lines.append('  <tbody>')
        for row in data_rows:
            lines.append('    <tr>')
            for cell in row:
                lines.append(f'      <td>{cell}</td>')
            lines.append('    </tr>')
        lines.append('  </tbody>')
        lines.append('</table>')

        return '\n'.join(lines)


class ContentHasher:
    """Generate content hashes for deduplication."""

    @staticmethod
    def hash_content(content: str, algorithm: str = 'sha256') -> str:
        """
        Generate hash of content.

        Args:
            content: Content to hash
            algorithm: Hash algorithm (md5, sha1, sha256)

        Returns:
            Hexadecimal hash string
        """
        hash_obj = hashlib.new(algorithm)
        hash_obj.update(content.encode('utf-8'))
        return hash_obj.hexdigest()

    @staticmethod
    def hash_file(file_path: Path, algorithm: str = 'sha256') -> str:
        """
        Generate hash of file.

        Args:
            file_path: Path to the file
            algorithm: Hash algorithm

        Returns:
            Hexadecimal hash string
        """
        hash_obj = hashlib.new(algorithm)

        with open(file_path, 'rb') as f:
            # Read in chunks for large files
            for chunk in iter(lambda: f.read(8192), b''):
                hash_obj.update(chunk)

        return hash_obj.hexdigest()


class DocumentMetadataExtractor:
    """Extract metadata from documents."""

    @staticmethod
    def extract_basic_metadata(file_path: Path) -> Dict[str, Any]:
        """
        Extract basic file metadata.

        Args:
            file_path: Path to the file

        Returns:
            Metadata dictionary
        """
        stat = file_path.stat()

        return {
            'filename': file_path.name,
            'extension': file_path.suffix.lstrip('.'),
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created_at': stat.st_ctime,
            'modified_at': stat.st_mtime,
            'file_hash': ContentHasher.hash_file(file_path),
        }

    @staticmethod
    def extract_text_statistics(text: str) -> Dict[str, Any]:
        """
        Extract statistics from text.

        Args:
            text: Input text

        Returns:
            Statistics dictionary
        """
        if not text:
            return {
                'char_count': 0,
                'word_count': 0,
                'line_count': 0,
                'paragraph_count': 0,
            }

        # Count characters (excluding whitespace)
        char_count = len(text)
        char_count_no_spaces = len(text.replace(' ', '').replace('\n', ''))

        # Count words
        words = text.split()
        word_count = len(words)

        # Count lines
        lines = text.split('\n')
        line_count = len(lines)

        # Count paragraphs (separated by blank lines)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraph_count = len([p for p in paragraphs if p.strip()])

        # Average word length
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0

        return {
            'char_count': char_count,
            'char_count_no_spaces': char_count_no_spaces,
            'word_count': word_count,
            'line_count': line_count,
            'paragraph_count': paragraph_count,
            'avg_word_length': round(avg_word_length, 2),
        }
