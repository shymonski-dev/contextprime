"""
DocTags Processor following IBM Docling approach.

Converts document structure into DocTags format, preserving:
- Hierarchical relationships
- Reading order
- Semantic elements
- Document flow

DocTags can be converted to multiple formats: Markdown, HTML, JSON, etc.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from loguru import logger

from .document_parser import ParsedDocument, DocumentElement


class DocTagType(Enum):
    """Semantic tags for document elements."""
    DOCUMENT = "document"
    TITLE = "title"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    LIST = "list"
    LIST_ITEM = "list_item"
    TABLE = "table"
    TABLE_ROW = "table_row"
    TABLE_CELL = "table_cell"
    FIGURE = "figure"
    CAPTION = "caption"
    CODE = "code"
    EQUATION = "equation"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_BREAK = "page_break"


@dataclass
class DocTag:
    """
    Represents a DocTag element in the document structure.

    Following IBM Docling's approach to preserve document semantics.
    """
    tag_type: DocTagType
    content: str
    tag_id: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    level: Optional[int] = None
    order: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'tag_type': self.tag_type.value,
            'content': self.content,
            'tag_id': self.tag_id,
            'parent_id': self.parent_id,
            'children_ids': self.children_ids,
            'level': self.level,
            'order': self.order,
            'metadata': self.metadata,
            'confidence': self.confidence,
        }


@dataclass
class DocTagsDocument:
    """
    Complete document in DocTags format.

    Maintains hierarchical structure and reading order.
    """
    doc_id: str
    title: str
    tags: List[DocTag]
    metadata: Dict[str, Any]
    hierarchy: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'tags': [tag.to_dict() for tag in self.tags],
            'metadata': self.metadata,
            'hierarchy': self.hierarchy,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_json(self, output_path: Path) -> None:
        """Save to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        logger.info(f"Saved DocTags to {output_path}")


class DocTagsProcessor:
    """
    Processes parsed documents into DocTags format.

    Following IBM Docling's principles:
    - Preserve document structure
    - Maintain hierarchical relationships
    - Keep reading order
    - Tag semantic elements
    - Support multiple output formats
    """

    def __init__(self):
        """Initialize DocTags processor."""
        self.tag_counter = 0

    def process(
        self,
        parsed_doc: ParsedDocument,
        doc_id: Optional[str] = None
    ) -> DocTagsDocument:
        """
        Convert parsed document to DocTags format.

        Args:
            parsed_doc: Parsed document from DocumentParser
            doc_id: Optional document ID (generated if None)

        Returns:
            Document in DocTags format
        """
        self.tag_counter = 0

        # Generate doc ID if not provided
        if doc_id is None:
            doc_id = self._generate_doc_id(parsed_doc)

        # Extract title
        title = self._extract_title(parsed_doc)

        # Convert elements to DocTags
        tags = []
        hierarchy = {'root': [], 'sections': {}}

        # Create root document tag
        doc_tag = self._create_tag(
            tag_type=DocTagType.DOCUMENT,
            content=title,
            parent_id=None
        )
        tags.append(doc_tag)

        # Process elements
        self._process_elements(
            parsed_doc.elements,
            tags,
            hierarchy,
            parent_id=doc_tag.tag_id
        )

        # Update parent-child relationships
        self._build_hierarchy(tags, hierarchy)

        # Prepare metadata
        metadata = parsed_doc.metadata.copy()
        metadata['total_tags'] = len(tags)
        metadata['doctags_version'] = '1.0'

        return DocTagsDocument(
            doc_id=doc_id,
            title=title,
            tags=tags,
            metadata=metadata,
            hierarchy=hierarchy
        )

    def _generate_doc_id(self, parsed_doc: ParsedDocument) -> str:
        """Generate unique document ID."""
        import hashlib
        from datetime import datetime

        # Use file hash if available, otherwise generate from content
        if 'file_hash' in parsed_doc.metadata:
            return parsed_doc.metadata['file_hash'][:16]

        # Hash from content and timestamp
        content_hash = hashlib.sha256(
            parsed_doc.text[:1000].encode('utf-8')
        ).hexdigest()[:12]

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

        return f"doc_{timestamp}_{content_hash}"

    def _extract_title(self, parsed_doc: ParsedDocument) -> str:
        """Extract document title."""
        # Try metadata first
        if 'title' in parsed_doc.metadata and parsed_doc.metadata['title']:
            return parsed_doc.metadata['title']

        # Try filename
        if 'filename' in parsed_doc.metadata:
            filename = parsed_doc.metadata['filename']
            # Remove extension
            title = Path(filename).stem
            return title.replace('_', ' ').replace('-', ' ')

        # Try first heading
        for element in parsed_doc.elements:
            if element.type == 'heading' and element.level == 1:
                return element.content

        # Default
        return "Untitled Document"

    def _create_tag(
        self,
        tag_type: DocTagType,
        content: str,
        parent_id: Optional[str] = None,
        level: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0
    ) -> DocTag:
        """Create a new DocTag with unique ID."""
        tag_id = f"tag_{self.tag_counter:06d}"
        self.tag_counter += 1

        return DocTag(
            tag_type=tag_type,
            content=content,
            tag_id=tag_id,
            parent_id=parent_id,
            level=level,
            order=self.tag_counter,
            metadata=metadata or {},
            confidence=confidence
        )

    def _process_elements(
        self,
        elements: List[DocumentElement],
        tags: List[DocTag],
        hierarchy: Dict[str, Any],
        parent_id: str,
        current_section: Optional[str] = None
    ) -> None:
        """
        Process document elements into DocTags.

        Args:
            elements: List of document elements
            tags: List to append tags to
            hierarchy: Hierarchy structure
            parent_id: Parent tag ID
            current_section: Current section ID
        """
        for element in elements:
            tag = self._element_to_doctag(element, parent_id, current_section)

            if tag:
                tags.append(tag)

                # Update hierarchy
                if tag.tag_type in [DocTagType.SECTION, DocTagType.SUBSECTION]:
                    current_section = tag.tag_id
                    hierarchy['sections'][tag.tag_id] = {
                        'title': tag.content,
                        'level': tag.level,
                        'children': []
                    }

                if current_section:
                    if current_section in hierarchy['sections']:
                        hierarchy['sections'][current_section]['children'].append(tag.tag_id)

                # Process children recursively
                if element.children:
                    self._process_elements(
                        element.children,
                        tags,
                        hierarchy,
                        parent_id=tag.tag_id,
                        current_section=current_section
                    )

    def _element_to_doctag(
        self,
        element: DocumentElement,
        parent_id: str,
        current_section: Optional[str] = None
    ) -> Optional[DocTag]:
        """
        Convert DocumentElement to DocTag.

        Args:
            element: Document element
            parent_id: Parent tag ID
            current_section: Current section ID

        Returns:
            DocTag or None if element should be skipped
        """
        # Map element type to DocTag type
        type_mapping = {
            'heading': self._heading_to_doctag,
            'paragraph': self._paragraph_to_doctag,
            'list': self._list_to_doctag,
            'table': self._table_to_doctag,
            'figure': self._figure_to_doctag,
            'code': self._code_to_doctag,
            'equation': self._equation_to_doctag,
        }

        converter = type_mapping.get(element.type)
        if converter:
            return converter(element, parent_id, current_section)

        # Default: treat as paragraph
        return self._paragraph_to_doctag(element, parent_id, current_section)

    def _heading_to_doctag(
        self,
        element: DocumentElement,
        parent_id: str,
        current_section: Optional[str]
    ) -> DocTag:
        """Convert heading to DocTag."""
        level = element.level or 1

        # Determine if section or subsection
        if level == 1:
            tag_type = DocTagType.TITLE
        elif level == 2:
            tag_type = DocTagType.SECTION
        else:
            tag_type = DocTagType.SUBSECTION

        confidence = element.metadata.get('confidence', 1.0)

        return self._create_tag(
            tag_type=tag_type,
            content=element.content,
            parent_id=parent_id,
            level=level,
            metadata=element.metadata,
            confidence=confidence
        )

    def _paragraph_to_doctag(
        self,
        element: DocumentElement,
        parent_id: str,
        current_section: Optional[str]
    ) -> DocTag:
        """Convert paragraph to DocTag."""
        confidence = element.metadata.get('confidence', 1.0)

        return self._create_tag(
            tag_type=DocTagType.PARAGRAPH,
            content=element.content,
            parent_id=parent_id,
            metadata=element.metadata,
            confidence=confidence
        )

    def _list_to_doctag(
        self,
        element: DocumentElement,
        parent_id: str,
        current_section: Optional[str]
    ) -> DocTag:
        """Convert list to DocTag."""
        confidence = element.metadata.get('confidence', 1.0)

        return self._create_tag(
            tag_type=DocTagType.LIST,
            content=element.content,
            parent_id=parent_id,
            metadata=element.metadata,
            confidence=confidence
        )

    def _table_to_doctag(
        self,
        element: DocumentElement,
        parent_id: str,
        current_section: Optional[str]
    ) -> DocTag:
        """Convert table to DocTag."""
        confidence = element.metadata.get('confidence', 1.0)

        return self._create_tag(
            tag_type=DocTagType.TABLE,
            content=element.content,
            parent_id=parent_id,
            metadata=element.metadata,
            confidence=confidence
        )

    def _figure_to_doctag(
        self,
        element: DocumentElement,
        parent_id: str,
        current_section: Optional[str]
    ) -> DocTag:
        """Convert figure to DocTag."""
        confidence = element.metadata.get('confidence', 1.0)

        return self._create_tag(
            tag_type=DocTagType.FIGURE,
            content=element.content,
            parent_id=parent_id,
            metadata=element.metadata,
            confidence=confidence
        )

    def _code_to_doctag(
        self,
        element: DocumentElement,
        parent_id: str,
        current_section: Optional[str]
    ) -> DocTag:
        """Convert code block to DocTag."""
        confidence = element.metadata.get('confidence', 1.0)

        return self._create_tag(
            tag_type=DocTagType.CODE,
            content=element.content,
            parent_id=parent_id,
            metadata=element.metadata,
            confidence=confidence
        )

    def _equation_to_doctag(
        self,
        element: DocumentElement,
        parent_id: str,
        current_section: Optional[str]
    ) -> DocTag:
        """Convert equation to DocTag."""
        confidence = element.metadata.get('confidence', 1.0)

        return self._create_tag(
            tag_type=DocTagType.EQUATION,
            content=element.content,
            parent_id=parent_id,
            metadata=element.metadata,
            confidence=confidence
        )

    def _build_hierarchy(
        self,
        tags: List[DocTag],
        hierarchy: Dict[str, Any]
    ) -> None:
        """
        Build parent-child relationships in tags.

        Args:
            tags: List of all tags
            hierarchy: Hierarchy structure
        """
        # Create tag lookup
        tag_lookup = {tag.tag_id: tag for tag in tags}

        # Build children lists
        for tag in tags:
            if tag.parent_id and tag.parent_id in tag_lookup:
                parent = tag_lookup[tag.parent_id]
                if tag.tag_id not in parent.children_ids:
                    parent.children_ids.append(tag.tag_id)


class DocTagsConverter:
    """
    Convert DocTags to various output formats.

    Supports: Markdown, HTML, plain text, JSON
    """

    @staticmethod
    def to_markdown(doc: DocTagsDocument) -> str:
        """
        Convert DocTags to Markdown format.

        Args:
            doc: DocTags document

        Returns:
            Markdown string
        """
        lines = []

        # Add title
        lines.append(f"# {doc.title}\n")

        # Add metadata as comment
        lines.append("<!--")
        lines.append(f"Document ID: {doc.doc_id}")
        if 'author' in doc.metadata:
            lines.append(f"Author: {doc.metadata['author']}")
        lines.append("-->\n")

        # Process tags in order
        for tag in doc.tags[1:]:  # Skip document tag
            md_line = DocTagsConverter._tag_to_markdown(tag)
            if md_line:
                lines.append(md_line)

        return '\n'.join(lines)

    @staticmethod
    def _tag_to_markdown(tag: DocTag) -> str:
        """Convert single tag to Markdown."""
        if tag.tag_type == DocTagType.TITLE:
            return f"\n# {tag.content}\n"

        elif tag.tag_type == DocTagType.SECTION:
            level = tag.level or 2
            return f"\n{'#' * level} {tag.content}\n"

        elif tag.tag_type == DocTagType.SUBSECTION:
            level = tag.level or 3
            return f"\n{'#' * level} {tag.content}\n"

        elif tag.tag_type == DocTagType.PARAGRAPH:
            return f"\n{tag.content}\n"

        elif tag.tag_type == DocTagType.LIST:
            return f"\n{tag.content}\n"

        elif tag.tag_type == DocTagType.TABLE:
            return f"\n{tag.content}\n"

        elif tag.tag_type == DocTagType.CODE:
            return f"\n```\n{tag.content}\n```\n"

        elif tag.tag_type == DocTagType.EQUATION:
            return f"\n$$\n{tag.content}\n$$\n"

        else:
            return f"\n{tag.content}\n"

    @staticmethod
    def to_html(doc: DocTagsDocument) -> str:
        """
        Convert DocTags to HTML format.

        Args:
            doc: DocTags document

        Returns:
            HTML string
        """
        lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"  <title>{doc.title}</title>",
            "  <meta charset='utf-8'>",
            "</head>",
            "<body>",
            f"  <h1>{doc.title}</h1>",
        ]

        # Process tags
        for tag in doc.tags[1:]:  # Skip document tag
            html_line = DocTagsConverter._tag_to_html(tag)
            if html_line:
                lines.append(f"  {html_line}")

        lines.extend([
            "</body>",
            "</html>"
        ])

        return '\n'.join(lines)

    @staticmethod
    def _tag_to_html(tag: DocTag) -> str:
        """Convert single tag to HTML."""
        if tag.tag_type == DocTagType.TITLE:
            return f"<h1>{tag.content}</h1>"

        elif tag.tag_type == DocTagType.SECTION:
            level = min(tag.level or 2, 6)
            return f"<h{level}>{tag.content}</h{level}>"

        elif tag.tag_type == DocTagType.SUBSECTION:
            level = min(tag.level or 3, 6)
            return f"<h{level}>{tag.content}</h{level}>"

        elif tag.tag_type == DocTagType.PARAGRAPH:
            return f"<p>{tag.content}</p>"

        elif tag.tag_type == DocTagType.LIST:
            # Convert markdown list to HTML
            items = tag.content.split('\n')
            html_items = [f"  <li>{item.lstrip('- ').lstrip('* ')}</li>" for item in items if item.strip()]
            return "<ul>\n" + '\n'.join(html_items) + "\n</ul>"

        elif tag.tag_type == DocTagType.TABLE:
            # Assume table is in HTML format already
            return tag.content

        elif tag.tag_type == DocTagType.CODE:
            return f"<pre><code>{tag.content}</code></pre>"

        elif tag.tag_type == DocTagType.EQUATION:
            return f"<div class='equation'>{tag.content}</div>"

        else:
            return f"<div>{tag.content}</div>"

    @staticmethod
    def to_text(doc: DocTagsDocument) -> str:
        """
        Convert DocTags to plain text.

        Args:
            doc: DocTags document

        Returns:
            Plain text string
        """
        lines = [doc.title, "=" * len(doc.title), ""]

        for tag in doc.tags[1:]:  # Skip document tag
            if tag.tag_type in [DocTagType.TITLE, DocTagType.SECTION, DocTagType.SUBSECTION]:
                lines.append("")
                lines.append(tag.content)
                lines.append("-" * len(tag.content))
                lines.append("")
            else:
                lines.append(tag.content)
                lines.append("")

        return '\n'.join(lines)
