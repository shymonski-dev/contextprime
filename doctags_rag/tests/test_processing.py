"""
Comprehensive tests for document processing module.

Tests all components:
- Document parsing (PDF, DOCX, HTML, TXT, images)
- OCR functionality
- DocTags generation
- Chunking strategies
- End-to-end pipeline
"""

import pytest
from pathlib import Path
import tempfile
import json

from src.processing import (
    DocumentParser,
    ParsedDocument,
    DocumentElement,
    OCREngineFactory,
    DocTagsProcessor,
    DocTagsDocument,
    DocTagType,
    DocTagsConverter,
    StructurePreservingChunker,
    Chunk,
    DocumentProcessingPipeline,
    PipelineConfig,
    ProcessingStage,
    create_pipeline,
    FileTypeDetector,
    TextCleaner,
    TableExtractor,
)


class TestFileTypeDetector:
    """Test file type detection."""

    def test_detect_pdf(self, tmp_path):
        """Test PDF file detection."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b'%PDF-1.4\n')

        file_type = FileTypeDetector.detect_file_type(pdf_file)
        assert file_type == 'pdf'

    def test_detect_by_extension(self, tmp_path):
        """Test detection by file extension."""
        docx_file = tmp_path / "test.docx"
        docx_file.write_bytes(b'dummy content')

        file_type = FileTypeDetector.detect_file_type(docx_file)
        assert file_type == 'docx'

    def test_is_supported(self, tmp_path):
        """Test supported format checking."""
        pdf_file = tmp_path / "test.pdf"
        pdf_file.write_bytes(b'%PDF-1.4\n')

        assert FileTypeDetector.is_supported(pdf_file, ['pdf', 'docx'])
        assert not FileTypeDetector.is_supported(pdf_file, ['docx', 'html'])

    def test_is_supported_uppercase_extension(self, tmp_path):
        """Ensure uppercase extensions are recognised."""
        pdf_file = tmp_path / "REPORT.PDF"
        pdf_file.write_bytes(b'%PDF-1.4\n')

        assert FileTypeDetector.is_supported(pdf_file, ['pdf'])


class TestTextCleaner:
    """Test text cleaning utilities."""

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        text = "Hello\r\nWorld\r\n\r\n\r\nTest"
        cleaned = TextCleaner.clean_text(text)

        assert '\r' not in cleaned
        assert cleaned.count('\n\n') <= 1

    def test_clean_text_aggressive(self):
        """Test aggressive cleaning."""
        text = "Hello  World    with   spaces"
        cleaned = TextCleaner.clean_text(text, aggressive=True)

        assert '  ' not in cleaned

    def test_remove_urls(self):
        """Test URL removal."""
        text = "Check out https://example.com for more info"
        cleaned = TextCleaner.remove_urls(text)

        assert 'https://example.com' not in cleaned
        assert 'Check out' in cleaned

    def test_extract_numbers(self):
        """Test number extraction."""
        text = "The values are 42, 3.14, and -17.5"
        numbers = TextCleaner.extract_numbers(text)

        assert 42.0 in numbers
        assert 3.14 in numbers
        assert -17.5 in numbers


class TestTableExtractor:
    """Test table extraction utilities."""

    def test_table_to_markdown(self):
        """Test table to markdown conversion."""
        cells = [
            ['Name', 'Age', 'City'],
            ['Alice', '30', 'NYC'],
            ['Bob', '25', 'LA']
        ]

        markdown = TableExtractor.table_to_markdown(cells, has_header=True)

        assert '| Name | Age | City |' in markdown
        assert '|---|---|---|' in markdown
        assert '| Alice | 30 | NYC |' in markdown

    def test_table_to_html(self):
        """Test table to HTML conversion."""
        cells = [
            ['Header1', 'Header2'],
            ['Value1', 'Value2']
        ]

        html = TableExtractor.table_to_html(cells, has_header=True)

        assert '<table>' in html
        assert '<thead>' in html
        assert '<th>Header1</th>' in html
        assert '<td>Value1</td>' in html


class TestDocumentParser:
    """Test document parsing."""

    def test_parse_text_file(self, tmp_path):
        """Test parsing plain text file."""
        text_file = tmp_path / "test.txt"
        content = "# Title\n\nThis is a paragraph.\n\nThis is another paragraph."
        text_file.write_text(content, encoding='utf-8')

        parser = DocumentParser()
        result = parser.parse(text_file)

        assert isinstance(result, ParsedDocument)
        assert result.text
        assert len(result.elements) > 0
        assert result.metadata['parser'] == 'text'

    def test_parse_html_file(self, tmp_path):
        """Test parsing HTML file."""
        html_file = tmp_path / "test.html"
        content = """
        <html>
        <head><title>Test Document</title></head>
        <body>
            <h1>Main Title</h1>
            <p>This is a paragraph.</p>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
        </body>
        </html>
        """
        html_file.write_text(content, encoding='utf-8')

        parser = DocumentParser()
        result = parser.parse(html_file)

        assert isinstance(result, ParsedDocument)
        assert 'Main Title' in result.text
        assert result.metadata['parser'] == 'html'

        # Check for different element types
        element_types = [e.type for e in result.elements]
        assert 'heading' in element_types
        assert 'paragraph' in element_types
        assert 'list' in element_types

    def test_parse_markdown_file(self, tmp_path):
        """Test parsing markdown file."""
        md_file = tmp_path / "test.md"
        content = """# Main Heading

This is a paragraph with some **bold** text.

## Subsection

- List item 1
- List item 2

Another paragraph here.
"""
        md_file.write_text(content, encoding='utf-8')

        parser = DocumentParser()
        result = parser.parse(md_file)

        assert isinstance(result, ParsedDocument)
        assert 'Main Heading' in result.text
        assert len(result.elements) > 0


class TestDocTagsProcessor:
    """Test DocTags processing."""

    def create_sample_parsed_doc(self) -> ParsedDocument:
        """Create a sample parsed document for testing."""
        elements = [
            DocumentElement(type='heading', content='Introduction', level=1),
            DocumentElement(type='paragraph', content='This is the introduction paragraph.'),
            DocumentElement(type='heading', content='Methods', level=1),
            DocumentElement(type='paragraph', content='Description of methods.'),
            DocumentElement(type='list', content='- Step 1\n- Step 2\n- Step 3'),
            DocumentElement(type='table', content='| Col1 | Col2 |\n|---|---|\n| A | B |'),
        ]

        return ParsedDocument(
            text='Introduction\nThis is the introduction...',
            elements=elements,
            metadata={'filename': 'test.txt', 'parser': 'text'},
            structure={'total_elements': len(elements)}
        )

    def test_process_document(self):
        """Test DocTags processing."""
        parsed_doc = self.create_sample_parsed_doc()
        processor = DocTagsProcessor()

        result = processor.process(parsed_doc)

        assert isinstance(result, DocTagsDocument)
        assert result.doc_id
        assert result.title
        assert len(result.tags) > 0
        assert result.hierarchy

    def test_doctags_structure(self):
        """Test DocTags structure preservation."""
        parsed_doc = self.create_sample_parsed_doc()
        processor = DocTagsProcessor()

        result = processor.process(parsed_doc)

        # Check for different tag types
        tag_types = [tag.tag_type for tag in result.tags]
        assert DocTagType.DOCUMENT in tag_types or DocTagType.TITLE in tag_types
        assert DocTagType.PARAGRAPH in tag_types

        # Check parent-child relationships
        for tag in result.tags:
            if tag.parent_id:
                parent = next((t for t in result.tags if t.tag_id == tag.parent_id), None)
                assert parent is not None

    def test_doctags_to_markdown(self):
        """Test DocTags to Markdown conversion."""
        parsed_doc = self.create_sample_parsed_doc()
        processor = DocTagsProcessor()

        doctags_doc = processor.process(parsed_doc)
        markdown = DocTagsConverter.to_markdown(doctags_doc)

        assert isinstance(markdown, str)
        assert '#' in markdown  # Should have headings
        assert doctags_doc.title in markdown

    def test_doctags_to_html(self):
        """Test DocTags to HTML conversion."""
        parsed_doc = self.create_sample_parsed_doc()
        processor = DocTagsProcessor()

        doctags_doc = processor.process(parsed_doc)
        html = DocTagsConverter.to_html(doctags_doc)

        assert isinstance(html, str)
        assert '<html>' in html
        assert '<body>' in html
        assert doctags_doc.title in html

    def test_doctags_to_json(self):
        """Test DocTags to JSON conversion."""
        parsed_doc = self.create_sample_parsed_doc()
        processor = DocTagsProcessor()

        doctags_doc = processor.process(parsed_doc)
        json_str = doctags_doc.to_json()

        assert isinstance(json_str, str)

        # Parse JSON to verify structure
        data = json.loads(json_str)
        assert 'doc_id' in data
        assert 'tags' in data
        assert isinstance(data['tags'], list)


class TestChunker:
    """Test chunking functionality."""

    def create_sample_doctags_doc(self) -> DocTagsDocument:
        """Create a sample DocTags document for testing."""
        from src.processing.doctags_processor import DocTag

        tags = [
            DocTag(
                tag_type=DocTagType.DOCUMENT,
                content='Test Document',
                tag_id='tag_000000',
                order=0
            ),
            DocTag(
                tag_type=DocTagType.SECTION,
                content='Introduction',
                tag_id='tag_000001',
                parent_id='tag_000000',
                level=1,
                order=1
            ),
            DocTag(
                tag_type=DocTagType.PARAGRAPH,
                content='This is a paragraph with some content. ' * 20,
                tag_id='tag_000002',
                parent_id='tag_000001',
                order=2
            ),
            DocTag(
                tag_type=DocTagType.PARAGRAPH,
                content='Another paragraph here. ' * 15,
                tag_id='tag_000003',
                parent_id='tag_000001',
                order=3
            ),
        ]

        return DocTagsDocument(
            doc_id='test_doc',
            title='Test Document',
            tags=tags,
            metadata={},
            hierarchy={}
        )

    def test_structure_preserving_chunker(self):
        """Test structure-preserving chunking."""
        doc = self.create_sample_doctags_doc()
        chunker = StructurePreservingChunker(
            chunk_size=500,
            chunk_overlap=100
        )

        chunks = chunker.chunk_document(doc)

        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_size_limit(self):
        """Test that chunks respect size limits."""
        doc = self.create_sample_doctags_doc()
        chunk_size = 300
        chunker = StructurePreservingChunker(chunk_size=chunk_size)

        chunks = chunker.chunk_document(doc)

        # Allow flexibility for structure preservation
        # Individual paragraphs/elements may exceed chunk_size to maintain coherence
        for chunk in chunks:
            assert len(chunk.content) <= chunk_size * 3  # Allow flexibility for structure preservation

    def test_chunk_context(self):
        """Test that chunks include context."""
        doc = self.create_sample_doctags_doc()
        chunker = StructurePreservingChunker(include_context=True)

        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 0

        # Check that context is included
        for chunk in chunks:
            assert 'context' in chunk.__dict__
            assert isinstance(chunk.context, dict)

    def test_chunk_metadata(self):
        """Test chunk metadata."""
        doc = self.create_sample_doctags_doc()
        chunker = StructurePreservingChunker()

        chunks = chunker.chunk_document(doc)

        for chunk in chunks:
            assert chunk.chunk_id
            assert chunk.doc_id == doc.doc_id
            assert isinstance(chunk.chunk_index, int)
            assert chunk.char_start >= 0
            assert chunk.char_end >= chunk.char_start


class TestProcessingPipeline:
    """Test end-to-end processing pipeline."""

    def test_create_pipeline(self):
        """Test pipeline creation."""
        pipeline = create_pipeline(
            chunk_size=1000,
            chunk_overlap=200
        )

        assert isinstance(pipeline, DocumentProcessingPipeline)
        assert pipeline.config.chunk_size == 1000
        assert pipeline.config.chunk_overlap == 200

    def test_process_text_file(self, tmp_path):
        """Test processing a text file."""
        # Create test file
        text_file = tmp_path / "test.txt"
        content = """# Introduction

This is a test document with multiple paragraphs.

## Section 1

This section contains information about the first topic.

## Section 2

This section contains information about the second topic.

- Point 1
- Point 2
- Point 3
"""
        text_file.write_text(content, encoding='utf-8')

        # Process file
        pipeline = create_pipeline(chunk_size=200, chunk_overlap=50)
        result = pipeline.process_file(text_file)

        # Verify result
        assert result.success
        assert result.stage == ProcessingStage.COMPLETED
        assert result.parsed_doc is not None
        assert result.doctags_doc is not None
        assert result.chunks is not None
        assert len(result.chunks) > 0

    def test_semantic_chunking_without_model_falls_back(self, tmp_path):
        """Semantic chunking should fall back to structure when no model configured."""
        text_file = tmp_path / "semantic.txt"
        text_file.write_text("Paragraph one. Paragraph two.", encoding='utf-8')

        config = PipelineConfig(
            chunk_size=100,
            chunk_overlap=20,
            chunking_method='semantic',
            semantic_model=None,
            enable_ocr=False,
        )
        pipeline = DocumentProcessingPipeline(config)
        result = pipeline.process_file(text_file)

        assert result.success
        metadata = result.metadata
        assert metadata.get('chunking_method') == 'structure'
        assert metadata.get('semantic_chunking_error')

    def test_find_supported_files_handles_uppercase_extensions(self, tmp_path):
        """Uppercase extensions should be detected as supported."""
        doc_dir = tmp_path
        upper_file = doc_dir / "WHITEPAPER.PDF"
        upper_file.write_bytes(b'%PDF-1.4\n')

        pipeline = DocumentProcessingPipeline()
        files = pipeline._find_supported_files(doc_dir, recursive=False)

        assert upper_file in files

    def test_process_html_file(self, tmp_path):
        """Test processing an HTML file."""
        html_file = tmp_path / "test.html"
        content = """
        <html>
        <head><title>Test</title></head>
        <body>
            <h1>Main Title</h1>
            <p>Paragraph 1</p>
            <h2>Subsection</h2>
            <p>Paragraph 2</p>
        </body>
        </html>
        """
        html_file.write_text(content, encoding='utf-8')

        pipeline = create_pipeline()
        result = pipeline.process_file(html_file)

        assert result.success
        assert len(result.chunks) > 0

    def test_pipeline_with_invalid_file(self, tmp_path):
        """Test pipeline with invalid file."""
        invalid_file = tmp_path / "nonexistent.txt"

        pipeline = create_pipeline()
        result = pipeline.process_file(invalid_file)

        assert not result.success
        assert result.error is not None

    def test_batch_processing(self, tmp_path):
        """Test batch processing."""
        # Create multiple test files
        files = []
        for i in range(3):
            file = tmp_path / f"test_{i}.txt"
            file.write_text(f"Document {i}\n\nContent for document {i}.", encoding='utf-8')
            files.append(file)

        pipeline = create_pipeline()
        results = pipeline.process_batch(files)

        assert len(results) == 3
        assert all(r.success for r in results)

    def test_pipeline_statistics(self, tmp_path):
        """Test pipeline statistics."""
        # Create test files
        files = []
        for i in range(3):
            file = tmp_path / f"test_{i}.txt"
            file.write_text(f"Document {i}\n\nContent here.", encoding='utf-8')
            files.append(file)

        pipeline = create_pipeline()
        results = pipeline.process_batch(files)
        stats = pipeline.get_statistics(results)

        assert 'total' in stats
        assert 'successful' in stats
        assert 'failed' in stats
        assert stats['total'] == 3

    def test_save_intermediate_results(self, tmp_path):
        """Test saving intermediate results."""
        text_file = tmp_path / "test.txt"
        content = "# Test\n\nThis is test content."
        text_file.write_text(content, encoding='utf-8')

        output_dir = tmp_path / "output"

        config = PipelineConfig(
            save_intermediate=True,
            save_json=True,
            save_markdown=True,
            output_dir=output_dir
        )

        pipeline = DocumentProcessingPipeline(config)
        result = pipeline.process_file(text_file)

        assert result.success

        # Check that output files were created
        doc_output_dir = output_dir / "test"
        assert doc_output_dir.exists()


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_full_workflow(self, tmp_path):
        """Test complete document processing workflow."""
        # Create a test document
        doc_file = tmp_path / "document.txt"
        content = """# Research Paper

## Abstract

This is the abstract of the research paper.

## Introduction

The introduction provides background information.

## Methods

### Data Collection

We collected data using the following methods:
- Method 1
- Method 2
- Method 3

### Analysis

The analysis was performed using statistical methods.

## Results

The results show significant findings.

## Conclusion

In conclusion, this research demonstrates important insights.
"""
        doc_file.write_text(content, encoding='utf-8')

        # Process document
        pipeline = create_pipeline(
            chunk_size=300,
            chunk_overlap=50
        )

        result = pipeline.process_file(doc_file)

        # Verify complete workflow
        assert result.success

        # Check parsed document
        assert result.parsed_doc is not None
        assert len(result.parsed_doc.elements) > 0
        assert 'Research Paper' in result.parsed_doc.text

        # Check DocTags
        assert result.doctags_doc is not None
        assert len(result.doctags_doc.tags) > 0
        tag_types = [t.tag_type for t in result.doctags_doc.tags]
        assert any(t in [DocTagType.TITLE, DocTagType.SECTION] for t in tag_types)

        # Check chunks
        assert result.chunks is not None
        assert len(result.chunks) >= 2  # Should have multiple chunks

        # Verify chunks have context
        for chunk in result.chunks:
            assert chunk.chunk_id
            assert chunk.doc_id
            assert chunk.content

        # Verify metadata
        assert 'num_chunks' in result.metadata
        assert result.metadata['num_chunks'] == len(result.chunks)


class TestDocTagsProcessorLegalDomain:
    """Test domain detection and legal-specific DocTag assignment."""

    def _make_legal_parsed_doc(self) -> ParsedDocument:
        """Parsed document containing three GDPR-style legal patterns."""
        elements = [
            DocumentElement(type='heading', content='Article 6', level=2),
            DocumentElement(type='paragraph', content='Processing shall be lawful only if at least one condition is met.'),
        ]
        # Three patterns: "Article 6", "Schedule 1", "Regulation (EU)"
        text = "Article 6 Schedule 1 Regulation (EU) 2016/679 provisions."
        return ParsedDocument(
            text=text,
            elements=elements,
            metadata={'filename': 'gdpr.txt', 'parser': 'text'},
            structure={},
        )

    def _make_general_parsed_doc(self) -> ParsedDocument:
        """Parsed document with non-legal technical content."""
        elements = [
            DocumentElement(type='heading', content='Introduction', level=1),
            DocumentElement(type='paragraph', content='This document describes a software API.'),
        ]
        return ParsedDocument(
            text='This document describes a software API with endpoints and parameters.',
            elements=elements,
            metadata={'filename': 'api_docs.txt', 'parser': 'text'},
            structure={},
        )

    def test_detect_general_for_non_legal_text(self):
        doc = self._make_general_parsed_doc()
        processor = DocTagsProcessor()
        assert processor._detect_document_domain(doc) == "general"

    def test_detect_legal_for_gdpr_like_text(self):
        doc = self._make_legal_parsed_doc()
        processor = DocTagsProcessor()
        assert processor._detect_document_domain(doc) == "legal"

    def test_detect_legal_requires_three_matches(self):
        """Only two legal patterns present — should remain general."""
        elements = [
            DocumentElement(type='paragraph', content='Article 6 and Schedule 1 apply.'),
        ]
        doc = ParsedDocument(
            text='Article 6 and Schedule 1 apply.',
            elements=elements,
            metadata={},
            structure={},
        )
        processor = DocTagsProcessor()
        assert processor._detect_document_domain(doc) == "general"

    def test_article_heading_produces_article_tag(self):
        processor = DocTagsProcessor()
        processor._document_domain = "legal"
        elem = DocumentElement(type='heading', content='Article 6', level=2)
        tag = processor._heading_to_doctag(elem, 'parent_id', None)
        assert tag.tag_type == DocTagType.ARTICLE

    def test_schedule_heading_produces_schedule_tag(self):
        processor = DocTagsProcessor()
        processor._document_domain = "legal"
        elem = DocumentElement(type='heading', content='Schedule 1', level=2)
        tag = processor._heading_to_doctag(elem, 'parent_id', None)
        assert tag.tag_type == DocTagType.SCHEDULE

    def test_definitions_heading_produces_definition_tag(self):
        processor = DocTagsProcessor()
        processor._document_domain = "legal"
        elem = DocumentElement(type='heading', content='Definitions', level=2)
        tag = processor._heading_to_doctag(elem, 'parent_id', None)
        assert tag.tag_type == DocTagType.DEFINITION

    def test_non_legal_doc_article_heading_stays_section(self):
        """In a general document, level-2 heading maps to SECTION regardless of content."""
        processor = DocTagsProcessor()
        processor._document_domain = "general"
        elem = DocumentElement(type='heading', content='Article 6', level=2)
        tag = processor._heading_to_doctag(elem, 'parent_id', None)
        assert tag.tag_type == DocTagType.SECTION

    def test_quoted_paragraph_is_definition(self):
        processor = DocTagsProcessor()
        processor._document_domain = "legal"
        elem = DocumentElement(type='paragraph', content='"Agreement" means a binding contract.')
        tag = processor._paragraph_to_doctag(elem, 'parent_id', None)
        assert tag.tag_type == DocTagType.DEFINITION

    def test_except_where_paragraph_is_exception(self):
        processor = DocTagsProcessor()
        processor._document_domain = "legal"
        elem = DocumentElement(
            type='paragraph',
            content='The obligation shall not apply except where necessary for public interest.',
        )
        tag = processor._paragraph_to_doctag(elem, 'parent_id', None)
        assert tag.tag_type == DocTagType.EXCEPTION

    def test_subject_to_article_paragraph_is_cross_reference(self):
        processor = DocTagsProcessor()
        processor._document_domain = "legal"
        elem = DocumentElement(
            type='paragraph',
            content='Processing is permitted subject to Article 17 of this regulation.',
        )
        tag = processor._paragraph_to_doctag(elem, 'parent_id', None)
        assert tag.tag_type == DocTagType.CROSS_REFERENCE

    def test_plain_paragraph_unchanged_in_legal_domain(self):
        processor = DocTagsProcessor()
        processor._document_domain = "legal"
        elem = DocumentElement(
            type='paragraph',
            content='The data subject has certain rights under this regulation.',
        )
        tag = processor._paragraph_to_doctag(elem, 'parent_id', None)
        assert tag.tag_type == DocTagType.PARAGRAPH


class TestChunkerLegalBoundaries:
    """Test that ARTICLE and SCHEDULE tags trigger chunk flush boundaries."""

    def _make_tag(self, tag_type, content, tag_id, parent_id='tag_000000', level=None, order=0):
        from src.processing.doctags_processor import DocTag
        return DocTag(
            tag_type=tag_type,
            content=content,
            tag_id=tag_id,
            parent_id=parent_id,
            level=level,
            order=order,
        )

    def _make_doc(self, tags) -> DocTagsDocument:
        return DocTagsDocument(
            doc_id='test_legal_doc',
            title='Legal Test Document',
            tags=tags,
            metadata={},
            hierarchy={},
        )

    def test_article_tag_triggers_chunk_flush(self):
        tags = [
            self._make_tag(DocTagType.DOCUMENT, 'Doc', 'tag_000000', None, order=0),
            self._make_tag(DocTagType.PARAGRAPH, 'Before Article content.', 'tag_000001', order=1),
            self._make_tag(DocTagType.ARTICLE, 'Article 6', 'tag_000002', level=2, order=2),
            self._make_tag(DocTagType.PARAGRAPH, 'After Article content.', 'tag_000003', order=3),
        ]
        doc = self._make_doc(tags)
        chunker = StructurePreservingChunker(chunk_size=2000)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 2
        assert 'Before Article' in chunks[0].content
        assert 'After Article' in chunks[1].content

    def test_schedule_tag_triggers_chunk_flush(self):
        tags = [
            self._make_tag(DocTagType.DOCUMENT, 'Doc', 'tag_000000', None, order=0),
            self._make_tag(DocTagType.PARAGRAPH, 'Before Schedule content.', 'tag_000001', order=1),
            self._make_tag(DocTagType.SCHEDULE, 'Schedule 1', 'tag_000002', level=2, order=2),
            self._make_tag(DocTagType.PARAGRAPH, 'After Schedule content.', 'tag_000003', order=3),
        ]
        doc = self._make_doc(tags)
        chunker = StructurePreservingChunker(chunk_size=2000)
        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 2
        assert 'Before Schedule' in chunks[0].content
        assert 'After Schedule' in chunks[1].content

    def test_article_recorded_in_context_hierarchy(self):
        article_tag_id = 'tag_art_001'
        tags = [
            self._make_tag(DocTagType.DOCUMENT, 'Legal Doc', 'tag_000000', None, order=0),
            self._make_tag(DocTagType.ARTICLE, 'Article 6', article_tag_id, level=2, order=1),
            self._make_tag(DocTagType.PARAGRAPH, 'Consent is required for lawful processing.', 'tag_000002', order=2),
        ]
        doc = self._make_doc(tags)
        chunker = StructurePreservingChunker(chunk_size=2000)

        # Verify internal hierarchy captures article tag_id
        hierarchy = chunker._build_context_hierarchy(doc)
        assert article_tag_id in hierarchy['sections']

        # Verify the paragraph chunk records the article as its section
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1
        assert chunks[0].context.get('section') is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
