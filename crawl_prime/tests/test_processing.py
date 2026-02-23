import pytest
from src.crawl_prime.processing.mapper import DocTagsMapper, DocTagType
from src.crawl_prime.crawler.engine import CrawlResult

class TestProcessing:
    def test_markdown_mapping(self):
        """Test converting markdown to doctags structure."""
        
        # Mock result from crawl4ai
        mock_result = CrawlResult(
            url="https://example.com",
            title="Test Page",
            markdown="# Main Title\n\n## Section 1\n\nSome text.\n\n### Subsection A\n\n- List item",
            html="<html>...</html>",
            crawled_at="2024-01-01",
            links=[],
            media=[],
            metadata={},
            success=True
        )
        
        mapper = DocTagsMapper()
        doctags = mapper.map_to_doctags(mock_result)
        
        assert doctags["title"] == "Test Page"
        assert len(doctags["tags"]) == 6 # Document + Title + Section + Text + Subsec + List
        
        # Check hierarchy (Section 1 is a section)
        section = doctags["tags"][2]
        assert section["tag_type"] == DocTagType.SECTION
        assert section["content"] == "Section 1"
        
        # Check list
        list_item = doctags["tags"][5]
        assert list_item["tag_type"] == DocTagType.LIST
