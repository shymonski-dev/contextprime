"""
Mapper to convert Crawl4AI Markdown to DocTags format.

Transforms raw markdown from the crawler into Contextprime's standard
DocTagsDocument structure, enabling seamless integration with the
existing chunking and indexing pipelines.
"""

import re
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

from ..doctags_processor import DocTag, DocTagType, DocTagsDocument
from .crawler import WebCrawlResult


class WebDocTagsMapper:
    """
    Converts raw Markdown content into a structured DocTagsDocument.
    """

    def __init__(self):
        self.tag_counter = 0

    def map_to_doctags(self, crawl_result: WebCrawlResult) -> DocTagsDocument:
        """
        Main entry point: Convert WebCrawlResult to DocTagsDocument.
        """
        self.tag_counter = 0
        
        doc_id = self._generate_id(crawl_result.url)
        timestamp = datetime.now().isoformat()
        
        tags: List[DocTag] = []
        
        # 1. Root Document Tag
        root_tag = self._create_tag(
            DocTagType.DOCUMENT,
            crawl_result.title,
            level=0
        )
        tags.append(root_tag)
        
        # 2. Parse Markdown Line-by-Line
        if crawl_result.markdown:
            lines = crawl_result.markdown.split('\n')
            current_section_id = root_tag.tag_id
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                tag = self._parse_line(line)
                if tag:
                    # Update hierarchy tracking if it's a section
                    if tag.tag_type in [DocTagType.SECTION, DocTagType.SUBSECTION]:
                        current_section_id = tag.tag_id
                        # Sections parent to root or previous major section could be refined
                        # For simple linear web pages, parenting to root/current_section is okay
                        tag.parent_id = root_tag.tag_id if tag.level == 1 else current_section_id
                        # If it's a new major section (level 1/2), update current
                        current_section_id = tag.tag_id
                    else:
                        tag.parent_id = current_section_id
                        
                    tags.append(tag)

        # 3. Construct Hierarchy (Simplified)
        hierarchy = {'root': [root_tag.tag_id], 'sections': {}}
        for tag in tags:
            if tag.parent_id:
                # Find parent and add to children
                # In a real implementation this would be more optimized
                pass 

        return DocTagsDocument(
            doc_id=doc_id,
            title=crawl_result.title,
            tags=tags,
            metadata={
                "source_type": "web",
                "source_url": crawl_result.url,
                "crawled_at": crawl_result.crawled_at,
                "http_status": 200 if crawl_result.success else 500,
                "domain": self._extract_domain(crawl_result.url),
                "total_links": len(crawl_result.links)
            },
            hierarchy=hierarchy # Placeholder for now, DocTagsProcessor usually builds this
        )

    def _parse_line(self, line: str) -> Optional[DocTag]:
        """Identify tag type from markdown syntax."""
        
        # Headers
        if line.startswith('# '):
            return self._create_tag(DocTagType.TITLE, line[2:], level=1)
        if line.startswith('## '):
            return self._create_tag(DocTagType.SECTION, line[3:], level=2)
        if line.startswith('### '):
            return self._create_tag(DocTagType.SUBSECTION, line[4:], level=3)
            
        # Lists
        if line.startswith('- ') or line.startswith('* '):
            return self._create_tag(DocTagType.LIST, line[2:])
            
        # Code Blocks
        if line.startswith('```'):
            return None 
            
        # Legal Detection (Heuristic)
        if re.match(r'^Article\s+\d+', line, re.IGNORECASE):
            return self._create_tag(DocTagType.ARTICLE, line)
            
        # Default to Paragraph
        return self._create_tag(DocTagType.PARAGRAPH, line)

    def _create_tag(self, tag_type: DocTagType, content: str, level: Optional[int] = None) -> DocTag:
        tag_id = f"tag_{self.tag_counter:06d}"
        self.tag_counter += 1
        
        return DocTag(
            tag_id=tag_id,
            tag_type=tag_type,
            content=content,
            level=level,
            order=self.tag_counter,
            children_ids=[],
            metadata={}
        )

    def _generate_id(self, url: str) -> str:
        """Hash URL to create stable ID."""
        hash_obj = hashlib.sha256(url.encode())
        return f"web_{hash_obj.hexdigest()[:16]}"

    def _extract_domain(self, url: str) -> str:
        from urllib.parse import urlparse
        return urlparse(url).netloc
