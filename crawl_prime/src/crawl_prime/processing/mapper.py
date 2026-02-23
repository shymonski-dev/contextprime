"""
Mapper to convert Crawl4AI Markdown to DocTags format.
"""

import re
import hashlib
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

# We define a lightweight version of the DocTag structures here
# to avoid hard dependency on the main repository.

class DocTagType:
    DOCUMENT = "document"
    TITLE = "title"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"
    LIST = "list"
    CODE = "code"
    LINK = "link"
    ARTICLE = "article"  # Legal
    SCHEDULE = "schedule" # Legal


class DocTagsMapper:
    """
    Converts raw Markdown content into a structured DocTags document dictionary.
    """

    def __init__(self):
        self.tag_counter = 0

    def map_to_doctags(self, crawl_result: Any) -> Dict[str, Any]:
        """
        Main entry point: Convert CrawlResult to DocTags JSON structure.
        """
        self.tag_counter = 0
        
        doc_id = self._generate_id(crawl_result.url)
        timestamp = datetime.now().isoformat()
        
        tags = []
        
        # 1. Root Document Tag
        tags.append(self._create_tag(
            DocTagType.DOCUMENT,
            crawl_result.title,
            level=0
        ))
        
        # 2. Parse Markdown Line-by-Line (Simple Parser)
        # In a full implementation, we might use a Markdown AST parser like 'mistune'
        # For now, we use a robust line-scanner which is sufficient for headers.
        
        lines = crawl_result.markdown.split('\n')
        current_section_id = tags[0]['tag_id']
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            tag = self._parse_line(line)
            if tag:
                # Update hierarchy tracking if it's a section
                if tag['tag_type'] in [DocTagType.SECTION, DocTagType.SUBSECTION]:
                    current_section_id = tag['tag_id']
                else:
                    tag['parent_id'] = current_section_id
                    
                tags.append(tag)

        # 3. Construct Final Object
        return {
            "doc_id": doc_id,
            "title": crawl_result.title,
            "source_type": "web",
            "source_url": crawl_result.url,
            "created_at": timestamp,
            "tags": tags,
            "metadata": {
                "crawled_at": crawl_result.crawled_at,
                "http_status": 200 if crawl_result.success else 500,
                "domain": self._extract_domain(crawl_result.url),
                "total_links": len(crawl_result.links)
            }
        }

    def _parse_line(self, line: str) -> Optional[Dict[str, Any]]:
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
            
        # Code Blocks (Simplified - usually multi-line, handling single lines here)
        if line.startswith('```'):
            return None # Skip fences
            
        # Legal Detection (Heuristic)
        if re.match(r'^Article\s+\d+', line, re.IGNORECASE):
            return self._create_tag(DocTagType.ARTICLE, line)
            
        # Default to Paragraph
        return self._create_tag(DocTagType.PARAGRAPH, line)

    def _create_tag(self, tag_type: str, content: str, level: Optional[int] = None) -> Dict[str, Any]:
        tag_id = f"tag_{self.tag_counter:06d}"
        self.tag_counter += 1
        
        return {
            "tag_id": tag_id,
            "tag_type": tag_type,
            "content": content,
            "level": level,
            "order": self.tag_counter,
            "children_ids": [],
            "metadata": {}
        }

    def _generate_id(self, url: str) -> str:
        """Hash URL to create stable ID."""
        hash_obj = hashlib.sha256(url.encode())
        return f"web_{hash_obj.hexdigest()[:16]}"

    def _extract_domain(self, url: str) -> str:
        from urllib.parse import urlparse
        return urlparse(url).netloc
