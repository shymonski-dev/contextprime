"""
Cross-reference extractor for legal and structured documents.

Detects explicit references between document provisions (Articles, Sections,
Schedules, Paragraphs) and returns structured CrossRef dataclasses for storage
as Neo4j REFERENCES edges.

Domain-agnostic: patterns also match technical documentation and academic papers.
"""

import re
from dataclasses import dataclass
from typing import List

# Ordered from most specific to least specific to avoid double-counting
LEGAL_XREF_PATTERNS = [
    # "Article 17(3)(a)", "Art. 6(1)", "article 4"
    (r'\b(?:see\s+)?[Aa]rticle\s+(\d+(?:\(\d+\))*(?:[a-z])?)', "article"),
    # "pursuant to Article 6", "under Article 4", "subject to Article 17"
    (r'\b(?:pursuant\s+to|under|subject\s+to)\s+[Aa]rticle\s+(\d+)', "article"),
    # "as defined in Article 4"
    (r'\bas\s+defined\s+in\s+[Aa]rticle\s+(\d+)', "article"),
    # "Section 12.3", "section 4"
    (r'\b[Ss]ection\s+(\d+(?:\.\d+)*)', "section"),
    # "Schedule 1", "Schedule 2"
    (r'\b[Ss]chedule\s+(\d+)', "schedule"),
    # "paragraph 3(a)", "paragraph 1"
    (r'\bparagraph\s+(\d+(?:\(\w+\))*)', "paragraph"),
    # "Annex I", "Annex A"
    (r'\b[Aa]nnex\s+([IVXLCDM]+|[A-Z]|\d+)', "schedule"),
]


@dataclass
class CrossRef:
    """Represents a cross-reference from one chunk to a provision label."""
    source_chunk_id: str
    target_label: str   # e.g. "article_17", "schedule_1"
    ref_type: str       # "article", "section", "schedule", "paragraph"
    doc_id: str


def extract_cross_references(
    chunk_id: str,
    content: str,
    doc_id: str,
) -> List[CrossRef]:
    """Extract cross-references from chunk content.

    Args:
        chunk_id: ID of the source chunk.
        content:  Text content of the chunk.
        doc_id:   Document ID the chunk belongs to.

    Returns:
        List of CrossRef dataclasses, one per unique (ref_type, target) pair.
    """
    seen: set = set()
    refs: List[CrossRef] = []

    for pattern, ref_type in LEGAL_XREF_PATTERNS:
        for match in re.finditer(pattern, content):
            target_number = match.group(1).strip()
            target_label = f"{ref_type}_{target_number}"

            key = (ref_type, target_label)
            if key in seen:
                continue
            seen.add(key)

            refs.append(CrossRef(
                source_chunk_id=chunk_id,
                target_label=target_label,
                ref_type=ref_type,
                doc_id=doc_id,
            ))

    return refs
