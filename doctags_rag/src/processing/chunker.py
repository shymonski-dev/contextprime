"""
Structure-Preserving Document Chunker.

Implements intelligent chunking that:
- Respects document structure (sections, paragraphs, tables)
- Preserves hierarchical context
- Maintains semantic coherence
- Supports configurable chunk size and overlap
- Includes metadata for each chunk
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import re

from loguru import logger

from .doctags_processor import DocTagsDocument, DocTag, DocTagType


@dataclass
class Chunk:
    """Represents a chunk of document content with context."""
    chunk_id: str
    content: str
    doc_id: str
    chunk_index: int
    char_start: int
    char_end: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'doc_id': self.doc_id,
            'chunk_index': self.chunk_index,
            'char_start': self.char_start,
            'char_end': self.char_end,
            'metadata': self.metadata,
            'context': self.context,
        }


class StructurePreservingChunker:
    """
    Intelligent chunker that preserves document structure.

    Key features:
    - Context-aware chunking
    - Respects section boundaries
    - Preserves table integrity
    - Maintains hierarchical context
    - Smart splitting for long sections
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        respect_boundaries: bool = True,
        include_context: bool = True
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size (avoid tiny chunks)
            respect_boundaries: Respect section/paragraph boundaries
            include_context: Include hierarchical context in chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.respect_boundaries = respect_boundaries
        self.include_context = include_context

        logger.info(
            f"Chunker initialized: size={chunk_size}, "
            f"overlap={chunk_overlap}, boundaries={respect_boundaries}"
        )

    def chunk_document(self, doc: DocTagsDocument) -> List[Chunk]:
        """
        Chunk entire document preserving structure.

        Args:
            doc: DocTags document

        Returns:
            List of chunks with context
        """
        chunks = []
        chunk_index = 0

        # Build context hierarchy
        context_hierarchy = self._build_context_hierarchy(doc)

        # Process tags in reading order
        current_section = None
        current_subsection = None
        accumulated_content = []
        accumulated_tags = []

        for tag in doc.tags:
            # Skip document root tag
            if tag.tag_type == DocTagType.DOCUMENT:
                continue

            # Update section context
            if tag.tag_type == DocTagType.SECTION:
                # Flush accumulated content
                if accumulated_content:
                    new_chunks = self._create_chunks_from_content(
                        accumulated_content,
                        accumulated_tags,
                        doc.doc_id,
                        chunk_index,
                        current_section,
                        current_subsection,
                        context_hierarchy
                    )
                    chunks.extend(new_chunks)
                    chunk_index += len(new_chunks)
                    accumulated_content = []
                    accumulated_tags = []

                current_section = tag.content
                current_subsection = None

            elif tag.tag_type == DocTagType.SUBSECTION:
                # Flush accumulated content
                if accumulated_content:
                    new_chunks = self._create_chunks_from_content(
                        accumulated_content,
                        accumulated_tags,
                        doc.doc_id,
                        chunk_index,
                        current_section,
                        current_subsection,
                        context_hierarchy
                    )
                    chunks.extend(new_chunks)
                    chunk_index += len(new_chunks)
                    accumulated_content = []
                    accumulated_tags = []

                current_subsection = tag.content

            # Handle special tags that should be kept intact
            elif tag.tag_type in [DocTagType.TABLE, DocTagType.CODE, DocTagType.EQUATION]:
                # Flush accumulated content first
                if accumulated_content:
                    new_chunks = self._create_chunks_from_content(
                        accumulated_content,
                        accumulated_tags,
                        doc.doc_id,
                        chunk_index,
                        current_section,
                        current_subsection,
                        context_hierarchy
                    )
                    chunks.extend(new_chunks)
                    chunk_index += len(new_chunks)
                    accumulated_content = []
                    accumulated_tags = []

                # Create dedicated chunk for this element
                chunk = self._create_single_chunk(
                    tag,
                    doc.doc_id,
                    chunk_index,
                    current_section,
                    current_subsection,
                    context_hierarchy
                )
                chunks.append(chunk)
                chunk_index += 1

            else:
                # Accumulate regular content
                accumulated_content.append(tag.content)
                accumulated_tags.append(tag)

                # Check if we should flush
                total_length = sum(len(c) for c in accumulated_content)
                if total_length >= self.chunk_size * 2:  # Flush if accumulated too much
                    new_chunks = self._create_chunks_from_content(
                        accumulated_content,
                        accumulated_tags,
                        doc.doc_id,
                        chunk_index,
                        current_section,
                        current_subsection,
                        context_hierarchy
                    )
                    chunks.extend(new_chunks)
                    chunk_index += len(new_chunks)
                    accumulated_content = []
                    accumulated_tags = []

        # Flush remaining content
        if accumulated_content:
            new_chunks = self._create_chunks_from_content(
                accumulated_content,
                accumulated_tags,
                doc.doc_id,
                chunk_index,
                current_section,
                current_subsection,
                context_hierarchy
            )
            chunks.extend(new_chunks)

        logger.info(f"Created {len(chunks)} chunks from document {doc.doc_id}")

        return chunks

    def _build_context_hierarchy(self, doc: DocTagsDocument) -> Dict[str, Any]:
        """
        Build hierarchical context for the document.

        Args:
            doc: DocTags document

        Returns:
            Context hierarchy dictionary
        """
        hierarchy = {
            'title': doc.title,
            'sections': {},
            'tag_to_section': {}
        }

        current_section = None
        current_subsection = None

        for tag in doc.tags:
            if tag.tag_type == DocTagType.SECTION:
                current_section = tag.tag_id
                hierarchy['sections'][tag.tag_id] = {
                    'title': tag.content,
                    'level': tag.level,
                    'subsections': {}
                }
                current_subsection = None

            elif tag.tag_type == DocTagType.SUBSECTION:
                if current_section:
                    current_subsection = tag.tag_id
                    hierarchy['sections'][current_section]['subsections'][tag.tag_id] = {
                        'title': tag.content,
                        'level': tag.level
                    }

            # Map tag to its section context
            hierarchy['tag_to_section'][tag.tag_id] = {
                'section': current_section,
                'subsection': current_subsection
            }

        return hierarchy

    def _create_chunks_from_content(
        self,
        content_list: List[str],
        tag_list: List[DocTag],
        doc_id: str,
        start_index: int,
        section: Optional[str],
        subsection: Optional[str],
        hierarchy: Dict[str, Any]
    ) -> List[Chunk]:
        """
        Create chunks from accumulated content.

        Args:
            content_list: List of content strings
            tag_list: List of corresponding tags
            doc_id: Document ID
            start_index: Starting chunk index
            section: Current section title
            subsection: Current subsection title
            hierarchy: Context hierarchy

        Returns:
            List of chunks
        """
        chunks = []

        # Combine content
        full_content = '\n\n'.join(content_list)

        if len(full_content) <= self.chunk_size:
            # Single chunk
            chunk = Chunk(
                chunk_id=f"{doc_id}_chunk_{start_index:04d}",
                content=full_content,
                doc_id=doc_id,
                chunk_index=start_index,
                char_start=0,
                char_end=len(full_content),
                metadata={
                    'num_tags': len(tag_list),
                    'tag_types': [tag.tag_type.value for tag in tag_list]
                },
                context=self._build_chunk_context(
                    section, subsection, hierarchy
                )
            )
            chunks.append(chunk)

        else:
            # Multiple chunks needed
            if self.respect_boundaries:
                # Chunk respecting paragraph boundaries
                chunks_content = self._split_respecting_boundaries(
                    content_list, tag_list
                )
            else:
                # Simple sliding window
                chunks_content = self._split_sliding_window(full_content)

            # Create chunk objects
            char_offset = 0
            for idx, chunk_content in enumerate(chunks_content):
                chunk = Chunk(
                    chunk_id=f"{doc_id}_chunk_{start_index + idx:04d}",
                    content=chunk_content,
                    doc_id=doc_id,
                    chunk_index=start_index + idx,
                    char_start=char_offset,
                    char_end=char_offset + len(chunk_content),
                    metadata={},
                    context=self._build_chunk_context(
                        section, subsection, hierarchy
                    )
                )
                chunks.append(chunk)
                char_offset += len(chunk_content) - self.chunk_overlap

        return chunks

    def _split_respecting_boundaries(
        self,
        content_list: List[str],
        tag_list: List[DocTag]
    ) -> List[str]:
        """
        Split content respecting element boundaries.

        Args:
            content_list: List of content strings
            tag_list: List of corresponding tags

        Returns:
            List of chunk contents
        """
        chunks = []
        current_chunk = []
        current_length = 0

        for content, tag in zip(content_list, tag_list):
            content_length = len(content)

            # Check if adding this would exceed chunk size
            if current_length + content_length > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                if self.chunk_overlap > 0 and len(current_chunk) > 0:
                    # Include last few items for overlap
                    overlap_content = []
                    overlap_length = 0
                    for item in reversed(current_chunk):
                        if overlap_length + len(item) <= self.chunk_overlap:
                            overlap_content.insert(0, item)
                            overlap_length += len(item)
                        else:
                            break

                    current_chunk = overlap_content
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0

            # Add content to current chunk
            current_chunk.append(content)
            current_length += content_length

        # Add remaining content
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(chunk_text)

        return chunks

    def _split_sliding_window(self, text: str) -> List[str]:
        """
        Split text using sliding window approach.

        Args:
            text: Text to split

        Returns:
            List of chunk contents
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                chunk_text = text[start:end]
                last_period = max(
                    chunk_text.rfind('. '),
                    chunk_text.rfind('.\n'),
                    chunk_text.rfind('! '),
                    chunk_text.rfind('? ')
                )

                if last_period > self.min_chunk_size:
                    end = start + last_period + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end

        return chunks

    def _create_single_chunk(
        self,
        tag: DocTag,
        doc_id: str,
        chunk_index: int,
        section: Optional[str],
        subsection: Optional[str],
        hierarchy: Dict[str, Any]
    ) -> Chunk:
        """
        Create a single chunk from a tag (for tables, code, etc.).

        Args:
            tag: DocTag to chunk
            doc_id: Document ID
            chunk_index: Chunk index
            section: Current section
            subsection: Current subsection
            hierarchy: Context hierarchy

        Returns:
            Chunk
        """
        return Chunk(
            chunk_id=f"{doc_id}_chunk_{chunk_index:04d}",
            content=tag.content,
            doc_id=doc_id,
            chunk_index=chunk_index,
            char_start=0,
            char_end=len(tag.content),
            metadata={
                'tag_type': tag.tag_type.value,
                'tag_id': tag.tag_id,
                'is_special': True,
                **tag.metadata
            },
            context=self._build_chunk_context(
                section, subsection, hierarchy
            )
        )

    def _build_chunk_context(
        self,
        section: Optional[str],
        subsection: Optional[str],
        hierarchy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build context information for a chunk.

        Args:
            section: Section title
            subsection: Subsection title
            hierarchy: Document hierarchy

        Returns:
            Context dictionary
        """
        context = {
            'document_title': hierarchy['title']
        }

        if section:
            context['section'] = section

        if subsection:
            context['subsection'] = subsection

        # Build breadcrumb path
        breadcrumbs = [hierarchy['title']]
        if section:
            breadcrumbs.append(section)
        if subsection:
            breadcrumbs.append(subsection)

        context['breadcrumbs'] = ' > '.join(breadcrumbs)

        return context


class SemanticChunker:
    """
    Advanced chunker using semantic similarity for chunk boundaries.

    Uses embeddings to find natural breaking points.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        similarity_threshold: float = 0.7,
        embeddings_model: Optional[Any] = None
    ):
        """
        Initialize semantic chunker.

        Args:
            chunk_size: Target chunk size
            similarity_threshold: Threshold for semantic similarity
            embeddings_model: Model for computing embeddings
        """
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        self.embeddings_model = embeddings_model

        logger.info(f"Semantic chunker initialized: size={chunk_size}")

    def chunk_document(self, doc: DocTagsDocument) -> List[Chunk]:
        """
        Chunk document using semantic boundaries.

        Args:
            doc: DocTags document

        Returns:
            List of semantically coherent chunks
        """
        # Fallback to structure-based chunking if no embeddings model
        if not self.embeddings_model:
            logger.warning("No embeddings model provided, using structure-based chunking")
            chunker = StructurePreservingChunker(chunk_size=self.chunk_size)
            return chunker.chunk_document(doc)

        # Extract sentences
        sentences = []
        for tag in doc.tags:
            if tag.tag_type in [DocTagType.PARAGRAPH, DocTagType.LIST]:
                # Split into sentences
                sent_list = self._split_sentences(tag.content)
                sentences.extend(sent_list)

        # Compute embeddings
        embeddings = self._compute_embeddings(sentences)

        # Find semantic boundaries
        boundaries = self._find_semantic_boundaries(
            sentences, embeddings
        )

        # Create chunks
        chunks = self._create_chunks_from_boundaries(
            sentences, boundaries, doc.doc_id
        )

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_embeddings(self, sentences: List[str]) -> List[List[float]]:
        """Compute sentence embeddings."""
        # Placeholder - would use actual embeddings model
        if self.embeddings_model:
            return self.embeddings_model.encode(sentences)
        return []

    def _find_semantic_boundaries(
        self,
        sentences: List[str],
        embeddings: List[List[float]]
    ) -> List[int]:
        """Find semantic boundaries based on embedding similarity."""
        boundaries = [0]

        for i in range(1, len(sentences)):
            # Compute similarity between consecutive sentences
            similarity = self._cosine_similarity(
                embeddings[i-1], embeddings[i]
            )

            # If similarity is below threshold, it's a boundary
            if similarity < self.similarity_threshold:
                boundaries.append(i)

        boundaries.append(len(sentences))

        return boundaries

    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Compute cosine similarity between two vectors."""
        import numpy as np

        v1 = np.array(vec1)
        v2 = np.array(vec2)

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _create_chunks_from_boundaries(
        self,
        sentences: List[str],
        boundaries: List[int],
        doc_id: str
    ) -> List[Chunk]:
        """Create chunks from semantic boundaries."""
        chunks = []

        for idx in range(len(boundaries) - 1):
            start = boundaries[idx]
            end = boundaries[idx + 1]

            chunk_sentences = sentences[start:end]
            content = ' '.join(chunk_sentences)

            chunk = Chunk(
                chunk_id=f"{doc_id}_chunk_{idx:04d}",
                content=content,
                doc_id=doc_id,
                chunk_index=idx,
                char_start=0,  # Would need to compute actual positions
                char_end=len(content),
                metadata={
                    'num_sentences': len(chunk_sentences),
                    'chunking_method': 'semantic'
                },
                context={}
            )
            chunks.append(chunk)

        return chunks
