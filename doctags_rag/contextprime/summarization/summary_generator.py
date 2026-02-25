"""
LLM-based Abstractive Summary Generator for RAPTOR System.

Implements multi-level summarization with:
- Context-aware abstractive summaries
- Fact extraction and preservation
- Quality control and hallucination detection
- Different summary styles for different levels
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import re
import json
from loguru import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available")

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic not available")


class SummaryLevel(Enum):
    """Summary level in the tree hierarchy."""
    LEAF = "leaf"           # Chunk-level summaries
    INTERMEDIATE = "intermediate"  # Section-level summaries
    ROOT = "root"           # Document-level summaries


@dataclass
class Summary:
    """Represents a generated summary."""
    content: str
    level: SummaryLevel
    source_ids: List[str]  # IDs of source chunks/summaries
    metadata: Dict[str, Any] = field(default_factory=dict)
    key_facts: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    quality_score: float = 0.0

    def __len__(self) -> int:
        return len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'content': self.content,
            'level': self.level.value,
            'source_ids': self.source_ids,
            'metadata': self.metadata,
            'key_facts': self.key_facts,
            'entities': self.entities,
            'quality_score': self.quality_score
        }


class SummaryGenerator:
    """
    Generates multi-level abstractive summaries using LLMs.

    Supports different summary styles and lengths for different
    levels of the hierarchy.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3
    ):
        """
        Initialize summary generator.

        Args:
            provider: LLM provider (openai or anthropic)
            model: Model name
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API (defaults to OPENAI_BASE_URL env var, supports OpenRouter)
            temperature: Sampling temperature
            max_retries: Max retry attempts
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        # Get API configuration from environment if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if base_url is None:
            base_url = os.getenv("OPENAI_BASE_URL")

        # Initialize client
        if provider == "openai" and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        elif provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = Anthropic(api_key=api_key)
        else:
            self.client = None
            logger.warning(f"LLM client for {provider} not available")

        # Summary length targets by level
        self.length_targets = {
            SummaryLevel.LEAF: 200,
            SummaryLevel.INTERMEDIATE: 500,
            SummaryLevel.ROOT: 1000
        }

        logger.info(
            f"SummaryGenerator initialized: provider={provider}, model={model}"
        )

    def generate_summary(
        self,
        texts: List[str],
        level: SummaryLevel,
        source_ids: List[str],
        context: Optional[Dict[str, Any]] = None
    ) -> Summary:
        """
        Generate summary from multiple texts.

        Args:
            texts: List of texts to summarize
            level: Summary level
            source_ids: IDs of source items
            context: Additional context

        Returns:
            Generated summary
        """
        if not texts:
            logger.warning("No texts provided for summarization")
            return Summary(
                content="",
                level=level,
                source_ids=source_ids
            )

        logger.info(f"Generating {level.value} summary from {len(texts)} texts")

        # Prepare input
        combined_text = self._prepare_input(texts, level, context)

        # Generate summary
        summary_content = self._call_llm(combined_text, level, context)

        # Extract key facts and entities
        key_facts = self._extract_key_facts(summary_content)
        entities = self._extract_entities(summary_content)

        # Assess quality
        quality_score = self._assess_quality(
            summary_content,
            texts,
            key_facts
        )

        summary = Summary(
            content=summary_content,
            level=level,
            source_ids=source_ids,
            metadata={
                'num_sources': len(texts),
                'target_length': self.length_targets[level],
                'actual_length': len(summary_content)
            },
            key_facts=key_facts,
            entities=entities,
            quality_score=quality_score
        )

        logger.info(
            f"Summary generated: {len(summary_content)} chars, "
            f"quality={quality_score:.3f}"
        )

        return summary

    def generate_batch_summaries(
        self,
        text_groups: List[List[str]],
        level: SummaryLevel,
        source_id_groups: List[List[str]],
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[Summary]:
        """
        Generate summaries for multiple text groups in batch.

        Args:
            text_groups: List of text groups
            level: Summary level
            source_id_groups: List of source ID groups
            contexts: Optional contexts for each group

        Returns:
            List of summaries
        """
        summaries = []

        if contexts is None:
            contexts = [None] * len(text_groups)

        for texts, source_ids, context in zip(text_groups, source_id_groups, contexts):
            summary = self.generate_summary(texts, level, source_ids, context)
            summaries.append(summary)

        return summaries

    def _prepare_input(
        self,
        texts: List[str],
        level: SummaryLevel,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Prepare input text for summarization.

        Args:
            texts: List of texts
            level: Summary level
            context: Additional context

        Returns:
            Prepared input string
        """
        # Add context if available
        parts = []

        if context:
            if 'document_title' in context:
                parts.append(f"Document: {context['document_title']}")
            if 'section' in context:
                parts.append(f"Section: {context['section']}")
            parts.append("")

        # Add texts with separators
        for i, text in enumerate(texts):
            parts.append(f"[Text {i+1}]")
            parts.append(text.strip())
            parts.append("")

        return "\n".join(parts)

    def _call_llm(
        self,
        text: str,
        level: SummaryLevel,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Call LLM to generate summary.

        Args:
            text: Input text
            level: Summary level
            context: Additional context

        Returns:
            Generated summary
        """
        if not self.client:
            # Fallback to extractive summary
            logger.warning("No LLM client available, using extractive summary")
            return self._extractive_summary(text, level)

        # Create prompt
        prompt = self._create_prompt(text, level, context)

        # Call LLM with retries
        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self._get_system_prompt(level)},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self._get_max_tokens(level)
                    )
                    return response.choices[0].message.content.strip()

                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=self._get_max_tokens(level),
                        temperature=self.temperature,
                        system=self._get_system_prompt(level),
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    return response.content[0].text.strip()

            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error("Max retries reached, falling back to extractive summary")
                    return self._extractive_summary(text, level)

        return self._extractive_summary(text, level)

    def _get_system_prompt(self, level: SummaryLevel) -> str:
        """Get system prompt for LLM based on summary level."""
        base = "You are an expert at creating concise, accurate summaries. "

        if level == SummaryLevel.LEAF:
            return base + (
                "Create a brief summary of the provided text chunk, "
                "focusing on key information and facts. "
                "Keep it concise (around 200 words) but preserve important details."
            )
        elif level == SummaryLevel.INTERMEDIATE:
            return base + (
                "Create a comprehensive summary that synthesizes information "
                "from multiple related text passages. "
                "Identify common themes and key points. "
                "Target length: around 500 words."
            )
        else:  # ROOT
            return base + (
                "Create a high-level overview that captures the main themes, "
                "key findings, and important conclusions from the entire document. "
                "Focus on the big picture while preserving critical details. "
                "Target length: around 1000 words."
            )

    def _create_prompt(
        self,
        text: str,
        level: SummaryLevel,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Create prompt for summarization."""
        parts = []

        if level == SummaryLevel.LEAF:
            parts.append("Summarize the following text concisely:")
        elif level == SummaryLevel.INTERMEDIATE:
            parts.append(
                "Synthesize the following related passages into a coherent summary. "
                "Identify common themes and integrate the information:"
            )
        else:  # ROOT
            parts.append(
                "Create a comprehensive overview of the following content. "
                "Capture the main themes, key findings, and important conclusions:"
            )

        parts.append("")
        parts.append(text)
        parts.append("")
        parts.append("Summary:")

        return "\n".join(parts)

    def _get_max_tokens(self, level: SummaryLevel) -> int:
        """Get max tokens for LLM based on summary level."""
        return {
            SummaryLevel.LEAF: 300,
            SummaryLevel.INTERMEDIATE: 700,
            SummaryLevel.ROOT: 1500
        }[level]

    def _extractive_summary(self, text: str, level: SummaryLevel) -> str:
        """
        Create extractive summary as fallback.

        Args:
            text: Input text
            level: Summary level

        Returns:
            Extractive summary
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Target number of sentences
        target_sentences = {
            SummaryLevel.LEAF: 3,
            SummaryLevel.INTERMEDIATE: 7,
            SummaryLevel.ROOT: 12
        }[level]

        # Take first N sentences (simple extractive approach)
        selected = sentences[:min(target_sentences, len(sentences))]

        return ' '.join(selected)

    def _extract_key_facts(self, summary: str) -> List[str]:
        """
        Extract key facts from summary.

        Args:
            summary: Summary text

        Returns:
            List of key facts
        """
        # Simple extraction: sentences with numbers, dates, or specific patterns
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        facts = []

        for sentence in sentences:
            # Check for factual indicators
            if any([
                re.search(r'\d+', sentence),  # Contains numbers
                re.search(r'\d{4}', sentence),  # Contains year
                re.search(r'(increase|decrease|percent|million|billion)', sentence, re.IGNORECASE),
                re.search(r'(shows|demonstrates|indicates|reveals)', sentence, re.IGNORECASE)
            ]):
                facts.append(sentence.strip())

        return facts[:5]  # Limit to top 5

    def _extract_entities(self, summary: str) -> List[str]:
        """
        Extract named entities from summary.

        Args:
            summary: Summary text

        Returns:
            List of entities
        """
        # Simple extraction: capitalized words/phrases
        # In production, would use spaCy or similar
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', summary)

        # Filter out common words
        stop_words = {'The', 'This', 'These', 'Those', 'However', 'Therefore', 'Additionally'}
        entities = [w for w in words if w not in stop_words]

        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity.lower() not in seen:
                seen.add(entity.lower())
                unique_entities.append(entity)

        return unique_entities[:10]  # Limit to top 10

    def _assess_quality(
        self,
        summary: str,
        source_texts: List[str],
        key_facts: List[str]
    ) -> float:
        """
        Assess summary quality.

        Args:
            summary: Generated summary
            source_texts: Source texts
            key_facts: Extracted key facts

        Returns:
            Quality score (0-1)
        """
        score = 0.0
        factors = 0

        # Length appropriateness (0-1)
        length_score = min(1.0, len(summary) / 100)
        score += length_score
        factors += 1

        # Fact density (0-1)
        fact_score = min(1.0, len(key_facts) / 5)
        score += fact_score
        factors += 1

        # Information coverage (0-1)
        # Check if summary contains terms from source texts
        source_words = set()
        for text in source_texts:
            words = re.findall(r'\b\w+\b', text.lower())
            source_words.update(words)

        summary_words = set(re.findall(r'\b\w+\b', summary.lower()))
        overlap = len(summary_words & source_words)
        coverage_score = min(1.0, overlap / max(len(source_words), 1) * 10)
        score += coverage_score
        factors += 1

        # Coherence (basic check)
        # Summary should have multiple sentences
        sentences = re.split(r'[.!?]', summary)
        coherence_score = min(1.0, len([s for s in sentences if s.strip()]) / 3)
        score += coherence_score
        factors += 1

        return score / factors if factors > 0 else 0.0

    def refine_summary(
        self,
        summary: Summary,
        feedback: str
    ) -> Summary:
        """
        Refine an existing summary based on feedback.

        Args:
            summary: Original summary
            feedback: Refinement feedback

        Returns:
            Refined summary
        """
        if not self.client:
            logger.warning("No LLM client available, cannot refine summary")
            return summary

        prompt = f"""Original summary:
{summary.content}

Feedback: {feedback}

Please refine the summary based on the feedback:"""

        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(summary.level)},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self._get_max_tokens(summary.level)
                )
                refined_content = response.choices[0].message.content.strip()

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self._get_max_tokens(summary.level),
                    temperature=self.temperature,
                    system=self._get_system_prompt(summary.level),
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                refined_content = response.content[0].text.strip()
            else:
                return summary

            # Create refined summary
            refined = Summary(
                content=refined_content,
                level=summary.level,
                source_ids=summary.source_ids,
                metadata={
                    **summary.metadata,
                    'refined': True,
                    'original_length': len(summary.content),
                    'refined_length': len(refined_content)
                },
                key_facts=self._extract_key_facts(refined_content),
                entities=self._extract_entities(refined_content),
                quality_score=self._assess_quality(
                    refined_content,
                    [summary.content],
                    self._extract_key_facts(refined_content)
                )
            )

            logger.info(f"Summary refined: quality improved from {summary.quality_score:.3f} to {refined.quality_score:.3f}")

            return refined

        except Exception as e:
            logger.error(f"Failed to refine summary: {e}")
            return summary
