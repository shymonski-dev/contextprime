"""
CRAG-Style Confidence Scoring System for DocTags RAG.

Implements comprehensive confidence assessment for retrieved chunks:
- Multi-signal retrieval quality scoring
- Corrective action recommendations
- Query-document relevance analysis
- Source reliability estimation
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import Counter

import numpy as np
from loguru import logger

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spacy not available, entity matching will be limited")


class ConfidenceLevel(Enum):
    """Confidence levels for retrieval results."""
    CORRECT = "correct"  # High confidence, use directly
    AMBIGUOUS = "ambiguous"  # Medium confidence, needs additional retrieval
    INCORRECT = "incorrect"  # Low confidence, needs query rewriting or web search


class CorrectiveAction(Enum):
    """Corrective actions based on confidence assessment."""
    USE_DIRECTLY = "use_directly"
    ADDITIONAL_RETRIEVAL = "additional_retrieval"
    QUERY_REWRITE = "query_rewrite"
    WEB_SEARCH_FALLBACK = "web_search_fallback"
    EXPAND_QUERY = "expand_query"


@dataclass
class ConfidenceSignals:
    """Individual signals contributing to confidence score."""
    semantic_similarity: float  # From vector score
    keyword_overlap: float  # Keyword matching score
    entity_match: float  # Entity alignment score
    graph_connectivity: float  # Graph relationship strength
    length_appropriateness: float  # Content length quality
    source_reliability: float  # Source trust score

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "semantic_similarity": self.semantic_similarity,
            "keyword_overlap": self.keyword_overlap,
            "entity_match": self.entity_match,
            "graph_connectivity": self.graph_connectivity,
            "length_appropriateness": self.length_appropriateness,
            "source_reliability": self.source_reliability,
        }


@dataclass
class ConfidenceScore:
    """Complete confidence assessment for a retrieved result."""
    overall_score: float  # 0-1 overall confidence
    level: ConfidenceLevel
    signals: ConfidenceSignals
    corrective_action: CorrectiveAction
    reasoning: str  # Human-readable explanation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfidenceThresholds:
    """Configurable thresholds for confidence levels."""
    correct_threshold: float = 0.75
    ambiguous_threshold: float = 0.45
    semantic_weight: float = 0.30
    keyword_weight: float = 0.25
    entity_weight: float = 0.20
    graph_weight: float = 0.15
    length_weight: float = 0.05
    source_weight: float = 0.05


class ConfidenceScorer:
    """
    CRAG-style confidence scoring for retrieved results.

    Features:
    - Multi-signal confidence assessment
    - Entity-aware scoring
    - Graph connectivity analysis
    - Corrective action recommendations
    - Configurable thresholds
    """

    def __init__(
        self,
        thresholds: Optional[ConfidenceThresholds] = None,
        spacy_model: str = "en_core_web_sm"
    ):
        """
        Initialize confidence scorer.

        Args:
            thresholds: Custom confidence thresholds
            spacy_model: spaCy model for NLP tasks
        """
        self.thresholds = thresholds or ConfidenceThresholds()

        # Initialize spaCy for entity extraction
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(spacy_model)
                logger.info(f"Loaded spaCy model: {spacy_model}")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")

        # Common stopwords for keyword filtering
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }

        logger.info("Confidence scorer initialized")

    def score_result(
        self,
        query: str,
        result_content: str,
        vector_score: Optional[float] = None,
        graph_score: Optional[float] = None,
        graph_context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConfidenceScore:
        """
        Score a single retrieval result.

        Args:
            query: Original query text
            result_content: Retrieved content
            vector_score: Semantic similarity score
            graph_score: Graph-based score
            graph_context: Graph connectivity information
            metadata: Additional metadata

        Returns:
            Complete confidence assessment
        """
        # Calculate individual signals
        signals = self._calculate_signals(
            query, result_content, vector_score, graph_score, graph_context, metadata
        )

        # Calculate weighted overall score
        overall_score = self._calculate_overall_score(signals)

        # Determine confidence level
        level = self._determine_confidence_level(overall_score)

        # Recommend corrective action
        action = self._recommend_action(level, signals)

        # Generate reasoning
        reasoning = self._generate_reasoning(level, signals, overall_score)

        return ConfidenceScore(
            overall_score=overall_score,
            level=level,
            signals=signals,
            corrective_action=action,
            reasoning=reasoning,
            metadata=metadata or {}
        )

    def score_results_batch(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[ConfidenceScore]:
        """
        Score multiple results in batch.

        Args:
            query: Original query text
            results: List of result dictionaries with keys:
                - content: str
                - vector_score: Optional[float]
                - graph_score: Optional[float]
                - graph_context: Optional[Dict]
                - metadata: Optional[Dict]

        Returns:
            List of confidence scores
        """
        scores = []
        for result in results:
            score = self.score_result(
                query=query,
                result_content=result.get("content", ""),
                vector_score=result.get("vector_score"),
                graph_score=result.get("graph_score"),
                graph_context=result.get("graph_context"),
                metadata=result.get("metadata")
            )
            scores.append(score)

        return scores

    def _calculate_signals(
        self,
        query: str,
        content: str,
        vector_score: Optional[float],
        graph_score: Optional[float],
        graph_context: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]]
    ) -> ConfidenceSignals:
        """Calculate all confidence signals."""
        return ConfidenceSignals(
            semantic_similarity=self._score_semantic_similarity(vector_score),
            keyword_overlap=self._score_keyword_overlap(query, content),
            entity_match=self._score_entity_match(query, content),
            graph_connectivity=self._score_graph_connectivity(graph_score, graph_context),
            length_appropriateness=self._score_length_appropriateness(content),
            source_reliability=self._score_source_reliability(metadata)
        )

    def _score_semantic_similarity(self, vector_score: Optional[float]) -> float:
        """Score based on vector similarity."""
        if vector_score is None:
            return 0.5  # Neutral score if no vector score

        # Normalize to 0-1 range
        # Typical vector scores are 0.5-1.0 for cosine similarity
        normalized = max(0.0, min(1.0, (vector_score - 0.5) * 2))
        return normalized

    def _score_keyword_overlap(self, query: str, content: str) -> float:
        """Score based on keyword overlap between query and content."""
        # Extract keywords (non-stopwords)
        query_words = self._extract_keywords(query.lower())
        content_words = self._extract_keywords(content.lower())

        if not query_words:
            return 0.5  # Neutral if no keywords

        # Calculate Jaccard similarity
        intersection = len(query_words & content_words)
        union = len(query_words | content_words)

        if union == 0:
            return 0.0

        jaccard = intersection / union

        # Also consider recall (how many query keywords are in content)
        recall = intersection / len(query_words)

        # Combine Jaccard and recall with weights
        score = 0.4 * jaccard + 0.6 * recall

        return min(1.0, score)

    def _score_entity_match(self, query: str, content: str) -> float:
        """Score based on entity alignment between query and content."""
        if not self.nlp:
            return 0.5  # Neutral if spaCy not available

        try:
            # Extract entities
            query_entities = self._extract_entities(query)
            content_entities = self._extract_entities(content)

            if not query_entities:
                return 0.7  # High score if no entities in query

            # Calculate entity overlap
            matched_entities = query_entities & content_entities
            entity_recall = len(matched_entities) / len(query_entities)

            return entity_recall

        except Exception as e:
            logger.warning(f"Entity matching failed: {e}")
            return 0.5

    def _score_graph_connectivity(
        self,
        graph_score: Optional[float],
        graph_context: Optional[Dict[str, Any]]
    ) -> float:
        """Score based on graph connectivity strength."""
        if graph_score is None and graph_context is None:
            return 0.5  # Neutral if no graph info

        base_score = graph_score or 0.5

        # Bonus for having neighbors (indicates strong connectivity)
        if graph_context and "neighbors" in graph_context:
            neighbor_count = len(graph_context["neighbors"])
            # More neighbors = stronger connectivity
            connectivity_bonus = min(0.3, neighbor_count * 0.05)
            base_score = min(1.0, base_score + connectivity_bonus)

        return base_score

    def _score_length_appropriateness(self, content: str) -> float:
        """Score based on content length appropriateness."""
        length = len(content.strip())

        # Optimal length range: 200-2000 characters
        if length < 50:
            return 0.3  # Too short
        elif length < 200:
            return 0.6  # Short but acceptable
        elif length <= 2000:
            return 1.0  # Ideal length
        elif length <= 4000:
            return 0.8  # Longer but still good
        else:
            return 0.6  # Very long, might be unfocused

    def _score_source_reliability(self, metadata: Optional[Dict[str, Any]]) -> float:
        """Score based on source reliability indicators."""
        if not metadata:
            return 0.7  # Default reliability

        score = 0.7  # Base score

        # Bonus for having citation/source information
        if metadata.get("source") or metadata.get("url"):
            score += 0.1

        # Bonus for having author information
        if metadata.get("author"):
            score += 0.1

        # Bonus for recent documents
        if metadata.get("created_at") or metadata.get("modified_at"):
            score += 0.1

        return min(1.0, score)

    def _calculate_overall_score(self, signals: ConfidenceSignals) -> float:
        """Calculate weighted overall confidence score."""
        t = self.thresholds

        overall = (
            signals.semantic_similarity * t.semantic_weight +
            signals.keyword_overlap * t.keyword_weight +
            signals.entity_match * t.entity_weight +
            signals.graph_connectivity * t.graph_weight +
            signals.length_appropriateness * t.length_weight +
            signals.source_reliability * t.source_weight
        )

        return min(1.0, max(0.0, overall))

    def _determine_confidence_level(self, overall_score: float) -> ConfidenceLevel:
        """Determine confidence level from overall score."""
        if overall_score >= self.thresholds.correct_threshold:
            return ConfidenceLevel.CORRECT
        elif overall_score >= self.thresholds.ambiguous_threshold:
            return ConfidenceLevel.AMBIGUOUS
        else:
            return ConfidenceLevel.INCORRECT

    def _recommend_action(
        self,
        level: ConfidenceLevel,
        signals: ConfidenceSignals
    ) -> CorrectiveAction:
        """Recommend corrective action based on confidence level and signals."""
        if level == ConfidenceLevel.CORRECT:
            return CorrectiveAction.USE_DIRECTLY

        elif level == ConfidenceLevel.AMBIGUOUS:
            # Check which signals are weak
            if signals.keyword_overlap < 0.4:
                return CorrectiveAction.EXPAND_QUERY
            elif signals.graph_connectivity < 0.4:
                return CorrectiveAction.ADDITIONAL_RETRIEVAL
            else:
                return CorrectiveAction.ADDITIONAL_RETRIEVAL

        else:  # INCORRECT
            # Low semantic similarity suggests query rewriting needed
            if signals.semantic_similarity < 0.3:
                return CorrectiveAction.QUERY_REWRITE
            # Low keyword overlap suggests different retrieval strategy
            elif signals.keyword_overlap < 0.2:
                return CorrectiveAction.WEB_SEARCH_FALLBACK
            else:
                return CorrectiveAction.QUERY_REWRITE

    def _generate_reasoning(
        self,
        level: ConfidenceLevel,
        signals: ConfidenceSignals,
        overall_score: float
    ) -> str:
        """Generate human-readable reasoning for confidence assessment."""
        reasons = []

        # Overall assessment
        reasons.append(f"Overall confidence: {overall_score:.2f} ({level.value})")

        # Identify strong signals
        signal_dict = signals.to_dict()
        strong_signals = [
            name for name, value in signal_dict.items() if value >= 0.7
        ]
        if strong_signals:
            reasons.append(f"Strong signals: {', '.join(strong_signals)}")

        # Identify weak signals
        weak_signals = [
            name for name, value in signal_dict.items() if value < 0.4
        ]
        if weak_signals:
            reasons.append(f"Weak signals: {', '.join(weak_signals)}")

        # Specific recommendations
        if level == ConfidenceLevel.CORRECT:
            reasons.append("High confidence - result is likely relevant")
        elif level == ConfidenceLevel.AMBIGUOUS:
            reasons.append("Medium confidence - consider additional retrieval")
        else:
            reasons.append("Low confidence - query refinement recommended")

        return "; ".join(reasons)

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text (non-stopwords)."""
        # Simple tokenization
        words = re.findall(r'\b\w+\b', text.lower())
        keywords = {w for w in words if w not in self.stopwords and len(w) > 2}
        return keywords

    def _extract_entities(self, text: str) -> Set[str]:
        """Extract named entities from text."""
        if not self.nlp:
            return set()

        doc = self.nlp(text)
        entities = {ent.text.lower() for ent in doc.ents}
        return entities

    def aggregate_confidence(
        self,
        confidence_scores: List[ConfidenceScore]
    ) -> Dict[str, Any]:
        """
        Aggregate confidence scores across multiple results.

        Args:
            confidence_scores: List of confidence scores

        Returns:
            Aggregated statistics
        """
        if not confidence_scores:
            return {
                "average_confidence": 0.0,
                "confidence_distribution": {},
                "recommended_action": CorrectiveAction.QUERY_REWRITE.value,
                "confidence_variance": 0.0
            }

        # Calculate statistics
        scores = [cs.overall_score for cs in confidence_scores]
        avg_confidence = np.mean(scores)
        confidence_variance = np.var(scores)

        # Count by level
        level_counts = Counter(cs.level for cs in confidence_scores)
        total = len(confidence_scores)
        distribution = {
            level.value: count / total
            for level, count in level_counts.items()
        }

        # Determine overall action
        # If majority are CORRECT, use directly
        if level_counts[ConfidenceLevel.CORRECT] / total >= 0.6:
            recommended_action = CorrectiveAction.USE_DIRECTLY
        # If majority are INCORRECT, rewrite query
        elif level_counts[ConfidenceLevel.INCORRECT] / total >= 0.5:
            recommended_action = CorrectiveAction.QUERY_REWRITE
        # Otherwise additional retrieval
        else:
            recommended_action = CorrectiveAction.ADDITIONAL_RETRIEVAL

        return {
            "average_confidence": float(avg_confidence),
            "confidence_variance": float(confidence_variance),
            "confidence_distribution": distribution,
            "recommended_action": recommended_action.value,
            "total_results": total,
            "correct_count": level_counts[ConfidenceLevel.CORRECT],
            "ambiguous_count": level_counts[ConfidenceLevel.AMBIGUOUS],
            "incorrect_count": level_counts[ConfidenceLevel.INCORRECT],
        }

    def filter_by_confidence(
        self,
        results: List[Tuple[Any, ConfidenceScore]],
        min_confidence: Optional[float] = None,
        required_level: Optional[ConfidenceLevel] = None
    ) -> List[Tuple[Any, ConfidenceScore]]:
        """
        Filter results by confidence criteria.

        Args:
            results: List of (result, confidence_score) tuples
            min_confidence: Minimum confidence score
            required_level: Required confidence level

        Returns:
            Filtered results
        """
        filtered = results

        # Filter by minimum confidence
        if min_confidence is not None:
            filtered = [
                (r, cs) for r, cs in filtered
                if cs.overall_score >= min_confidence
            ]

        # Filter by required level
        if required_level is not None:
            filtered = [
                (r, cs) for r, cs in filtered
                if cs.level == required_level
            ]

        return filtered
