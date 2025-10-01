"""
Advanced Result Reranking System for DocTags RAG.

Implements multi-strategy reranking:
- Cross-encoder reranking
- Feature-based scoring
- Learning to rank
- Diversity optimization
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

import numpy as np
from loguru import logger

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not available, cross-encoder reranking disabled")


@dataclass
class RerankingFeatures:
    """Features used for reranking."""
    semantic_score: float
    cross_encoder_score: Optional[float]
    recency_score: float
    authority_score: float
    entity_coverage: float
    diversity_score: float
    length_score: float


@dataclass
class RerankedResult:
    """Result with reranking metadata."""
    content: str
    original_score: float
    reranked_score: float
    rank: int
    original_rank: int
    features: RerankingFeatures
    result_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Reranker:
    """
    Advanced reranking system with multiple strategies.

    Features:
    - Cross-encoder based reranking
    - Feature-based scoring
    - Temporal relevance
    - Source authority
    - Entity coverage
    - Diversity optimization
    """

    def __init__(
        self,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_cross_encoder: bool = True,
        feature_weights: Optional[Dict[str, float]] = None,
        diversity_lambda: float = 0.5
    ):
        """
        Initialize reranker.

        Args:
            cross_encoder_model: Cross-encoder model name
            enable_cross_encoder: Enable cross-encoder reranking
            feature_weights: Custom feature weights
            diversity_lambda: Lambda for diversity-relevance tradeoff (0-1)
        """
        self.enable_cross_encoder = enable_cross_encoder and CROSS_ENCODER_AVAILABLE
        self.diversity_lambda = diversity_lambda

        # Initialize cross-encoder
        self.cross_encoder = None
        if self.enable_cross_encoder:
            try:
                self.cross_encoder = CrossEncoder(cross_encoder_model)
                logger.info(f"Loaded cross-encoder: {cross_encoder_model}")
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder: {e}")
                self.enable_cross_encoder = False

        # Feature weights for learning to rank
        self.feature_weights = feature_weights or {
            "semantic": 0.25,
            "cross_encoder": 0.35,
            "recency": 0.10,
            "authority": 0.10,
            "entity_coverage": 0.10,
            "diversity": 0.05,
            "length": 0.05
        }

        # Normalize weights
        total_weight = sum(self.feature_weights.values())
        self.feature_weights = {
            k: v / total_weight for k, v in self.feature_weights.items()
        }

        logger.info("Reranker initialized")

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        enable_diversity: bool = True
    ) -> List[RerankedResult]:
        """
        Rerank results using multiple strategies.

        Args:
            query: Original query
            results: List of result dicts with keys: content, score, metadata, etc.
            top_k: Number of top results to return
            enable_diversity: Enable diversity optimization

        Returns:
            Reranked results
        """
        if not results:
            return []

        logger.info(f"Reranking {len(results)} results")

        # Extract cross-encoder scores if enabled
        cross_encoder_scores = None
        if self.enable_cross_encoder:
            cross_encoder_scores = self._compute_cross_encoder_scores(query, results)

        # Compute features for each result
        reranked_results = []
        for idx, result in enumerate(results):
            features = self._compute_features(
                query=query,
                result=result,
                cross_encoder_score=cross_encoder_scores[idx] if cross_encoder_scores else None,
                result_idx=idx,
                all_results=results
            )

            # Calculate combined score
            combined_score = self._compute_combined_score(features)

            reranked_result = RerankedResult(
                content=result.get("content", ""),
                original_score=result.get("score", 0.0),
                reranked_score=combined_score,
                rank=0,  # Will be set after sorting
                original_rank=idx,
                features=features,
                result_id=result.get("id", self._generate_id(result.get("content", ""))),
                metadata=result.get("metadata", {})
            )

            reranked_results.append(reranked_result)

        # Sort by reranked score
        reranked_results.sort(key=lambda x: x.reranked_score, reverse=True)

        # Apply diversity if enabled
        if enable_diversity:
            reranked_results = self._apply_diversity_optimization(
                reranked_results, query
            )

        # Update ranks
        for rank, result in enumerate(reranked_results):
            result.rank = rank

        # Return top_k if specified
        if top_k:
            reranked_results = reranked_results[:top_k]

        logger.info(f"Reranking complete: top score = {reranked_results[0].reranked_score:.3f}")

        return reranked_results

    def _compute_cross_encoder_scores(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[float]:
        """Compute cross-encoder scores for query-document pairs."""
        if not self.cross_encoder:
            return [0.0] * len(results)

        try:
            # Prepare pairs
            pairs = [[query, result.get("content", "")] for result in results]

            # Compute scores
            scores = self.cross_encoder.predict(pairs)

            # Normalize to 0-1 range
            scores = np.array(scores)
            min_score, max_score = scores.min(), scores.max()
            if max_score > min_score:
                scores = (scores - min_score) / (max_score - min_score)
            else:
                scores = np.ones_like(scores) * 0.5

            return scores.tolist()

        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {e}")
            return [0.0] * len(results)

    def _compute_features(
        self,
        query: str,
        result: Dict[str, Any],
        cross_encoder_score: Optional[float],
        result_idx: int,
        all_results: List[Dict[str, Any]]
    ) -> RerankingFeatures:
        """Compute all reranking features for a result."""
        return RerankingFeatures(
            semantic_score=self._normalize_score(result.get("score", 0.0)),
            cross_encoder_score=cross_encoder_score if cross_encoder_score is not None else 0.5,
            recency_score=self._compute_recency_score(result.get("metadata", {})),
            authority_score=self._compute_authority_score(result.get("metadata", {})),
            entity_coverage=self._compute_entity_coverage(query, result.get("content", "")),
            diversity_score=self._compute_diversity_score(
                result.get("content", ""),
                all_results,
                result_idx
            ),
            length_score=self._compute_length_score(result.get("content", ""))
        )

    def _compute_combined_score(self, features: RerankingFeatures) -> float:
        """Combine features into final score using learned weights."""
        w = self.feature_weights

        score = (
            features.semantic_score * w["semantic"] +
            (features.cross_encoder_score or 0.5) * w["cross_encoder"] +
            features.recency_score * w["recency"] +
            features.authority_score * w["authority"] +
            features.entity_coverage * w["entity_coverage"] +
            features.diversity_score * w["diversity"] +
            features.length_score * w["length"]
        )

        return score

    def _normalize_score(self, score: float) -> float:
        """Normalize score to 0-1 range."""
        # Assume scores are typically 0-1 for cosine similarity
        return max(0.0, min(1.0, score))

    def _compute_recency_score(self, metadata: Dict[str, Any]) -> float:
        """Score based on temporal relevance."""
        # Check for date fields
        date_fields = ["created_at", "modified_at", "published_date", "date"]

        for field in date_fields:
            if field in metadata:
                try:
                    # Parse date
                    date_str = metadata[field]
                    if isinstance(date_str, str):
                        # Simple heuristic: more recent = higher score
                        # This is a simplified version
                        # In production, use proper date parsing
                        current_year = datetime.now().year
                        if str(current_year) in date_str:
                            return 1.0
                        elif str(current_year - 1) in date_str:
                            return 0.8
                        elif str(current_year - 2) in date_str:
                            return 0.6
                        else:
                            return 0.4
                except:
                    pass

        # Default: neutral score if no date info
        return 0.5

    def _compute_authority_score(self, metadata: Dict[str, Any]) -> float:
        """Score based on source authority."""
        score = 0.5  # Base score

        # Indicators of authority
        if metadata.get("source"):
            score += 0.1

        if metadata.get("author"):
            score += 0.1

        if metadata.get("citations"):
            # More citations = higher authority
            citations = metadata["citations"]
            if isinstance(citations, int):
                score += min(0.3, citations * 0.01)

        if metadata.get("verified"):
            score += 0.2

        return min(1.0, score)

    def _compute_entity_coverage(self, query: str, content: str) -> float:
        """Score based on entity coverage."""
        # Simple keyword-based entity coverage
        # Extract important words from query
        query_words = set(query.lower().split())
        stopwords = {'the', 'a', 'an', 'is', 'are', 'what', 'who', 'where', 'when', 'how', 'why', 'in', 'on', 'at'}
        query_entities = query_words - stopwords

        if not query_entities:
            return 0.5

        # Check coverage in content
        content_lower = content.lower()
        covered = sum(1 for entity in query_entities if entity in content_lower)

        coverage = covered / len(query_entities)
        return coverage

    def _compute_diversity_score(
        self,
        content: str,
        all_results: List[Dict[str, Any]],
        current_idx: int
    ) -> float:
        """Score based on diversity from other results."""
        if current_idx == 0:
            return 1.0  # First result is always diverse

        # Compare with previous results
        content_words = set(content.lower().split())

        max_similarity = 0.0
        for i in range(current_idx):
            other_content = all_results[i].get("content", "")
            other_words = set(other_content.lower().split())

            # Jaccard similarity
            if content_words and other_words:
                intersection = len(content_words & other_words)
                union = len(content_words | other_words)
                similarity = intersection / union if union > 0 else 0.0
                max_similarity = max(max_similarity, similarity)

        # Diversity is inverse of similarity
        diversity = 1.0 - max_similarity
        return diversity

    def _compute_length_score(self, content: str) -> float:
        """Score based on content length appropriateness."""
        length = len(content)

        # Optimal range: 200-2000 characters
        if length < 100:
            return 0.4
        elif length < 200:
            return 0.7
        elif length <= 2000:
            return 1.0
        elif length <= 4000:
            return 0.8
        else:
            return 0.6

    def _apply_diversity_optimization(
        self,
        results: List[RerankedResult],
        query: str
    ) -> List[RerankedResult]:
        """Apply maximal marginal relevance for diversity."""
        if len(results) <= 1:
            return results

        # MMR: balance relevance and diversity
        selected = []
        remaining = results.copy()

        # Always take the top result first
        selected.append(remaining.pop(0))

        while remaining:
            # Calculate MMR scores for remaining items
            mmr_scores = []

            for candidate in remaining:
                # Relevance component
                relevance = candidate.reranked_score

                # Diversity component (max similarity to selected)
                max_sim = 0.0
                for selected_result in selected:
                    sim = self._compute_similarity(
                        candidate.content,
                        selected_result.content
                    )
                    max_sim = max(max_sim, sim)

                # MMR score
                mmr = (
                    self.diversity_lambda * relevance -
                    (1 - self.diversity_lambda) * max_sim
                )
                mmr_scores.append(mmr)

            # Select item with highest MMR
            best_idx = np.argmax(mmr_scores)
            selected.append(remaining.pop(best_idx))

        return selected

    def _compute_similarity(self, content1: str, content2: str) -> float:
        """Compute similarity between two contents."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _generate_id(self, content: str) -> str:
        """Generate ID for content."""
        return hashlib.md5(content[:200].encode()).hexdigest()

    def update_feature_weights(
        self,
        new_weights: Dict[str, float]
    ) -> None:
        """
        Update feature weights based on feedback.

        Args:
            new_weights: New weight values
        """
        self.feature_weights.update(new_weights)

        # Normalize
        total = sum(self.feature_weights.values())
        self.feature_weights = {
            k: v / total for k, v in self.feature_weights.items()
        }

        logger.info(f"Feature weights updated: {self.feature_weights}")

    def explain_ranking(
        self,
        result: RerankedResult
    ) -> Dict[str, Any]:
        """
        Explain ranking decision for a result.

        Args:
            result: Reranked result to explain

        Returns:
            Explanation with feature contributions
        """
        features = result.features
        w = self.feature_weights

        contributions = {
            "semantic": features.semantic_score * w["semantic"],
            "cross_encoder": (features.cross_encoder_score or 0.5) * w["cross_encoder"],
            "recency": features.recency_score * w["recency"],
            "authority": features.authority_score * w["authority"],
            "entity_coverage": features.entity_coverage * w["entity_coverage"],
            "diversity": features.diversity_score * w["diversity"],
            "length": features.length_score * w["length"]
        }

        # Sort by contribution
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            "total_score": result.reranked_score,
            "rank": result.rank,
            "rank_change": result.original_rank - result.rank,
            "contributions": contributions,
            "top_features": [f[0] for f in sorted_contributions[:3]],
            "feature_values": {
                "semantic": features.semantic_score,
                "cross_encoder": features.cross_encoder_score,
                "recency": features.recency_score,
                "authority": features.authority_score,
                "entity_coverage": features.entity_coverage,
                "diversity": features.diversity_score,
                "length": features.length_score
            }
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get reranker statistics."""
        return {
            "cross_encoder_enabled": self.enable_cross_encoder,
            "feature_weights": self.feature_weights,
            "diversity_lambda": self.diversity_lambda
        }
