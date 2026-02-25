"""
Iterative Refinement System for Contextprime.

Implements self-reflection and multi-round retrieval:
- Evaluates initial retrieval results
- Identifies information gaps
- Generates refined queries
- Merges results intelligently
- Tracks provenance
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from loguru import logger

from .confidence_scorer import ConfidenceScorer, ConfidenceScore, ConfidenceLevel


class RefinementReason(Enum):
    """Reasons for query refinement."""
    LOW_CONFIDENCE = "low_confidence"
    MISSING_INFORMATION = "missing_information"
    AMBIGUOUS_RESULTS = "ambiguous_results"
    INSUFFICIENT_RESULTS = "insufficient_results"
    PARTIAL_ANSWER = "partial_answer"


@dataclass
class InformationGap:
    """Represents a gap in retrieved information."""
    description: str
    missing_entities: List[str] = field(default_factory=list)
    missing_keywords: List[str] = field(default_factory=list)
    suggested_query: Optional[str] = None


@dataclass
class RefinementStep:
    """Represents one refinement iteration."""
    iteration: int
    original_query: str
    refined_query: str
    reason: RefinementReason
    results_count: int
    avg_confidence: float
    information_gaps: List[InformationGap] = field(default_factory=list)


@dataclass
class RefinedResult:
    """Result with refinement metadata."""
    content: str
    score: float
    confidence: float
    source_iteration: int
    result_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class IterativeRefiner:
    """
    Iterative refinement system with self-reflection.

    Features:
    - Multi-round retrieval
    - Gap analysis
    - Query refinement generation
    - Result deduplication
    - Provenance tracking
    - Convergence detection
    """

    def __init__(
        self,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        max_iterations: int = 3,
        min_confidence_threshold: float = 0.7,
        min_results_threshold: int = 3,
        convergence_threshold: float = 0.05,
        enable_deduplication: bool = True
    ):
        """
        Initialize iterative refiner.

        Args:
            confidence_scorer: Scorer for result evaluation
            max_iterations: Maximum refinement iterations
            min_confidence_threshold: Target confidence level
            min_results_threshold: Minimum acceptable results
            convergence_threshold: Threshold for convergence detection
            enable_deduplication: Enable result deduplication
        """
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.max_iterations = max_iterations
        self.min_confidence_threshold = min_confidence_threshold
        self.min_results_threshold = min_results_threshold
        self.convergence_threshold = convergence_threshold
        self.enable_deduplication = enable_deduplication

        logger.info(
            f"Iterative refiner initialized (max_iterations={max_iterations})"
        )

    def refine_retrieval(
        self,
        original_query: str,
        initial_results: List[Dict[str, Any]],
        retrieval_func: callable,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[RefinedResult], List[RefinementStep]]:
        """
        Perform iterative refinement on retrieval results.

        Args:
            original_query: Original user query
            initial_results: Initial retrieval results
            retrieval_func: Function to call for additional retrieval
                           Should accept (query, context) and return List[Dict]
            context: Optional context to pass to retrieval_func

        Returns:
            Tuple of (refined_results, refinement_steps)
        """
        # Initialize tracking
        all_results: List[RefinedResult] = []
        refinement_steps: List[RefinementStep] = []
        seen_content_hashes: Set[str] = set()

        current_query = original_query
        iteration = 0

        logger.info(f"Starting iterative refinement for: {original_query}")

        # Process initial results
        initial_refined = self._process_results(
            results=initial_results,
            iteration=0,
            query=original_query,
            seen_hashes=seen_content_hashes
        )
        all_results.extend(initial_refined)

        # Evaluate initial results
        confidence_scores = [
            self.confidence_scorer.score_result(
                query=original_query,
                result_content=r.content,
                vector_score=r.score,
                metadata=r.metadata
            )
            for r in initial_refined
        ]

        # Check if refinement is needed
        needs_refinement, reason = self._needs_refinement(
            results=initial_refined,
            confidence_scores=confidence_scores
        )

        if not needs_refinement:
            logger.info("Initial results sufficient, no refinement needed")
            return all_results, refinement_steps

        # Iterative refinement loop
        previous_avg_confidence = 0.0

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Refinement iteration {iteration}/{self.max_iterations}")

            # Identify information gaps
            gaps = self._identify_gaps(
                query=current_query,
                results=all_results,
                confidence_scores=confidence_scores
            )

            if not gaps:
                logger.info("No information gaps identified, stopping refinement")
                break

            # Generate refined query
            refined_query = self._generate_refined_query(
                original_query=current_query,
                gaps=gaps,
                reason=reason
            )

            logger.info(f"Refined query: {refined_query}")

            # Retrieve with refined query
            try:
                new_results = retrieval_func(refined_query, context)

                # Process and add new results
                new_refined = self._process_results(
                    results=new_results,
                    iteration=iteration,
                    query=refined_query,
                    seen_hashes=seen_content_hashes
                )

                all_results.extend(new_refined)

                # Score new results
                new_confidence_scores = [
                    self.confidence_scorer.score_result(
                        query=original_query,  # Score against original query
                        result_content=r.content,
                        vector_score=r.score,
                        metadata=r.metadata
                    )
                    for r in new_refined
                ]

                confidence_scores.extend(new_confidence_scores)

                # Calculate metrics for this iteration
                avg_confidence = sum(cs.overall_score for cs in new_confidence_scores) / len(new_confidence_scores) if new_confidence_scores else 0.0

                # Record refinement step
                step = RefinementStep(
                    iteration=iteration,
                    original_query=current_query,
                    refined_query=refined_query,
                    reason=reason,
                    results_count=len(new_refined),
                    avg_confidence=avg_confidence,
                    information_gaps=gaps
                )
                refinement_steps.append(step)

                # Check for convergence
                if self._has_converged(avg_confidence, previous_avg_confidence):
                    logger.info(f"Converged after {iteration} iterations")
                    break

                previous_avg_confidence = avg_confidence

                # Check if we've met thresholds
                overall_avg_confidence = sum(cs.overall_score for cs in confidence_scores) / len(confidence_scores)
                if (overall_avg_confidence >= self.min_confidence_threshold and
                    len(all_results) >= self.min_results_threshold):
                    logger.info("Thresholds met, stopping refinement")
                    break

                # Update for next iteration
                current_query = refined_query
                needs_refinement, reason = self._needs_refinement(
                    results=all_results,
                    confidence_scores=confidence_scores
                )

                if not needs_refinement:
                    logger.info("No further refinement needed")
                    break

            except Exception as e:
                logger.error(f"Refinement iteration {iteration} failed: {e}")
                break

        # Final deduplication and ranking
        final_results = self._finalize_results(all_results, confidence_scores)

        logger.info(
            f"Refinement complete: {len(final_results)} results "
            f"after {iteration} iterations"
        )

        return final_results, refinement_steps

    def _process_results(
        self,
        results: List[Dict[str, Any]],
        iteration: int,
        query: str,
        seen_hashes: Set[str]
    ) -> List[RefinedResult]:
        """Process and deduplicate results."""
        refined_results = []

        for result in results:
            content = result.get("content", "")
            if not content:
                continue

            # Calculate content hash for deduplication
            content_hash = self._hash_content(content)

            if self.enable_deduplication and content_hash in seen_hashes:
                continue

            seen_hashes.add(content_hash)

            refined_result = RefinedResult(
                content=content,
                score=result.get("score", 0.0),
                confidence=result.get("confidence", 0.0),
                source_iteration=iteration,
                result_id=result.get("id", content_hash),
                metadata=result.get("metadata", {})
            )

            refined_results.append(refined_result)

        return refined_results

    def _needs_refinement(
        self,
        results: List[RefinedResult],
        confidence_scores: List[ConfidenceScore]
    ) -> Tuple[bool, RefinementReason]:
        """Determine if refinement is needed."""
        if not results:
            return True, RefinementReason.INSUFFICIENT_RESULTS

        if len(results) < self.min_results_threshold:
            return True, RefinementReason.INSUFFICIENT_RESULTS

        if not confidence_scores:
            return True, RefinementReason.LOW_CONFIDENCE

        # Calculate average confidence
        avg_confidence = sum(cs.overall_score for cs in confidence_scores) / len(confidence_scores)

        if avg_confidence < self.min_confidence_threshold:
            # Check distribution
            correct_count = sum(1 for cs in confidence_scores if cs.level == ConfidenceLevel.CORRECT)
            ambiguous_count = sum(1 for cs in confidence_scores if cs.level == ConfidenceLevel.AMBIGUOUS)

            if ambiguous_count > correct_count:
                return True, RefinementReason.AMBIGUOUS_RESULTS
            else:
                return True, RefinementReason.LOW_CONFIDENCE

        return False, RefinementReason.LOW_CONFIDENCE

    def _identify_gaps(
        self,
        query: str,
        results: List[RefinedResult],
        confidence_scores: List[ConfidenceScore]
    ) -> List[InformationGap]:
        """Identify information gaps in current results."""
        gaps = []

        # Extract entities from query
        query_keywords = set(query.lower().split())

        # Check for missing keywords in results
        covered_keywords = set()
        for result in results:
            result_words = set(result.content.lower().split())
            covered_keywords.update(result_words & query_keywords)

        missing_keywords = list(query_keywords - covered_keywords - {'the', 'a', 'an', 'is', 'are', 'what', 'who', 'where', 'when', 'how', 'why'})

        if missing_keywords:
            gap = InformationGap(
                description="Query keywords not covered in results",
                missing_keywords=missing_keywords[:5]  # Top 5
            )
            gaps.append(gap)

        # Check for low confidence areas
        low_confidence_results = [
            (r, cs) for r, cs in zip(results, confidence_scores)
            if cs.level in [ConfidenceLevel.AMBIGUOUS, ConfidenceLevel.INCORRECT]
        ]

        if len(low_confidence_results) > len(results) / 2:
            gap = InformationGap(
                description="Many results have low confidence",
                suggested_query=f"more details about {' '.join(missing_keywords[:3]) if missing_keywords else query}"
            )
            gaps.append(gap)

        return gaps

    def _generate_refined_query(
        self,
        original_query: str,
        gaps: List[InformationGap],
        reason: RefinementReason
    ) -> str:
        """Generate a refined query based on identified gaps."""
        # Use suggested query if available
        for gap in gaps:
            if gap.suggested_query:
                return gap.suggested_query

        # Construct refined query based on gaps
        refinements = []

        # Add missing keywords
        all_missing_keywords = []
        for gap in gaps:
            all_missing_keywords.extend(gap.missing_keywords)

        if all_missing_keywords:
            # Add top missing keywords to query
            top_missing = all_missing_keywords[:3]
            refinements.extend(top_missing)

        # Add missing entities
        all_missing_entities = []
        for gap in gaps:
            all_missing_entities.extend(gap.missing_entities)

        if all_missing_entities:
            refinements.extend(all_missing_entities[:2])

        if refinements:
            # Combine original query with refinements
            refined = f"{original_query} {' '.join(refinements)}"
            return refined

        # Fallback: rephrase based on reason
        if reason == RefinementReason.AMBIGUOUS_RESULTS:
            return f"clarify {original_query}"
        elif reason == RefinementReason.INSUFFICIENT_RESULTS:
            return f"more information about {original_query}"
        else:
            return f"detailed explanation of {original_query}"

    def _has_converged(
        self,
        current_confidence: float,
        previous_confidence: float
    ) -> bool:
        """Check if refinement has converged."""
        if previous_confidence == 0.0:
            return False

        improvement = abs(current_confidence - previous_confidence)
        return improvement < self.convergence_threshold

    def _finalize_results(
        self,
        results: List[RefinedResult],
        confidence_scores: List[ConfidenceScore]
    ) -> List[RefinedResult]:
        """Finalize results with confidence-based ranking."""
        if not results:
            return []

        # Update confidence scores
        for result, conf_score in zip(results, confidence_scores):
            result.confidence = conf_score.overall_score

        # Sort by confidence and score
        results.sort(
            key=lambda r: (r.confidence, r.score),
            reverse=True
        )

        return results

    def _hash_content(self, content: str) -> str:
        """Generate hash for content deduplication."""
        # Use first 500 chars for hashing to handle minor variations
        content_sample = content[:500].lower().strip()
        return hashlib.md5(content_sample.encode()).hexdigest()

    def merge_results(
        self,
        result_sets: List[List[RefinedResult]]
    ) -> List[RefinedResult]:
        """
        Merge multiple result sets intelligently.

        Args:
            result_sets: List of result lists to merge

        Returns:
            Merged and deduplicated results
        """
        seen_hashes = set()
        merged = []

        for result_set in result_sets:
            for result in result_set:
                content_hash = self._hash_content(result.content)
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    merged.append(result)

        # Sort by confidence and score
        merged.sort(
            key=lambda r: (r.confidence, r.score),
            reverse=True
        )

        return merged

    def get_refinement_summary(
        self,
        steps: List[RefinementStep]
    ) -> Dict[str, Any]:
        """Generate summary of refinement process."""
        if not steps:
            return {
                "total_iterations": 0,
                "refinement_applied": False
            }

        total_results = sum(s.results_count for s in steps)
        avg_confidence_progression = [s.avg_confidence for s in steps]

        return {
            "total_iterations": len(steps),
            "refinement_applied": True,
            "total_new_results": total_results,
            "confidence_progression": avg_confidence_progression,
            "final_confidence": avg_confidence_progression[-1] if avg_confidence_progression else 0.0,
            "reasons": [s.reason.value for s in steps],
            "refinement_queries": [s.refined_query for s in steps],
        }
