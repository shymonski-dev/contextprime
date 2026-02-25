"""
Evaluation Agent for quality assessment and feedback generation.

The evaluation agent:
- Assesses answer quality
- Checks relevance, completeness, consistency
- Detects hallucinations
- Generates improvement feedback
- Enforces quality gates
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from .base_agent import BaseAgent, AgentRole, AgentMessage, AgentState


class QualityLevel(Enum):
    """Quality levels for assessment."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class QualityAssessment:
    """Comprehensive quality assessment."""
    query: str
    results: List[Dict[str, Any]]

    # Quality scores (0-1)
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    hallucination_score: float = 0.0  # Lower is better
    overall_score: float = 0.0

    quality_level: QualityLevel = QualityLevel.ACCEPTABLE
    meets_threshold: bool = True

    # Feedback
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)

    # Metadata
    evaluation_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvaluationAgent(BaseAgent):
    """
    Agent responsible for evaluating result quality.

    Capabilities:
    - Quality scoring
    - Relevance assessment
    - Completeness checking
    - Consistency validation
    - Feedback generation
    """

    def __init__(
        self,
        agent_id: str = "evaluator",
        min_quality_threshold: float = 0.6
    ):
        """
        Initialize evaluation agent.

        Args:
            agent_id: Agent identifier
            min_quality_threshold: Minimum acceptable quality score
        """
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.EVALUATOR,
            capabilities={
                "quality_scoring",
                "relevance_assessment",
                "completeness_checking",
                "consistency_validation",
                "hallucination_detection",
                "feedback_generation"
            }
        )

        self.min_quality_threshold = min_quality_threshold
        self.evaluations_performed = 0
        self.total_evaluation_time_ms = 0.0

    async def process_message(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """Process incoming messages."""
        content = message.content
        action = content.get("action")

        if action == "evaluate_results":
            query = content.get("query")
            results = content.get("results", [])
            assessment = await self.evaluate_results(query, results)

            return await self.send_message(
                recipient_id=message.sender_id,
                content={
                    "action": "evaluation_complete",
                    "assessment": assessment.__dict__
                }
            )

        return None

    async def execute_action(
        self,
        action_type: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute evaluation actions."""
        if action_type == "evaluate":
            return await self.evaluate_results(
                parameters["query"],
                parameters["results"]
            )
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    async def evaluate_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> QualityAssessment:
        """
        Evaluate retrieval results.

        Args:
            query: Original query
            results: Retrieved results
            context: Additional context

        Returns:
            Quality assessment
        """
        start_time = time.time()
        self.update_state(AgentState.BUSY)

        logger.info(f"Evaluating {len(results)} results for query: {query}")

        try:
            # Score different quality dimensions
            relevance = self._assess_relevance(query, results)
            completeness = self._assess_completeness(query, results)
            consistency = self._assess_consistency(results)
            hallucination = self._detect_hallucination(results)

            # Calculate overall score
            overall = (
                relevance * 0.4 +
                completeness * 0.3 +
                consistency * 0.2 +
                (1.0 - hallucination) * 0.1
            )

            # Determine quality level
            if overall >= 0.9:
                quality_level = QualityLevel.EXCELLENT
            elif overall >= 0.75:
                quality_level = QualityLevel.GOOD
            elif overall >= 0.6:
                quality_level = QualityLevel.ACCEPTABLE
            elif overall >= 0.4:
                quality_level = QualityLevel.POOR
            else:
                quality_level = QualityLevel.UNACCEPTABLE

            # Generate feedback
            strengths = self._identify_strengths(
                relevance, completeness, consistency, hallucination
            )
            weaknesses = self._identify_weaknesses(
                relevance, completeness, consistency, hallucination
            )
            suggestions = self._generate_suggestions(
                query, results, weaknesses
            )

            # Create assessment
            evaluation_time = (time.time() - start_time) * 1000
            assessment = QualityAssessment(
                query=query,
                results=results,
                relevance_score=relevance,
                completeness_score=completeness,
                consistency_score=consistency,
                hallucination_score=hallucination,
                overall_score=overall,
                quality_level=quality_level,
                meets_threshold=overall >= self.min_quality_threshold,
                strengths=strengths,
                weaknesses=weaknesses,
                improvement_suggestions=suggestions,
                evaluation_time_ms=evaluation_time
            )

            # Record action
            self.record_action(
                action_type="evaluate_results",
                parameters={"query": query, "num_results": len(results)},
                result=assessment,
                success=True,
                duration_ms=evaluation_time
            )

            self.evaluations_performed += 1
            self.total_evaluation_time_ms += evaluation_time

            logger.info(
                f"Evaluation complete: {quality_level.value} "
                f"(overall: {overall:.2f})"
            )

            return assessment

        finally:
            self.update_state(AgentState.IDLE)

    def _assess_relevance(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> float:
        """Assess relevance of results to query."""
        if not results:
            return 0.0

        # Simple keyword overlap approach
        # In production, would use semantic similarity
        query_words = set(query.lower().split())

        relevance_scores = []
        for result in results:
            content = result.get("content", "").lower()
            content_words = set(content.split())

            # Calculate overlap
            overlap = len(query_words & content_words)
            score = min(overlap / len(query_words), 1.0) if query_words else 0.0
            relevance_scores.append(score)

        # Use score from result if available
        if results and "score" in results[0]:
            result_scores = [r.get("score", 0.0) for r in results]
            avg_result_score = sum(result_scores) / len(result_scores)
            avg_overlap_score = sum(relevance_scores) / len(relevance_scores)
            return (avg_result_score * 0.7 + avg_overlap_score * 0.3)

        return sum(relevance_scores) / len(relevance_scores)

    def _assess_completeness(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> float:
        """Assess completeness of results."""
        if not results:
            return 0.0

        # Check if we have enough results
        min_desired = 5
        count_score = min(len(results) / min_desired, 1.0)

        # Check result diversity
        unique_contents = len(set(r.get("content", "")[:100] for r in results))
        diversity_score = unique_contents / len(results)

        # Check if results have substantive content
        avg_length = sum(len(r.get("content", "")) for r in results) / len(results)
        length_score = min(avg_length / 500, 1.0)  # 500 chars is good

        return (count_score * 0.4 + diversity_score * 0.3 + length_score * 0.3)

    def _assess_consistency(
        self,
        results: List[Dict[str, Any]]
    ) -> float:
        """Assess consistency across results."""
        if len(results) < 2:
            return 1.0

        # Simple consistency check
        # In production, would check for contradictions

        # Check if scores are consistent
        scores = [r.get("score", 0.5) for r in results]
        score_variance = max(scores) - min(scores)
        score_consistency = 1.0 - min(score_variance, 1.0)

        # Check metadata consistency
        sources = set(r.get("metadata", {}).get("source", "") for r in results)
        source_diversity = len(sources) / len(results)

        return (score_consistency * 0.6 + source_diversity * 0.4)

    def _detect_hallucination(
        self,
        results: List[Dict[str, Any]]
    ) -> float:
        """
        Detect potential hallucinations.

        Returns:
            Score from 0 (no hallucination) to 1 (high hallucination)
        """
        if not results:
            return 0.0

        # In production, would use fact-checking and source verification
        # For now, use simple heuristics

        hallucination_indicators = 0
        total_checks = 0

        for result in results:
            content = result.get("content", "")
            metadata = result.get("metadata", {})

            # Check if result has source
            if not metadata.get("source"):
                hallucination_indicators += 1
            total_checks += 1

            # Check for confidence indicators
            if result.get("confidence", 1.0) < 0.5:
                hallucination_indicators += 1
            total_checks += 1

        return hallucination_indicators / total_checks if total_checks > 0 else 0.0

    def _identify_strengths(
        self,
        relevance: float,
        completeness: float,
        consistency: float,
        hallucination: float
    ) -> List[str]:
        """Identify strengths in the results."""
        strengths = []

        if relevance >= 0.8:
            strengths.append("High relevance to query")
        if completeness >= 0.8:
            strengths.append("Comprehensive result set")
        if consistency >= 0.8:
            strengths.append("Consistent information across results")
        if hallucination <= 0.2:
            strengths.append("Well-sourced, reliable information")

        return strengths

    def _identify_weaknesses(
        self,
        relevance: float,
        completeness: float,
        consistency: float,
        hallucination: float
    ) -> List[str]:
        """Identify weaknesses in the results."""
        weaknesses = []

        if relevance < 0.6:
            weaknesses.append("Low relevance to query")
        if completeness < 0.6:
            weaknesses.append("Incomplete result set")
        if consistency < 0.6:
            weaknesses.append("Inconsistent information")
        if hallucination > 0.4:
            weaknesses.append("Potential hallucinations or unsourced claims")

        return weaknesses

    def _generate_suggestions(
        self,
        query: str,
        results: List[Dict[str, Any]],
        weaknesses: List[str]
    ) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []

        if "Low relevance" in str(weaknesses):
            suggestions.append("Expand query or use alternative search strategy")
        if "Incomplete result set" in str(weaknesses):
            suggestions.append("Increase top_k or use additional retrieval methods")
        if "Inconsistent information" in str(weaknesses):
            suggestions.append("Apply stronger reranking and deduplication")
        if "hallucination" in str(weaknesses).lower():
            suggestions.append("Verify sources and apply fact-checking")

        if not suggestions:
            suggestions.append("Results meet quality standards")

        return suggestions
