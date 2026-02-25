"""
Feedback Aggregator for collecting and processing feedback from all sources.
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger

from .base_agent import BaseAgent, AgentRole, AgentMessage, AgentState


@dataclass
class AggregatedFeedback:
    """Aggregated feedback from multiple sources."""
    query: str
    agent_feedback: Dict[str, Any] = field(default_factory=dict)
    user_feedback: Optional[Dict[str, Any]] = None
    system_metrics: Dict[str, Any] = field(default_factory=dict)

    weighted_score: float = 0.0
    consensus_level: float = 0.0
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FeedbackAggregator(BaseAgent):
    """
    Aggregates feedback from multiple sources.

    Sources:
    - Agent feedback (planner, executor, evaluator)
    - User feedback
    - System metrics
    - External validators
    """

    def __init__(self, agent_id: str = "feedback_aggregator"):
        """Initialize feedback aggregator."""
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.COORDINATOR,
            capabilities={
                "feedback_collection",
                "aggregation",
                "insight_generation",
                "trend_analysis"
            }
        )

        # Feedback storage
        self.feedback_history: List[AggregatedFeedback] = []
        self.feedback_by_query: Dict[str, List[AggregatedFeedback]] = defaultdict(list)

    async def process_message(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """Process incoming messages."""
        content = message.content
        action = content.get("action")

        if action == "aggregate_feedback":
            feedback = await self.aggregate_feedback(
                content.get("query"),
                content.get("agent_feedback", {}),
                content.get("user_feedback"),
                content.get("system_metrics", {})
            )

            return await self.send_message(
                recipient_id=message.sender_id,
                content={
                    "action": "feedback_aggregated",
                    "feedback": feedback.__dict__
                }
            )

        return None

    async def execute_action(
        self,
        action_type: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute aggregation actions."""
        if action_type == "aggregate":
            return await self.aggregate_feedback(
                parameters["query"],
                parameters.get("agent_feedback", {}),
                parameters.get("user_feedback"),
                parameters.get("system_metrics", {})
            )
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    async def aggregate_feedback(
        self,
        query: str,
        agent_feedback: Dict[str, Any],
        user_feedback: Optional[Dict[str, Any]] = None,
        system_metrics: Optional[Dict[str, Any]] = None
    ) -> AggregatedFeedback:
        """
        Aggregate feedback from all sources.

        Args:
            query: Original query
            agent_feedback: Feedback from agents
            user_feedback: Feedback from user
            system_metrics: System performance metrics

        Returns:
            Aggregated feedback
        """
        start_time = time.time()
        self.update_state(AgentState.BUSY)

        logger.info(f"Aggregating feedback for query: {query}")

        try:
            # Calculate weighted score
            weighted_score = self._calculate_weighted_score(
                agent_feedback, user_feedback, system_metrics
            )

            # Calculate consensus
            consensus = self._calculate_consensus(agent_feedback)

            # Generate insights
            insights = self._generate_insights(
                agent_feedback, user_feedback, system_metrics
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(
                weighted_score, consensus, insights
            )

            # Create aggregated feedback
            feedback = AggregatedFeedback(
                query=query,
                agent_feedback=agent_feedback,
                user_feedback=user_feedback,
                system_metrics=system_metrics or {},
                weighted_score=weighted_score,
                consensus_level=consensus,
                insights=insights,
                recommendations=recommendations
            )

            # Store feedback
            self.feedback_history.append(feedback)
            self.feedback_by_query[query].append(feedback)

            # Record action
            duration_ms = (time.time() - start_time) * 1000
            self.record_action(
                action_type="aggregate_feedback",
                parameters={"query": query},
                result=feedback,
                success=True,
                duration_ms=duration_ms
            )

            logger.info(
                f"Feedback aggregated: score={weighted_score:.2f}, "
                f"consensus={consensus:.2f}"
            )

            return feedback

        finally:
            self.update_state(AgentState.IDLE)

    def _calculate_weighted_score(
        self,
        agent_feedback: Dict[str, Any],
        user_feedback: Optional[Dict[str, Any]],
        system_metrics: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate weighted score from all feedback sources."""
        scores = []
        weights = []

        # Agent feedback (weight: 0.5)
        if agent_feedback:
            agent_score = agent_feedback.get("overall_score", 0.5)
            scores.append(agent_score)
            weights.append(0.5)

        # User feedback (weight: 0.3)
        if user_feedback:
            user_score = user_feedback.get("satisfaction", 0.5)
            scores.append(user_score)
            weights.append(0.3)

        # System metrics (weight: 0.2)
        if system_metrics:
            # Convert metrics to score
            latency = system_metrics.get("latency_ms", 1000)
            latency_score = max(1.0 - (latency / 5000), 0.0)
            scores.append(latency_score)
            weights.append(0.2)

        if not scores:
            return 0.5

        # Calculate weighted average
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)

        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def _calculate_consensus(
        self,
        agent_feedback: Dict[str, Any]
    ) -> float:
        """Calculate consensus level among agents."""
        # In a real system, would compare feedback from multiple agents
        # For now, return a simple measure
        return 0.8

    def _generate_insights(
        self,
        agent_feedback: Dict[str, Any],
        user_feedback: Optional[Dict[str, Any]],
        system_metrics: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate insights from feedback."""
        insights = []

        # Agent insights
        if agent_feedback:
            score = agent_feedback.get("overall_score", 0.5)
            if score >= 0.8:
                insights.append("High quality results achieved")
            elif score < 0.5:
                insights.append("Results quality below threshold")

        # User insights
        if user_feedback:
            if user_feedback.get("helpful", False):
                insights.append("User found results helpful")

        # System insights
        if system_metrics:
            latency = system_metrics.get("latency_ms", 0)
            if latency > 3000:
                insights.append("High latency detected")

        return insights

    def _generate_recommendations(
        self,
        weighted_score: float,
        consensus: float,
        insights: List[str]
    ) -> List[str]:
        """Generate recommendations based on feedback."""
        recommendations = []

        if weighted_score < 0.6:
            recommendations.append("Improve retrieval strategy")
        if consensus < 0.7:
            recommendations.append("Increase agent agreement")
        if any("latency" in i.lower() for i in insights):
            recommendations.append("Optimize for speed")

        return recommendations

    def get_trends(
        self,
        time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Analyze feedback trends.

        Args:
            time_window_minutes: Time window for analysis

        Returns:
            Trend analysis
        """
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        recent_feedback = [
            f for f in self.feedback_history
            if f.timestamp >= cutoff_time
        ]

        if not recent_feedback:
            return {"trend": "insufficient_data"}

        avg_score = sum(f.weighted_score for f in recent_feedback) / len(recent_feedback)
        avg_consensus = sum(f.consensus_level for f in recent_feedback) / len(recent_feedback)

        return {
            "time_window_minutes": time_window_minutes,
            "feedback_count": len(recent_feedback),
            "average_score": avg_score,
            "average_consensus": avg_consensus,
            "trend": "improving" if avg_score > 0.7 else "needs_attention"
        }
