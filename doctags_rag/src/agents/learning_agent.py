"""
Learning Agent for reinforcement learning and system adaptation.

The learning agent:
- Recognizes successful patterns
- Identifies failure modes
- Updates model weights and strategies
- Tracks performance over time
- Enables continuous improvement
"""

import time
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from loguru import logger

from .base_agent import BaseAgent, AgentRole, AgentMessage, AgentState


@dataclass
class LearningMetrics:
    """Metrics for learning performance."""
    patterns_learned: int = 0
    successful_optimizations: int = 0
    failed_optimizations: int = 0
    model_updates: int = 0
    performance_improvement: float = 0.0
    learning_rate: float = 0.1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pattern:
    """Learned pattern."""
    pattern_id: str
    pattern_type: str  # success, failure, optimization
    conditions: Dict[str, Any]
    outcomes: Dict[str, Any]
    frequency: int = 1
    confidence: float = 0.5
    last_seen: datetime = field(default_factory=datetime.now)


class LearningAgent(BaseAgent):
    """
    Agent responsible for learning and adaptation.

    Capabilities:
    - Pattern recognition
    - Strategy optimization
    - Weight adjustment
    - Performance tracking
    - Knowledge base updates
    """

    def __init__(
        self,
        agent_id: str = "learner",
        learning_rate: float = 0.1,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize learning agent.

        Args:
            agent_id: Agent identifier
            learning_rate: Learning rate for updates
            storage_path: Path to persist learned knowledge
        """
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.LEARNER,
            capabilities={
                "pattern_recognition",
                "strategy_optimization",
                "weight_adjustment",
                "performance_tracking",
                "knowledge_updates"
            }
        )

        self.learning_rate = learning_rate
        self.storage_path = storage_path or Path("data/learned_knowledge.json")

        # Learning state
        self.patterns: Dict[str, Pattern] = {}
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.query_patterns: Dict[str, int] = defaultdict(int)
        self.optimization_history: List[Dict[str, Any]] = []

        # Metrics
        self.metrics = LearningMetrics(learning_rate=learning_rate)

        # Load existing knowledge
        self._load_knowledge()

    async def process_message(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """Process incoming messages."""
        content = message.content
        action = content.get("action")

        if action == "learn_from_execution":
            query = content.get("query")
            plan = content.get("plan")
            results = content.get("results")
            assessment = content.get("assessment")

            insights = await self.learn_from_execution(
                query, plan, results, assessment
            )

            return await self.send_message(
                recipient_id=message.sender_id,
                content={
                    "action": "learning_complete",
                    "insights": insights
                }
            )
        elif action == "get_recommendations":
            query = content.get("query")
            recommendations = self.get_strategy_recommendations(query)

            return await self.send_message(
                recipient_id=message.sender_id,
                content={
                    "action": "recommendations_ready",
                    "recommendations": recommendations
                }
            )

        return None

    async def execute_action(
        self,
        action_type: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute learning actions."""
        if action_type == "learn":
            return await self.learn_from_execution(
                parameters["query"],
                parameters["plan"],
                parameters["results"],
                parameters["assessment"]
            )
        elif action_type == "optimize":
            return self.optimize_strategy(parameters["strategy_name"])
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    async def learn_from_execution(
        self,
        query: str,
        plan: Dict[str, Any],
        results: List[Dict[str, Any]],
        assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Learn from a query execution.

        Args:
            query: Original query
            plan: Execution plan used
            results: Execution results
            assessment: Quality assessment

        Returns:
            Learning insights
        """
        start_time = time.time()
        self.update_state(AgentState.BUSY)

        logger.info(f"Learning from execution for query: {query}")

        try:
            insights = {
                "patterns_found": [],
                "optimizations": [],
                "recommendations": []
            }

            # Extract key information
            strategy = plan.get("metadata", {}).get("strategy", "unknown")
            overall_score = assessment.get("overall_score", 0.0)
            execution_time = sum(r.get("execution_time_ms", 0) for r in results)

            # Record strategy performance
            self.strategy_performance[strategy].append(overall_score)

            # Track query patterns
            complexity = plan.get("metadata", {}).get("complexity", "unknown")
            pattern_key = f"{complexity}_{strategy}"
            self.query_patterns[pattern_key] += 1

            # Identify success/failure patterns
            if overall_score >= 0.8:
                pattern = self._create_pattern(
                    "success",
                    query, plan, overall_score, execution_time
                )
                self.patterns[pattern.pattern_id] = pattern
                insights["patterns_found"].append(pattern.pattern_id)
                self.metrics.patterns_learned += 1
            elif overall_score < 0.5:
                pattern = self._create_pattern(
                    "failure",
                    query, plan, overall_score, execution_time
                )
                self.patterns[pattern.pattern_id] = pattern
                insights["patterns_found"].append(pattern.pattern_id)
                self.metrics.patterns_learned += 1

            # Generate optimizations
            if overall_score < 0.7 or execution_time > 5000:
                optimizations = self._generate_optimizations(
                    strategy, overall_score, execution_time
                )
                insights["optimizations"] = optimizations

            # Update recommendations
            recommendations = self._update_recommendations(
                query, strategy, overall_score
            )
            insights["recommendations"] = recommendations

            # Record learning
            learning_time = (time.time() - start_time) * 1000
            self.record_action(
                action_type="learn_from_execution",
                parameters={"query": query, "strategy": strategy},
                result=insights,
                success=True,
                duration_ms=learning_time
            )

            # Persist knowledge periodically
            if self.metrics.patterns_learned % 10 == 0:
                self._save_knowledge()

            logger.info(
                f"Learning complete: {len(insights['patterns_found'])} patterns, "
                f"{len(insights['optimizations'])} optimizations"
            )

            return insights

        finally:
            self.update_state(AgentState.IDLE)

    def _create_pattern(
        self,
        pattern_type: str,
        query: str,
        plan: Dict[str, Any],
        score: float,
        execution_time: float
    ) -> Pattern:
        """Create a pattern from execution data."""
        pattern_id = f"{pattern_type}_{len(self.patterns)}"

        conditions = {
            "complexity": plan.get("metadata", {}).get("complexity"),
            "strategy": plan.get("metadata", {}).get("strategy"),
            "query_length": len(query.split()),
            "step_count": len(plan.get("steps", []))
        }

        outcomes = {
            "score": score,
            "execution_time_ms": execution_time,
            "success": score >= 0.7
        }

        return Pattern(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            conditions=conditions,
            outcomes=outcomes
        )

    def _generate_optimizations(
        self,
        strategy: str,
        score: float,
        execution_time: float
    ) -> List[str]:
        """Generate optimization suggestions."""
        optimizations = []

        if score < 0.5:
            optimizations.append(f"Switch from {strategy} to alternative strategy")
            optimizations.append("Increase query expansion aggressiveness")

        if execution_time > 3000:
            optimizations.append("Enable result caching")
            optimizations.append("Reduce top_k to improve speed")

        if score >= 0.5 and score < 0.7:
            optimizations.append("Apply stronger reranking")
            optimizations.append("Increase refinement iterations")

        return optimizations

    def _update_recommendations(
        self,
        query: str,
        strategy: str,
        score: float
    ) -> List[str]:
        """Update strategy recommendations."""
        recommendations = []

        # Analyze strategy performance
        if strategy in self.strategy_performance:
            scores = self.strategy_performance[strategy]
            avg_score = sum(scores) / len(scores)

            if avg_score < 0.6 and len(scores) >= 5:
                recommendations.append(
                    f"Strategy '{strategy}' showing poor performance (avg: {avg_score:.2f})"
                )
                recommendations.append("Consider alternative strategies")

        # Analyze query patterns
        query_length = len(query.split())
        if query_length > 15:
            recommendations.append(
                "Long query detected: Consider query decomposition"
            )

        return recommendations

    def get_strategy_recommendations(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get strategy recommendations based on learned patterns.

        Args:
            query: Query to analyze
            context: Additional context

        Returns:
            Recommendations dictionary
        """
        recommendations = {
            "preferred_strategy": None,
            "alternative_strategies": [],
            "parameter_adjustments": {},
            "confidence": 0.0
        }

        # Analyze query
        query_length = len(query.split())

        # Find best performing strategy for similar queries
        best_strategy = None
        best_score = 0.0

        for strategy, scores in self.strategy_performance.items():
            if not scores:
                continue

            avg_score = sum(scores) / len(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_strategy = strategy

        if best_strategy:
            recommendations["preferred_strategy"] = best_strategy
            recommendations["confidence"] = best_score

        # Get alternatives
        sorted_strategies = sorted(
            self.strategy_performance.items(),
            key=lambda x: sum(x[1]) / len(x[1]) if x[1] else 0,
            reverse=True
        )
        recommendations["alternative_strategies"] = [
            s[0] for s in sorted_strategies[1:4]
        ]

        # Parameter adjustments based on patterns
        if query_length > 15:
            recommendations["parameter_adjustments"]["enable_decomposition"] = True
        if query_length < 5:
            recommendations["parameter_adjustments"]["enable_expansion"] = True

        return recommendations

    def optimize_strategy(
        self,
        strategy_name: str
    ) -> Dict[str, Any]:
        """
        Optimize a specific strategy based on learned patterns.

        Args:
            strategy_name: Strategy to optimize

        Returns:
            Optimization results
        """
        if strategy_name not in self.strategy_performance:
            return {"success": False, "reason": "Strategy not found"}

        scores = self.strategy_performance[strategy_name]
        if not scores:
            return {"success": False, "reason": "No performance data"}

        avg_score = sum(scores) / len(scores)

        optimizations = {
            "current_performance": avg_score,
            "sample_size": len(scores),
            "recommended_changes": []
        }

        # Generate recommendations
        if avg_score < 0.6:
            optimizations["recommended_changes"].append(
                "Increase query expansion weight"
            )
            optimizations["recommended_changes"].append(
                "Enable additional retrieval rounds"
            )
        elif avg_score < 0.75:
            optimizations["recommended_changes"].append(
                "Tune reranking parameters"
            )

        self.metrics.model_updates += 1

        return optimizations

    def _load_knowledge(self) -> None:
        """Load previously learned knowledge."""
        if not self.storage_path.exists():
            logger.info("No existing knowledge to load")
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Restore patterns
            for p_data in data.get("patterns", []):
                pattern = Pattern(
                    pattern_id=p_data["pattern_id"],
                    pattern_type=p_data["pattern_type"],
                    conditions=p_data["conditions"],
                    outcomes=p_data["outcomes"],
                    frequency=p_data.get("frequency", 1),
                    confidence=p_data.get("confidence", 0.5)
                )
                self.patterns[pattern.pattern_id] = pattern

            # Restore strategy performance
            self.strategy_performance = defaultdict(
                list,
                data.get("strategy_performance", {})
            )

            logger.info(f"Loaded {len(self.patterns)} patterns from storage")

        except Exception as e:
            logger.error(f"Error loading knowledge: {e}")

    def _save_knowledge(self) -> None:
        """Save learned knowledge to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "patterns": [
                    {
                        "pattern_id": p.pattern_id,
                        "pattern_type": p.pattern_type,
                        "conditions": p.conditions,
                        "outcomes": p.outcomes,
                        "frequency": p.frequency,
                        "confidence": p.confidence
                    }
                    for p in self.patterns.values()
                ],
                "strategy_performance": dict(self.strategy_performance),
                "metrics": self.metrics.__dict__
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.patterns)} patterns to storage")

        except Exception as e:
            logger.error(f"Error saving knowledge: {e}")
