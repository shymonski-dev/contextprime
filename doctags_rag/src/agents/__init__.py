"""
Agentic feedback loop system for DocTags RAG.

This module implements a comprehensive multi-agent system with:
- Base agent architecture
- Planning, execution, evaluation, and learning agents
- Agent coordination and communication
- Reinforcement learning for continuous improvement
- Memory and performance monitoring
"""

from .base_agent import BaseAgent, AgentRole, AgentMessage, AgentState
from .planning_agent import PlanningAgent, QueryPlan, PlanStep
from .execution_agent import ExecutionAgent, ExecutionResult
from .evaluation_agent import EvaluationAgent, QualityAssessment
from .learning_agent import LearningAgent, LearningMetrics
from .coordinator import AgentCoordinator, CoordinationResult
from .feedback_aggregator import FeedbackAggregator, AggregatedFeedback
from .reinforcement_learning import RLModule, RewardSignal
from .memory_system import MemorySystem, ShortTermMemory, LongTermMemory
from .performance_monitor import PerformanceMonitor, PerformanceMetrics
from .agentic_pipeline import AgenticPipeline, AgenticMode

__all__ = [
    # Base classes
    "BaseAgent",
    "AgentRole",
    "AgentMessage",
    "AgentState",
    # Specialized agents
    "PlanningAgent",
    "ExecutionAgent",
    "EvaluationAgent",
    "LearningAgent",
    # Orchestration
    "AgentCoordinator",
    "FeedbackAggregator",
    # Learning
    "RLModule",
    "RewardSignal",
    "MemorySystem",
    "ShortTermMemory",
    "LongTermMemory",
    # Monitoring
    "PerformanceMonitor",
    "PerformanceMetrics",
    # Pipeline
    "AgenticPipeline",
    "AgenticMode",
    # Plan types
    "QueryPlan",
    "PlanStep",
    "ExecutionResult",
    "QualityAssessment",
    "LearningMetrics",
    "CoordinationResult",
    "AggregatedFeedback",
]
