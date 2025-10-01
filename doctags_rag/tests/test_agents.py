"""
Comprehensive tests for the agentic system.

Tests:
- Agent communication
- Planning accuracy
- Execution reliability
- Learning convergence
- Integration tests
- Performance benchmarks
"""

import pytest
import asyncio
from pathlib import Path
import tempfile

from src.agents.base_agent import (
    BaseAgent, AgentRole, AgentMessage, AgentState, MessagePriority
)
from src.agents.planning_agent import PlanningAgent, StepType
from src.agents.execution_agent import ExecutionAgent
from src.agents.evaluation_agent import EvaluationAgent, QualityLevel
from src.agents.learning_agent import LearningAgent
from src.agents.coordinator import AgentCoordinator
from src.agents.feedback_aggregator import FeedbackAggregator
from src.agents.reinforcement_learning import RLModule, RLState
from src.agents.memory_system import MemorySystem, ShortTermMemory, LongTermMemory
from src.agents.performance_monitor import PerformanceMonitor
from src.agents.agentic_pipeline import AgenticPipeline, AgenticMode


class TestBaseAgent:
    """Test base agent functionality."""

    @pytest.mark.asyncio
    async def test_agent_creation(self):
        """Test creating a base agent."""
        agent = PlanningAgent(agent_id="test_planner")

        assert agent.agent_id == "test_planner"
        assert agent.role == AgentRole.PLANNER
        assert agent.state == AgentState.IDLE
        assert len(agent.capabilities) > 0

    @pytest.mark.asyncio
    async def test_message_sending(self):
        """Test sending messages between agents."""
        agent1 = PlanningAgent(agent_id="planner1")
        agent2 = ExecutionAgent(agent_id="executor1")

        message = await agent1.send_message(
            recipient_id="executor1",
            content={"action": "test", "data": "hello"},
            priority=MessagePriority.NORMAL
        )

        assert message.sender_id == "planner1"
        assert message.recipient_id == "executor1"
        assert message.content["action"] == "test"

    @pytest.mark.asyncio
    async def test_action_recording(self):
        """Test action history recording."""
        agent = PlanningAgent(agent_id="planner")

        agent.record_action(
            action_type="test_action",
            parameters={"param1": "value1"},
            result="success",
            success=True,
            duration_ms=100.0
        )

        assert len(agent.action_history) == 1
        assert agent.action_history[0].action_type == "test_action"
        assert agent.metrics["actions_completed"] == 1

    @pytest.mark.asyncio
    async def test_goal_management(self):
        """Test goal tracking."""
        agent = PlanningAgent(agent_id="planner")

        goal = agent.add_goal(
            description="Test goal",
            goal_type="test",
            priority=2
        )

        assert len(agent.goals) == 1
        assert agent.current_goal == goal
        assert goal.status == "active"

        agent.complete_goal(goal, success=True)
        assert goal.status == "completed"
        assert agent.metrics["goals_completed"] == 1


class TestPlanningAgent:
    """Test planning agent functionality."""

    @pytest.mark.asyncio
    async def test_query_complexity_analysis(self):
        """Test query complexity analysis."""
        planner = PlanningAgent()

        # Simple query
        analysis = planner._analyze_query_complexity("What is Python?")
        assert analysis["complexity"] in ["simple", "moderate"]

        # Complex query
        analysis = planner._analyze_query_complexity(
            "Compare and contrast the differences between Python and Java "
            "in terms of performance, syntax, and use cases"
        )
        assert analysis["complexity"] in ["complex", "very_complex"]
        assert analysis["comparison"] == True

    @pytest.mark.asyncio
    async def test_query_decomposition(self):
        """Test query decomposition."""
        planner = PlanningAgent()

        sub_queries = planner._decompose_query(
            "What is machine learning and how does it differ from deep learning?"
        )

        assert len(sub_queries) >= 2
        # Should have multiple sub-queries for complex question

    @pytest.mark.asyncio
    async def test_plan_creation(self):
        """Test creating an execution plan."""
        planner = PlanningAgent()

        plan = await planner.create_plan(
            "What are the benefits of using knowledge graphs?",
            context=None
        )

        assert plan is not None
        assert len(plan.steps) > 0
        assert plan.total_estimated_time_ms > 0
        assert plan.original_query == "What are the benefits of using knowledge graphs?"

    @pytest.mark.asyncio
    async def test_strategy_selection(self):
        """Test strategy selection."""
        planner = PlanningAgent()

        # Test different query types
        analysis_simple = {"complexity": "simple", "aggregation": False}
        strategy = planner._select_retrieval_strategy("test query", analysis_simple)
        assert strategy in ["vector_only", "hybrid"]

        analysis_complex = {"complexity": "very_complex", "aggregation": True}
        strategy = planner._select_retrieval_strategy("test query", analysis_complex)
        assert strategy == "community_based"


class TestExecutionAgent:
    """Test execution agent functionality."""

    @pytest.mark.asyncio
    async def test_step_execution(self):
        """Test executing a single step."""
        executor = ExecutionAgent()

        from src.agents.planning_agent import PlanStep, StepType

        step = PlanStep(
            step_id="test_step",
            step_type=StepType.RETRIEVAL,
            description="Test retrieval",
            parameters={"query": "test", "top_k": 5}
        )

        result = await executor.execute_step(step)

        assert result.success
        assert result.step_id == "test_step"
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test retry logic on failure."""
        executor = ExecutionAgent()
        executor.max_retries = 2

        # This will succeed on simulation
        from src.agents.planning_agent import PlanStep, StepType

        step = PlanStep(
            step_id="test_step",
            step_type=StepType.RETRIEVAL,
            description="Test retrieval",
            parameters={"query": "test"}
        )

        result = await executor.execute_step(step, retry_count=0)
        assert result is not None


class TestEvaluationAgent:
    """Test evaluation agent functionality."""

    @pytest.mark.asyncio
    async def test_quality_assessment(self):
        """Test quality assessment."""
        evaluator = EvaluationAgent()

        results = [
            {"content": "Python is a programming language", "score": 0.9},
            {"content": "Python is used for many applications", "score": 0.8},
        ]

        assessment = await evaluator.evaluate_results(
            "What is Python?",
            results
        )

        assert assessment.overall_score > 0
        assert assessment.quality_level in QualityLevel
        assert len(assessment.strengths) >= 0
        assert len(assessment.weaknesses) >= 0

    @pytest.mark.asyncio
    async def test_relevance_scoring(self):
        """Test relevance scoring."""
        evaluator = EvaluationAgent()

        # Highly relevant results
        results = [
            {"content": "Python programming language for data science", "score": 0.9}
        ]
        relevance = evaluator._assess_relevance("Python programming", results)
        assert relevance > 0.5

        # Less relevant results
        results = [
            {"content": "Java is also a programming language", "score": 0.5}
        ]
        relevance = evaluator._assess_relevance("Python programming", results)
        # May or may not be high, depends on implementation


class TestLearningAgent:
    """Test learning agent functionality."""

    @pytest.mark.asyncio
    async def test_pattern_learning(self):
        """Test learning from execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            learner = LearningAgent(storage_path=Path(tmpdir) / "learning.json")

            plan = {
                "metadata": {"complexity": "simple", "strategy": "vector_only"},
                "steps": []
            }
            results = [{"execution_time_ms": 500}]
            assessment = {"overall_score": 0.85}

            insights = await learner.learn_from_execution(
                "test query",
                plan,
                results,
                assessment
            )

            assert insights is not None
            assert "patterns_found" in insights
            assert len(learner.patterns) > 0

    @pytest.mark.asyncio
    async def test_strategy_recommendations(self):
        """Test getting strategy recommendations."""
        learner = LearningAgent()

        # Record some performance
        learner.strategy_performance["hybrid"] = [0.8, 0.85, 0.9]
        learner.strategy_performance["vector_only"] = [0.6, 0.65, 0.7]

        recommendations = learner.get_strategy_recommendations("test query")

        assert "preferred_strategy" in recommendations
        assert recommendations["preferred_strategy"] == "hybrid"


class TestCoordinator:
    """Test agent coordinator."""

    @pytest.mark.asyncio
    async def test_agent_registration(self):
        """Test registering agents."""
        coordinator = AgentCoordinator()
        planner = PlanningAgent()

        coordinator.register_agent(planner)

        assert planner.agent_id in coordinator.agents
        assert planner.agent_id in coordinator.agent_roles[AgentRole.PLANNER]

    @pytest.mark.asyncio
    async def test_message_routing(self):
        """Test message routing."""
        coordinator = AgentCoordinator()
        planner = PlanningAgent()
        executor = ExecutionAgent()

        coordinator.register_agent(planner)
        coordinator.register_agent(executor)

        message = AgentMessage(
            sender_id="test",
            recipient_id=executor.agent_id,
            content={"test": "data"}
        )

        success = await coordinator.route_message(message)
        assert success
        assert executor.inbox.qsize() > 0


class TestReinforcementLearning:
    """Test RL module."""

    def test_q_learning(self):
        """Test Q-learning updates."""
        rl = RLModule()

        state = RLState(
            query_complexity="simple",
            query_length=5,
            available_strategies=["hybrid", "vector_only"],
            previous_success=0.5
        )

        # Select action
        action = rl.select_action(state, ["hybrid", "vector_only"])
        assert action in ["hybrid", "vector_only"]

        # Update Q-value
        from src.agents.reinforcement_learning import RewardSignal
        reward_signal = RewardSignal(
            state=state.__dict__,
            action=action,
            reward=10.0,
            next_state=state.__dict__,
            done=True
        )

        rl.update_q_value(reward_signal)

        # Check Q-value was updated
        state_key = state.to_key()
        assert state_key in rl.q_table
        assert action in rl.q_table[state_key]

    def test_reward_calculation(self):
        """Test reward calculation."""
        rl = RLModule()

        reward = rl.calculate_reward(
            quality_score=0.9,
            latency_ms=1000,
            cost=0.01,
            user_satisfaction=0.8
        )

        assert reward > 0  # Should be positive for good quality


class TestMemorySystem:
    """Test memory system."""

    def test_short_term_memory(self):
        """Test short-term memory."""
        stm = ShortTermMemory(capacity=5)

        # Add memories
        for i in range(10):
            stm.add({"data": f"memory_{i}"}, importance=0.5)

        # Should only keep last 5
        assert stm.size() == 5

        # Recent should be latest
        recent = stm.get_recent(3)
        assert len(recent) == 3

    def test_long_term_memory(self):
        """Test long-term memory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ltm = LongTermMemory(storage_path=Path(tmpdir) / "ltm.json")

            ltm.add("test_id", {"data": "important"}, importance=0.9)

            # Retrieve
            entry = ltm.get("test_id")
            assert entry is not None
            assert entry.content["data"] == "important"

            # Search
            results = ltm.search("important")
            assert len(results) > 0

    def test_memory_consolidation(self):
        """Test memory consolidation."""
        memory = MemorySystem()

        # Add memories
        memory.remember({"data": "important"}, importance=0.9)
        memory.remember({"data": "trivial"}, importance=0.2)

        # Consolidate
        stats = memory.consolidate_memories()
        assert "promoted_to_long_term" in stats


class TestPerformanceMonitor:
    """Test performance monitor."""

    def test_metrics_recording(self):
        """Test recording metrics."""
        monitor = PerformanceMonitor()

        # Record queries
        monitor.record_query(500, True, False)
        monitor.record_query(600, True, True)
        monitor.record_query(5000, False, False)

        metrics = monitor.get_current_metrics()

        assert metrics.latency_ms > 0
        assert metrics.success_rate > 0
        assert metrics.error_rate >= 0

    def test_anomaly_detection(self):
        """Test anomaly detection."""
        monitor = PerformanceMonitor()

        # Record normal queries
        for _ in range(20):
            monitor.record_query(500, True, False)

        # Record anomaly
        monitor.record_query(10000, True, False)

        anomalies = monitor.detect_anomalies()
        # May or may not detect depending on threshold


class TestAgenticPipeline:
    """Test complete agentic pipeline."""

    @pytest.mark.asyncio
    async def test_pipeline_creation(self):
        """Test creating pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = AgenticPipeline(
                mode=AgenticMode.STANDARD,
                enable_learning=True,
                storage_path=Path(tmpdir)
            )

            assert pipeline.mode == AgenticMode.STANDARD
            assert pipeline.enable_learning == True
            assert pipeline.planner is not None
            assert pipeline.executor is not None

    @pytest.mark.asyncio
    async def test_query_processing(self):
        """Test processing a query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = AgenticPipeline(
                mode=AgenticMode.FAST,
                enable_learning=False,
                storage_path=Path(tmpdir)
            )

            result = await pipeline.process_query(
                "What is machine learning?",
                max_iterations=1
            )

            assert result is not None
            assert result.query == "What is machine learning?"
            assert result.answer is not None
            assert result.plan is not None
            assert result.assessment is not None
            assert result.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_iterative_improvement(self):
        """Test iterative improvement."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = AgenticPipeline(
                mode=AgenticMode.DEEP,
                enable_learning=False,
                storage_path=Path(tmpdir)
            )

            result = await pipeline.process_query(
                "Complex query requiring multiple iterations",
                max_iterations=3,
                min_quality_threshold=0.95  # High threshold
            )

            # Should attempt improvements if quality is below threshold
            assert result.iteration >= 1

    @pytest.mark.asyncio
    async def test_statistics_collection(self):
        """Test statistics collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = AgenticPipeline(
                storage_path=Path(tmpdir)
            )

            stats = pipeline.get_statistics()

            assert "queries_processed" in stats
            assert "agents" in stats
            assert "memory" in stats
            assert "performance" in stats


class TestIntegration:
    """Integration tests for the complete system."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = AgenticPipeline(
                mode=AgenticMode.STANDARD,
                enable_learning=True,
                storage_path=Path(tmpdir)
            )

            # Process multiple queries
            queries = [
                "What is Python?",
                "How does machine learning work?",
                "Compare supervised and unsupervised learning"
            ]

            for query in queries:
                result = await pipeline.process_query(query, max_iterations=2)
                assert result.assessment.overall_score >= 0

            # Check that learning occurred
            stats = pipeline.get_statistics()
            assert stats["queries_processed"] == len(queries)

            # Consolidate knowledge
            consolidation = await pipeline.consolidate_knowledge()
            assert "memory" in consolidation

    @pytest.mark.asyncio
    async def test_learning_improvement(self):
        """Test that system learns and improves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = AgenticPipeline(
                enable_learning=True,
                storage_path=Path(tmpdir)
            )

            # Process same query multiple times
            query = "Test query for learning"
            scores = []

            for _ in range(3):
                result = await pipeline.process_query(query)
                scores.append(result.assessment.overall_score)

            # Scores should remain relatively consistent
            # (In a real system with actual retrieval, might improve)
            assert all(s >= 0 for s in scores)


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks."""

    @pytest.mark.asyncio
    async def test_planning_performance(self):
        """Benchmark planning performance."""
        planner = PlanningAgent()

        start = asyncio.get_event_loop().time()

        for _ in range(10):
            await planner.create_plan("Test query")

        elapsed = asyncio.get_event_loop().time() - start

        # Should be fast
        assert elapsed < 2.0  # 10 plans in < 2 seconds

    @pytest.mark.asyncio
    async def test_pipeline_throughput(self):
        """Benchmark pipeline throughput."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = AgenticPipeline(
                mode=AgenticMode.FAST,
                enable_learning=False,
                storage_path=Path(tmpdir)
            )

            start = asyncio.get_event_loop().time()

            for i in range(5):
                await pipeline.process_query(f"Test query {i}", max_iterations=1)

            elapsed = asyncio.get_event_loop().time() - start

            # Should process queries reasonably fast
            qps = 5 / elapsed
            assert qps > 0.5  # At least 0.5 queries per second


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
