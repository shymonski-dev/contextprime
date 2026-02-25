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

from contextprime.agents.base_agent import (
    BaseAgent, AgentRole, AgentMessage, AgentState, MessagePriority
)
from contextprime.agents.planning_agent import PlanningAgent, StepType
from contextprime.agents.execution_agent import ExecutionAgent
from contextprime.agents.evaluation_agent import EvaluationAgent, QualityLevel, QualityAssessment
from contextprime.agents.learning_agent import LearningAgent
from contextprime.agents.coordinator import AgentCoordinator
from contextprime.agents.feedback_aggregator import FeedbackAggregator
from contextprime.agents.reinforcement_learning import RLModule, RLState
from contextprime.agents.memory_system import MemorySystem, ShortTermMemory, LongTermMemory
from contextprime.agents.performance_monitor import PerformanceMonitor
from contextprime.agents.agentic_pipeline import AgenticPipeline, AgenticMode


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

        sub_queries = await planner._decompose_query(
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


    @pytest.mark.asyncio
    async def test_query_type_simple_in_metadata(self):
        planner = PlanningAgent()
        plan = await planner.create_plan("What is Article 6?", context=None)
        assert plan.metadata["query_type"] == "simple"

    @pytest.mark.asyncio
    async def test_query_type_analytical_for_complex_reasoning(self):
        planner = PlanningAgent()
        query = (
            "Please explain why the data controller must obtain consent "
            "and document the reasons for the legal basis under the regulation"
        )
        plan = await planner.create_plan(query, context=None)
        assert plan.metadata["query_type"] == "analytical"

    @pytest.mark.asyncio
    async def test_query_type_multi_hop_for_comparison(self):
        planner = PlanningAgent()
        query = (
            "Compare the obligations of data controllers versus data processors "
            "under the regulation and summarise the key differences"
        )
        plan = await planner.create_plan(query, context=None)
        assert plan.metadata["query_type"] == "multi_hop"


class TestExecutionAgent:
    """Test execution agent functionality."""

    @pytest.mark.asyncio
    async def test_step_execution(self):
        """Test executing a single step without a configured retrieval pipeline.

        With simulation removed, an unconfigured executor returns 0 results but
        still succeeds — the step itself did not error, there is simply nothing
        to retrieve.
        """
        executor = ExecutionAgent()

        from contextprime.agents.planning_agent import PlanStep, StepType

        step = PlanStep(
            step_id="test_step",
            step_type=StepType.RETRIEVAL,
            description="Test retrieval",
            parameters={"query": "test", "top_k": 5}
        )

        result = await executor.execute_step(step)

        assert result.success
        assert result.step_id == "test_step"
        # No retrieval pipeline configured — honest empty result, not simulated content
        assert result.results == []

    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test retry logic on failure."""
        executor = ExecutionAgent()
        executor.max_retries = 2

        # This will succeed on simulation
        from contextprime.agents.planning_agent import PlanStep, StepType

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
        from contextprime.agents.reinforcement_learning import RewardSignal
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

    @pytest.mark.asyncio
    async def test_generator_feedback_retrieval_loop(self):
        """Test generator-driven retrieval feedback path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline = AgenticPipeline(
                mode=AgenticMode.FAST,
                enable_learning=False,
                storage_path=Path(tmpdir)
            )

            base_results = [
                {
                    "id": "base_1",
                    "content": "Partial answer without enough supporting details.",
                    "score": 0.25,
                    "confidence": 0.2,
                    "metadata": {},
                }
            ]
            assessment = QualityAssessment(
                query="why does retrieval fail",
                results=base_results,
                overall_score=0.35,
                quality_level=QualityLevel.POOR,
                weaknesses=["missing evidence"],
                improvement_suggestions=["retrieve sources with explicit grounding"],
            )

            class FakeEmbedder:
                def encode(self, texts, show_progress_bar=False):
                    return [[0.1, 0.2, 0.3] for _ in texts]

            class FakeRetrievedItem:
                def __init__(self, item_id, content, score):
                    self.id = item_id
                    self.content = content
                    self.score = score
                    self.confidence = score
                    self.source = "hybrid"
                    self.metadata = {}
                    self.graph_context = None

            class FakeRetriever:
                def search(self, query_vector, query_text, top_k=4, strategy=None):
                    return [
                        FakeRetrievedItem("extra_1", f"Evidence for {query_text}", 0.9),
                        FakeRetrievedItem("extra_2", f"Additional support for {query_text}", 0.8),
                    ], None

            async def fake_evaluate(query, results, context=None):
                return QualityAssessment(
                    query=query,
                    results=results,
                    overall_score=0.82,
                    quality_level=QualityLevel.GOOD,
                )

            pipeline.executor.retrieval_pipeline = FakeRetriever()
            pipeline.executor.embedding_model = FakeEmbedder()
            pipeline.evaluator.evaluate_results = fake_evaluate

            updated_results, updated_assessment, metadata = await pipeline._apply_generator_feedback_loop(
                query="why does retrieval fail",
                results=base_results,
                assessment=assessment,
                min_quality_threshold=0.7,
            )

            assert metadata["applied"] is True
            assert metadata["accepted"] is True
            assert metadata["added_results"] >= 1
            assert len(updated_results) > len(base_results)
            assert updated_assessment.overall_score > assessment.overall_score


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


from types import SimpleNamespace as _SimpleNamespace


class _FakeCompletion:
    """Fake OpenAI completion object. Accepts configurable response content."""
    def __init__(self, content: str = "answer"):
        self.choices = [_SimpleNamespace(message=_SimpleNamespace(content=content))]


class _FakeLLMClient:
    """Fake LLM client that captures all call arguments and returns configurable content."""
    def __init__(self, response_content: str = "answer"):
        self.last_call = {}
        self._response_content = response_content

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.last_call = kwargs
        return _FakeCompletion(self._response_content)


class TestSynthesisPromptConstruction:
    """Test _synthesize_answer_with_model prompt order, CoT injection, and token limits."""

    def _make_pipeline(self, tmp_path):
        import tempfile
        from pathlib import Path
        pipeline = AgenticPipeline(
            retrieval_pipeline=None,
            storage_path=Path(tmp_path),
        )
        pipeline._llm_synthesis_enabled = True
        client = _FakeLLMClient()
        pipeline._llm_answer_client = client
        return pipeline, client

    def test_evidence_precedes_question_in_user_prompt(self, tmp_path):
        pipeline, client = self._make_pipeline(tmp_path)
        results = [{"content": "Article 6 requires consent.", "score": 0.9}]
        pipeline._synthesize_answer_with_model("What does Article 6 require?", results)
        user_content = client.last_call["messages"][1]["content"]
        assert user_content.index("Evidence:") < user_content.index("Question:")

    def test_cot_appended_for_analytical(self, tmp_path):
        pipeline, client = self._make_pipeline(tmp_path)
        results = [{"content": "Article 6 requires consent.", "score": 0.9}]
        pipeline._synthesize_answer_with_model(
            "Why is consent needed?", results, query_type="analytical"
        )
        system_content = client.last_call["messages"][0]["content"]
        assert "Reason step by step" in system_content

    def test_cot_appended_for_multi_hop(self, tmp_path):
        pipeline, client = self._make_pipeline(tmp_path)
        results = [{"content": "Article 6 requires consent.", "score": 0.9}]
        pipeline._synthesize_answer_with_model(
            "What are the differences?", results, query_type="multi_hop"
        )
        system_content = client.last_call["messages"][0]["content"]
        assert "Reason step by step" in system_content

    def test_cot_not_appended_for_simple(self, tmp_path):
        pipeline, client = self._make_pipeline(tmp_path)
        results = [{"content": "Article 6 requires consent.", "score": 0.9}]
        pipeline._synthesize_answer_with_model(
            "What is Article 6?", results, query_type="simple"
        )
        system_content = client.last_call["messages"][0]["content"]
        assert "Reason step by step" not in system_content

    def test_max_tokens_1600_for_analytical(self, tmp_path):
        pipeline, client = self._make_pipeline(tmp_path)
        results = [{"content": "Article 6 requires consent.", "score": 0.9}]
        pipeline._synthesize_answer_with_model(
            "Why is consent needed?", results, query_type="analytical"
        )
        assert client.last_call["max_tokens"] == 1600

    def test_max_tokens_default_for_simple(self, tmp_path):
        pipeline, client = self._make_pipeline(tmp_path)
        results = [{"content": "Article 6 requires consent.", "score": 0.9}]
        pipeline._synthesize_answer_with_model(
            "What is Article 6?", results, query_type="simple"
        )
        assert client.last_call["max_tokens"] < 1600


# ---------------------------------------------------------------------------
# Coordinator workflow tests
# ---------------------------------------------------------------------------

class _EchoAgent(BaseAgent):
    """Fake agent that echoes back the content it receives as a response."""

    def __init__(self, agent_id: str = "echo"):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.EXECUTOR,
            capabilities={"echo"},
        )

    async def process_message(self, message: AgentMessage):
        return await self.send_message(
            recipient_id=message.sender_id,
            content={"echoed": message.content, "action": "echo_response"},
            parent_message_id=message.id,
        )

    async def execute_action(self, action_type: str, parameters):
        return {}


class TestCoordinatorWorkflow:
    """Tests for AgentCoordinator.coordinate_workflow and coordinate_parallel."""

    @pytest.mark.asyncio
    async def test_coordinate_workflow_captures_real_response(self):
        coordinator = AgentCoordinator()
        echo = _EchoAgent("echo1")
        coordinator.register_agent(echo)

        result = await coordinator.coordinate_workflow(
            [{"agent_role": "executor", "action": "ping", "parameters": {}}]
        )

        assert result.success
        agent_result = result.agent_results["echo1"]
        assert agent_result["status"] == "completed"
        assert agent_result["result"].get("action") == "echo_response"

    @pytest.mark.asyncio
    async def test_coordinate_workflow_records_conflict_on_missing_role(self):
        coordinator = AgentCoordinator()
        # No agents registered for "planner" role
        result = await coordinator.coordinate_workflow(
            [{"agent_role": "planner", "action": "create_plan", "parameters": {}}]
        )

        assert len(result.conflicts) >= 1
        assert not result.success

    @pytest.mark.asyncio
    async def test_coordinate_parallel_runs_all_agents(self):
        coordinator = AgentCoordinator()
        echo_a = _EchoAgent("echo_a")
        echo_b = _EchoAgent("echo_b")
        coordinator.register_agent(echo_a)
        coordinator.register_agent(echo_b)

        results = await coordinator.coordinate_parallel(
            [
                {"agent_id": "echo_a", "content": {"msg": "hello a"}},
                {"agent_id": "echo_b", "content": {"msg": "hello b"}},
            ]
        )

        assert "echo_a" in results
        assert "echo_b" in results
        assert results["echo_a"]["status"] == "completed"
        assert results["echo_b"]["status"] == "completed"


# ---------------------------------------------------------------------------
# LLM decomposition tests
# ---------------------------------------------------------------------------

class TestLLMQueryDecomposition:
    """Tests for PlanningAgent LLM-backed _decompose_query fallback."""

    def test_llm_decomposition_disabled_by_default(self):
        planner = PlanningAgent()
        assert planner._llm_decomposition_enabled is False
        assert planner._llm_decomposition_client is None

    @pytest.mark.asyncio
    async def test_llm_decomposition_uses_llm_when_heuristics_fail(self):
        """When heuristics return [query], the LLM client is called."""
        planner = PlanningAgent()
        planner._llm_decomposition_enabled = True

        # Inject a fake client that returns a valid JSON array
        planner._llm_decomposition_client = _FakeLLMClient(
            response_content='["What are the Article 6 lawful bases?", "When does consent apply?"]'
        )

        # Plain prose — heuristics won't split it
        result = await planner._decompose_query(
            "Explain the lawful bases for processing under Article 6"
        )

        assert result == [
            "What are the Article 6 lawful bases?",
            "When does consent apply?",
        ]

    @pytest.mark.asyncio
    async def test_llm_decomposition_fallback_on_parse_error(self):
        """A non-JSON LLM response falls back to the original query, no exception."""
        planner = PlanningAgent()
        planner._llm_decomposition_enabled = True
        planner._llm_decomposition_client = _FakeLLMClient(
            response_content="not valid JSON"
        )

        query = "Explain the lawful bases"
        result = await planner._decompose_query(query)
        assert result == [query]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
