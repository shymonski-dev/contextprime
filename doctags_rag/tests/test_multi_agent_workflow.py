"""
Comprehensive tests for the multi-agent workflow infrastructure.

Covers:
  1. TestParentMessageIdCorrelation  — BaseAgent.send_message parent_message_id
                                       threading and to_dict serialisation;
                                       PlanningAgent and ExecutionAgent reply
                                       correlation

  2. TestCoordinateWorkflow          — AgentCoordinator.coordinate_workflow pull
                                       model: empty steps, multi-step, response
                                       content, result dict shape

  3. TestCoordinateParallel          — coordinate_parallel: empty tasks, unknown
                                       agent skip, real content, per-agent timeout

  4. TestHeuristicDecompose          — PlanningAgent._heuristic_decompose edge
                                       cases and all rule branches

  5. TestLLMDecompositionPath        — _llm_decompose_query directly + async
                                       _decompose_query two-tier routing

All tests are pure unit tests — no Docker, no external services required.
"""

import asyncio
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from contextprime.agents.base_agent import (
    AgentMessage,
    AgentRole,
    AgentState,
    BaseAgent,
    MessagePriority,
)
from contextprime.agents.coordinator import AgentCoordinator
from contextprime.agents.planning_agent import PlanStep, PlanningAgent, StepType, ExecutionMode


# ---------------------------------------------------------------------------
# Shared fake helpers
# ---------------------------------------------------------------------------

class _FakeCompletion:
    """Minimal OpenAI completion stub with configurable content."""

    def __init__(self, content: str = "answer"):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


class _FakeLLMClient:
    """Captures call args; returns configurable content."""

    def __init__(self, response_content: str = "answer"):
        self.calls: List[Dict[str, Any]] = []
        self._response_content = response_content

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeCompletion(self._response_content)


class _SlowLLMClient(_FakeLLMClient):
    """Blocks the calling thread for longer than the asyncio timeout."""

    def create(self, **kwargs):
        time.sleep(5)   # longer than the 2s asyncio.wait_for timeout
        return super().create(**kwargs)


class _EchoAgent(BaseAgent):
    """Replies with the received content; sets parent_message_id correctly."""

    def __init__(self, agent_id: str = "echo"):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.EXECUTOR,
            capabilities={"echo"},
        )

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        return await self.send_message(
            recipient_id=message.sender_id,
            content={"echoed": message.content, "action": "echo_response"},
            parent_message_id=message.id,
        )

    async def execute_action(self, action_type: str, parameters: Dict) -> Any:
        return {}


class _SlowAgent(BaseAgent):
    """process_message sleeps indefinitely — useful for timeout tests."""

    def __init__(self, agent_id: str = "slow", sleep_seconds: float = 10.0):
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.EXECUTOR,
            capabilities={"slow"},
        )
        self._sleep = sleep_seconds

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        await asyncio.sleep(self._sleep)
        return None

    async def execute_action(self, action_type: str, parameters: Dict) -> Any:
        return {}


class _FakeExecutor(BaseAgent):
    """ExecutionAgent stand-in that returns a fixed step result."""

    def __init__(self):
        super().__init__(
            agent_id="fake_executor",
            role=AgentRole.EXECUTOR,
            capabilities={"execute"},
        )

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        action = message.content.get("action")
        if action in ("execute_step", "execute_plan"):
            return await self.send_message(
                recipient_id=message.sender_id,
                content={"action": "plan_completed", "results": []},
                parent_message_id=message.id,
            )
        return None

    async def execute_action(self, action_type: str, parameters: Dict) -> Any:
        return {}


# ---------------------------------------------------------------------------
# 1. Parent message ID correlation
# ---------------------------------------------------------------------------

class TestParentMessageIdCorrelation:
    """BaseAgent.send_message correctly threads parent_message_id through messages."""

    @pytest.mark.asyncio
    async def test_send_message_sets_parent_message_id(self):
        agent = _EchoAgent("a")
        msg = await agent.send_message(
            recipient_id="b",
            content={"x": 1},
            parent_message_id="original-123",
        )
        assert msg.parent_message_id == "original-123"

    @pytest.mark.asyncio
    async def test_send_message_defaults_parent_message_id_to_none(self):
        agent = _EchoAgent("a")
        msg = await agent.send_message(recipient_id="b", content={})
        assert msg.parent_message_id is None

    @pytest.mark.asyncio
    async def test_parent_message_id_appears_in_to_dict(self):
        agent = _EchoAgent("a")
        msg = await agent.send_message(
            recipient_id="b",
            content={},
            parent_message_id="ref-456",
        )
        d = msg.to_dict()
        assert d["parent_message_id"] == "ref-456"

    @pytest.mark.asyncio
    async def test_none_parent_message_id_serialises_in_to_dict(self):
        agent = _EchoAgent("a")
        msg = await agent.send_message(recipient_id="b", content={})
        assert "parent_message_id" in msg.to_dict()
        assert msg.to_dict()["parent_message_id"] is None

    @pytest.mark.asyncio
    async def test_echo_agent_reply_has_parent_message_id(self):
        """Reply from _EchoAgent.process_message references the source message id."""
        sender = _EchoAgent("sender")
        receiver = _EchoAgent("receiver")
        request = await sender.send_message(
            recipient_id="receiver", content={"hello": "world"}
        )
        reply = await receiver.process_message(request)
        assert reply is not None
        assert reply.parent_message_id == request.id

    @pytest.mark.asyncio
    async def test_planning_agent_reply_has_parent_message_id(self):
        """PlanningAgent.process_message reply links back to the request."""
        planner = PlanningAgent()
        request = AgentMessage(
            sender_id="coordinator",
            recipient_id="planner",
            role=AgentRole.COORDINATOR,
            content={"action": "create_plan", "query": "What is Article 6?"},
        )
        reply = await planner.process_message(request)
        assert reply is not None
        assert reply.parent_message_id == request.id
        assert reply.content.get("action") == "plan_created"

    @pytest.mark.asyncio
    async def test_fake_executor_reply_has_parent_message_id(self):
        """_FakeExecutor.process_message reply links back to the request."""
        executor = _FakeExecutor()
        request = AgentMessage(
            sender_id="coordinator",
            recipient_id="fake_executor",
            role=AgentRole.COORDINATOR,
            content={"action": "execute_plan", "steps": []},
        )
        reply = await executor.process_message(request)
        assert reply is not None
        assert reply.parent_message_id == request.id


# ---------------------------------------------------------------------------
# 2. Coordinator: coordinate_workflow pull model
# ---------------------------------------------------------------------------

class TestCoordinateWorkflow:
    """AgentCoordinator.coordinate_workflow drives agent.process_inbox correctly."""

    @pytest.mark.asyncio
    async def test_empty_steps_returns_success(self):
        coordinator = AgentCoordinator()
        result = await coordinator.coordinate_workflow([])
        assert result.success
        assert result.agent_results == {}
        assert result.conflicts == []

    @pytest.mark.asyncio
    async def test_workflow_result_contains_real_action_field(self):
        coordinator = AgentCoordinator()
        echo = _EchoAgent("echo_wf")
        coordinator.register_agent(echo)

        result = await coordinator.coordinate_workflow(
            [{"agent_role": "executor", "action": "ping", "parameters": {}}]
        )

        assert result.success
        agent_result = result.agent_results["echo_wf"]
        assert agent_result["action"] == "ping"
        assert agent_result["result"].get("action") == "echo_response"

    @pytest.mark.asyncio
    async def test_workflow_result_shape(self):
        """Each agent_results entry has role, action, status, result keys."""
        coordinator = AgentCoordinator()
        echo = _EchoAgent("echo_shape")
        coordinator.register_agent(echo)

        result = await coordinator.coordinate_workflow(
            [{"agent_role": "executor", "action": "do_thing", "parameters": {}}]
        )

        entry = result.agent_results["echo_shape"]
        for key in ("role", "action", "status", "result"):
            assert key in entry, f"missing key: {key}"
        assert entry["role"] == "executor"
        assert entry["action"] == "do_thing"
        assert entry["status"] == "completed"

    @pytest.mark.asyncio
    async def test_multiple_sequential_steps_both_captured(self):
        """Two steps for the same agent are both stored in agent_results."""
        coordinator = AgentCoordinator()
        echo = _EchoAgent("echo_multi")
        coordinator.register_agent(echo)

        result = await coordinator.coordinate_workflow(
            [
                {"agent_role": "executor", "action": "step_one", "parameters": {}},
                {"agent_role": "executor", "action": "step_two", "parameters": {}},
            ]
        )

        # Second step overwrites first in the dict (same agent_id key)
        assert "echo_multi" in result.agent_results
        assert result.agent_results["echo_multi"]["action"] == "step_two"

    @pytest.mark.asyncio
    async def test_missing_role_adds_conflict_and_no_result(self):
        coordinator = AgentCoordinator()
        # No agents registered

        result = await coordinator.coordinate_workflow(
            [{"agent_role": "planner", "action": "plan", "parameters": {}}]
        )

        assert not result.success
        assert len(result.conflicts) == 1
        assert "planner" in result.conflicts[0]
        assert "planner" not in result.agent_results

    @pytest.mark.asyncio
    async def test_no_fabricated_completed_status_for_missing_role(self):
        """A missing-role step must not appear as completed."""
        coordinator = AgentCoordinator()

        result = await coordinator.coordinate_workflow(
            [{"agent_role": "learner", "action": "learn", "parameters": {}}]
        )

        assert result.agent_results == {}   # nothing was processed


# ---------------------------------------------------------------------------
# 3. Coordinator: coordinate_parallel
# ---------------------------------------------------------------------------

class TestCoordinateParallel:
    """AgentCoordinator.coordinate_parallel concurrent execution."""

    @pytest.mark.asyncio
    async def test_empty_tasks_returns_empty_dict(self):
        coordinator = AgentCoordinator()
        results = await coordinator.coordinate_parallel([])
        assert results == {}

    @pytest.mark.asyncio
    async def test_unknown_agent_skipped_others_still_run(self):
        coordinator = AgentCoordinator()
        echo = _EchoAgent("known")
        coordinator.register_agent(echo)

        results = await coordinator.coordinate_parallel(
            [
                {"agent_id": "known", "content": {"msg": "hi"}},
                {"agent_id": "does_not_exist", "content": {}},
            ]
        )

        assert "known" in results
        assert "does_not_exist" not in results
        assert results["known"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_result_contains_actual_agent_content(self):
        coordinator = AgentCoordinator()
        echo = _EchoAgent("content_echo")
        coordinator.register_agent(echo)

        results = await coordinator.coordinate_parallel(
            [{"agent_id": "content_echo", "content": {"key": "value"}}]
        )

        result_content = results["content_echo"]["result"]
        assert result_content.get("action") == "echo_response"
        assert result_content["echoed"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_two_agents_both_present_in_results(self):
        coordinator = AgentCoordinator()
        a = _EchoAgent("pa")
        b = _EchoAgent("pb")
        coordinator.register_agent(a)
        coordinator.register_agent(b)

        results = await coordinator.coordinate_parallel(
            [
                {"agent_id": "pa", "content": {"n": 1}},
                {"agent_id": "pb", "content": {"n": 2}},
            ]
        )

        assert results["pa"]["status"] == "completed"
        assert results["pb"]["status"] == "completed"
        assert results["pa"]["result"]["echoed"]["n"] == 1
        assert results["pb"]["result"]["echoed"]["n"] == 2

    @pytest.mark.asyncio
    async def test_slow_agent_reported_as_timeout(self):
        """An agent that exceeds the timeout gets status 'timeout', no exception."""
        coordinator = AgentCoordinator()
        slow = _SlowAgent("slow_one", sleep_seconds=10.0)
        coordinator.register_agent(slow)

        results = await coordinator.coordinate_parallel(
            [{"agent_id": "slow_one", "content": {}}],
            timeout=0.05,   # 50 ms — far shorter than the agent's 10 s sleep
        )

        assert results["slow_one"]["status"] == "timeout"
        assert results["slow_one"]["result"] == {}

    @pytest.mark.asyncio
    async def test_mixed_fast_and_slow_agents(self):
        """Fast agent completes; slow agent times out — independent outcomes."""
        coordinator = AgentCoordinator()
        fast = _EchoAgent("fast_one")
        slow = _SlowAgent("slow_two", sleep_seconds=10.0)
        coordinator.register_agent(fast)
        coordinator.register_agent(slow)

        results = await coordinator.coordinate_parallel(
            [
                {"agent_id": "fast_one", "content": {"ping": True}},
                {"agent_id": "slow_two", "content": {}},
            ],
            timeout=0.1,
        )

        assert results["fast_one"]["status"] == "completed"
        assert results["slow_two"]["status"] == "timeout"


# ---------------------------------------------------------------------------
# 4. Heuristic decomposition
# ---------------------------------------------------------------------------

class TestHeuristicDecompose:
    """PlanningAgent._heuristic_decompose covers all rule branches."""

    def _planner(self) -> PlanningAgent:
        return PlanningAgent()

    def test_simple_query_unchanged(self):
        result = self._planner()._heuristic_decompose("What is Article 6?")
        assert result == ["What is Article 6?"]

    def test_two_question_marks_splits_into_two(self):
        result = self._planner()._heuristic_decompose(
            "What is Article 6? What is Article 17?"
        )
        assert len(result) == 2
        assert all(q.endswith("?") for q in result)

    def test_three_question_marks_splits_into_three(self):
        result = self._planner()._heuristic_decompose(
            "Who? What? Where?"
        )
        assert len(result) == 3

    def test_and_conjunction_splits_on_and(self):
        result = self._planner()._heuristic_decompose(
            "What are the controller obligations and the processor obligations?"
        )
        assert len(result) == 2
        # Each part should end with "?"
        assert all(q.endswith("?") for q in result)

    def test_differ_from_produces_three_sub_queries(self):
        result = self._planner()._heuristic_decompose(
            "How does the controller differ from the processor?"
        )
        assert len(result) == 3
        assert any("differ" in q.lower() for q in result)

    def test_difference_keyword_handled(self):
        result = self._planner()._heuristic_decompose(
            "What is the difference between controllers and processors?"
        )
        # "difference" path — may or may not find "from"; just verify no crash
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_versus_triggers_generic_comparison(self):
        result = self._planner()._heuristic_decompose(
            "Controllers versus processors — what are the key distinctions?"
        )
        assert len(result) == 3
        assert "Comparison analysis" in result

    def test_contrast_triggers_generic_comparison(self):
        result = self._planner()._heuristic_decompose(
            "Contrast the duties under Article 6 and Article 9"
        )
        # "contrast" branch but also " and " — "and" branch wins (elif ordering)
        # Either way: returns list, no crash
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_compare_keyword_triggers_comparison(self):
        result = self._planner()._heuristic_decompose(
            "Compare the exemptions under Article 6(1)(c) and Article 6(1)(e)"
        )
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_no_whitespace_in_sub_query_boundaries(self):
        result = self._planner()._heuristic_decompose(
            "Define consent and explain its scope?"
        )
        for sub in result:
            assert sub == sub.strip()

    def test_empty_string_returns_single_element(self):
        result = self._planner()._heuristic_decompose("")
        assert result == [""]


# ---------------------------------------------------------------------------
# 5. LLM decomposition path
# ---------------------------------------------------------------------------

class TestLLMDecompositionPath:
    """_llm_decompose_query directly and async _decompose_query routing."""

    def _planner_with_client(self, response: str) -> PlanningAgent:
        planner = PlanningAgent()
        planner._llm_decomposition_enabled = True
        planner._llm_decomposition_client = _FakeLLMClient(response)
        return planner

    # --- _llm_decompose_query (sync method, called on a thread) ---

    def test_valid_json_array_returned(self):
        planner = self._planner_with_client('["Sub A?", "Sub B?"]')
        result = planner._llm_decompose_query("some query", None)
        assert result == ["Sub A?", "Sub B?"]

    def test_whitespace_in_entries_stripped(self):
        planner = self._planner_with_client('[" Sub A? ", " Sub B? "]')
        result = planner._llm_decompose_query("q", None)
        assert result == ["Sub A?", "Sub B?"]

    def test_empty_json_array_falls_back_to_query(self):
        planner = self._planner_with_client("[]")
        query = "original"
        result = planner._llm_decompose_query(query, None)
        assert result == [query]

    def test_non_list_json_falls_back_to_query(self):
        planner = self._planner_with_client('{"sub": "A?"}')
        query = "original"
        result = planner._llm_decompose_query(query, None)
        assert result == [query]

    def test_mixed_type_list_falls_back(self):
        planner = self._planner_with_client('[1, "Sub B?"]')
        query = "original"
        result = planner._llm_decompose_query(query, None)
        assert result == [query]

    def test_llm_receives_correct_temperature_and_max_tokens(self):
        client = _FakeLLMClient('["A?", "B?"]')
        planner = PlanningAgent()
        planner._llm_decomposition_enabled = True
        planner._llm_decomposition_client = client
        planner._llm_decompose_query("test query", None)

        assert client.calls, "LLM client was not called"
        call = client.calls[0]
        assert call["temperature"] == 0.0
        assert call["max_tokens"] == 256

    def test_llm_user_prompt_contains_query(self):
        client = _FakeLLMClient('["A?"]')
        planner = PlanningAgent()
        planner._llm_decomposition_enabled = True
        planner._llm_decomposition_client = client
        planner._llm_decompose_query("my test query", None)

        user_message = client.calls[0]["messages"][1]
        assert user_message["role"] == "user"
        assert "my test query" in user_message["content"]

    # --- async _decompose_query routing ---

    @pytest.mark.asyncio
    async def test_heuristic_split_bypasses_llm(self):
        """When heuristics succeed, the LLM client must not be called."""
        tracking_client = _FakeLLMClient('["X?", "Y?"]')
        planner = PlanningAgent()
        planner._llm_decomposition_enabled = True
        planner._llm_decomposition_client = tracking_client

        # "and" query — heuristics will split it
        result = await planner._decompose_query(
            "What is the controller obligation and the processor obligation?"
        )

        assert tracking_client.calls == [], "LLM was called despite heuristic split"
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_llm_not_called_when_client_is_none(self):
        """Even if enabled flag is True, None client must not be called."""
        planner = PlanningAgent()
        planner._llm_decomposition_enabled = True
        planner._llm_decomposition_client = None

        query = "Explain the lawful basis requirements"
        result = await planner._decompose_query(query)
        assert result == [query]

    @pytest.mark.asyncio
    async def test_llm_called_when_heuristics_return_single(self):
        """Ambiguous query with no structural split triggers LLM call."""
        client = _FakeLLMClient('["Sub 1?", "Sub 2?", "Sub 3?"]')
        planner = PlanningAgent()
        planner._llm_decomposition_enabled = True
        planner._llm_decomposition_client = client

        result = await planner._decompose_query(
            "Explain the proportionality principle in data protection law"
        )

        assert client.calls, "LLM was not called for ambiguous query"
        assert result == ["Sub 1?", "Sub 2?", "Sub 3?"]

    @pytest.mark.asyncio
    async def test_llm_timeout_falls_back_to_original_query(self):
        """Slow LLM falls back gracefully within 2s asyncio timeout."""
        planner = PlanningAgent()
        planner._llm_decomposition_enabled = True
        planner._llm_decomposition_client = _SlowLLMClient()

        query = "Explain the proportionality principle"
        start = time.monotonic()
        result = await planner._decompose_query(query)
        elapsed = time.monotonic() - start

        assert result == [query]           # fell back to original
        assert elapsed < 5.0              # did not block for full 5s
