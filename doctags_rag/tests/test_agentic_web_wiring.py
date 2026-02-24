"""
Tier 1 mock-wiring tests for the agentic web ingestion integration.

No Docker, no Playwright. Runs always.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.planning_agent import PlanningAgent, StepType, ExecutionMode, PlanStep
from src.agents.execution_agent import ExecutionAgent
from src.agents.agentic_pipeline import AgenticPipeline


class TestStepTypeEnum:
    """WEB_INGESTION must be present in the StepType enum."""

    def test_web_ingestion_enum_exists(self):
        assert hasattr(StepType, "WEB_INGESTION")

    def test_web_ingestion_value(self):
        assert StepType.WEB_INGESTION.value == "web_ingestion"

    def test_all_original_members_intact(self):
        expected = {
            "RETRIEVAL", "GRAPH_QUERY", "SUMMARIZATION",
            "COMMUNITY_ANALYSIS", "RAPTOR_QUERY", "RERANKING",
            "SYNTHESIS", "WEB_INGESTION",
        }
        assert expected.issubset({m.name for m in StepType})


class TestURLDetectionInPlanner:
    """_generate_plan_steps inserts a WEB_INGESTION step when query contains URL."""

    def _make_planner(self) -> PlanningAgent:
        return PlanningAgent(agent_id="test_planner")

    def test_url_in_query_produces_web_ingestion_step(self):
        planner = self._make_planner()
        steps = planner._generate_plan_steps(
            query="summarise https://example.com/page",
            sub_queries=[],
            strategy="vector_only",
            analysis={"complexity": "simple"},
        )
        web_steps = [s for s in steps if s.step_type == StepType.WEB_INGESTION]
        assert len(web_steps) == 1

    def test_web_ingestion_step_carries_correct_url(self):
        planner = self._make_planner()
        url = "https://example.com/page"
        steps = planner._generate_plan_steps(
            query=f"summarise {url}",
            sub_queries=[],
            strategy="vector_only",
            analysis={"complexity": "simple"},
        )
        web_step = next(s for s in steps if s.step_type == StepType.WEB_INGESTION)
        assert web_step.parameters["url"] == url

    def test_web_ingestion_step_is_first(self):
        planner = self._make_planner()
        steps = planner._generate_plan_steps(
            query="summarise https://example.com",
            sub_queries=[],
            strategy="vector_only",
            analysis={"complexity": "simple"},
        )
        assert steps[0].step_type == StepType.WEB_INGESTION

    def test_no_url_query_has_no_web_ingestion_step(self):
        planner = self._make_planner()
        steps = planner._generate_plan_steps(
            query="what is the definition of tort law?",
            sub_queries=[],
            strategy="vector_only",
            analysis={"complexity": "simple"},
        )
        web_steps = [s for s in steps if s.step_type == StepType.WEB_INGESTION]
        assert len(web_steps) == 0

    def test_retrieval_step_depends_on_web_ingestion(self):
        """H1: RETRIEVAL must not run until WEB_INGESTION completes."""
        planner = self._make_planner()
        steps = planner._generate_plan_steps(
            query="summarise https://example.com",
            sub_queries=[],
            strategy="vector_only",
            analysis={"complexity": "simple"},
        )
        web_step = next(s for s in steps if s.step_type == StepType.WEB_INGESTION)
        retrieval_steps = [s for s in steps if s.step_type == StepType.RETRIEVAL]
        assert retrieval_steps, "Expected at least one RETRIEVAL step"
        for r in retrieval_steps:
            assert web_step.step_id in r.dependencies, (
                f"RETRIEVAL step {r.step_id} must depend on WEB_INGESTION {web_step.step_id}"
            )

    def test_retrieval_step_has_no_web_dep_without_url(self):
        """Without a URL, RETRIEVAL steps must have no dependencies."""
        planner = self._make_planner()
        steps = planner._generate_plan_steps(
            query="what is the definition of tort law?",
            sub_queries=[],
            strategy="vector_only",
            analysis={"complexity": "simple"},
        )
        retrieval_steps = [s for s in steps if s.step_type == StepType.RETRIEVAL]
        assert retrieval_steps
        for r in retrieval_steps:
            assert r.dependencies == [], (
                f"RETRIEVAL step {r.step_id} should have no deps when no URL present"
            )

    def test_graph_query_depends_on_retrieval_not_web_ingestion(self):
        """H1: GRAPH_QUERY must depend on RETRIEVAL, not WEB_INGESTION."""
        planner = self._make_planner()
        steps = planner._generate_plan_steps(
            query="summarise https://example.com",
            sub_queries=[],
            strategy="graph_hybrid",
            analysis={"complexity": "simple"},
        )
        web_step = next(s for s in steps if s.step_type == StepType.WEB_INGESTION)
        retrieval_step = next(s for s in steps if s.step_type == StepType.RETRIEVAL)
        graph_step = next(s for s in steps if s.step_type == StepType.GRAPH_QUERY)
        assert retrieval_step.step_id in graph_step.dependencies, (
            "GRAPH_QUERY must depend on RETRIEVAL"
        )
        assert web_step.step_id not in graph_step.dependencies, (
            "GRAPH_QUERY must NOT depend directly on WEB_INGESTION"
        )


class TestAgenticPipelineWiring:
    """AgenticPipeline wires web_pipeline through to ExecutionAgent."""

    def test_pipeline_passes_web_pipeline_to_executor(self):
        mock_web = MagicMock()
        with patch(
            "src.agents.agentic_pipeline.WebIngestionPipeline",
            return_value=mock_web,
        ), patch("src.agents.agentic_pipeline.HybridRetriever", side_effect=Exception("no db")):
            pipeline = AgenticPipeline()

        assert pipeline.executor.web_pipeline is mock_web

    def test_pipeline_accepts_explicit_web_pipeline(self):
        explicit_web = MagicMock()
        with patch(
            "src.agents.agentic_pipeline.WebIngestionPipeline",
        ) as mock_cls, patch(
            "src.agents.agentic_pipeline.HybridRetriever", side_effect=Exception("no db")
        ):
            pipeline = AgenticPipeline(web_pipeline=explicit_web)
            # Should not instantiate a new one when provided explicitly
            mock_cls.assert_not_called()

        assert pipeline.executor.web_pipeline is explicit_web

    def test_pipeline_tolerates_web_pipeline_init_failure(self):
        with patch(
            "src.agents.agentic_pipeline.WebIngestionPipeline",
            side_effect=Exception("playwright not installed"),
        ), patch("src.agents.agentic_pipeline.HybridRetriever", side_effect=Exception("no db")):
            pipeline = AgenticPipeline()

        assert pipeline.executor.web_pipeline is None


class TestExecutionAgentDispatcher:
    """ExecutionAgent routes WEB_INGESTION steps to _execute_web_ingestion."""

    @pytest.mark.asyncio
    async def test_web_ingestion_step_routed_correctly(self):
        mock_report = MagicMock()
        mock_report.chunks_ingested = 5
        mock_report.failed_documents = []
        mock_report.metadata = {}

        mock_web = MagicMock()
        mock_web.ingest_url = AsyncMock(return_value=mock_report)

        executor = ExecutionAgent(web_pipeline=mock_web)
        step = PlanStep(
            step_id="step_0",
            step_type=StepType.WEB_INGESTION,
            description="Ingest https://example.com",
            parameters={"url": "https://example.com"},
        )
        result = await executor.execute_step(step)

        mock_web.ingest_url.assert_awaited_once_with("https://example.com")
        assert result.success is True
        assert result.results[0]["chunks_ingested"] == 5

    @pytest.mark.asyncio
    async def test_web_ingestion_without_pipeline_returns_empty(self):
        executor = ExecutionAgent(web_pipeline=None)
        step = PlanStep(
            step_id="step_0",
            step_type=StepType.WEB_INGESTION,
            description="Ingest https://example.com",
            parameters={"url": "https://example.com"},
        )
        result = await executor.execute_step(step)

        # Should succeed (no crash) but return empty results
        assert result.results == []

    @pytest.mark.asyncio
    async def test_execute_web_ingestion_calls_pipeline_with_url(self):
        url = "https://docs.example.com/api"
        mock_report = MagicMock()
        mock_report.chunks_ingested = 3
        mock_report.failed_documents = []
        mock_report.metadata = {}

        mock_web = MagicMock()
        mock_web.ingest_url = AsyncMock(return_value=mock_report)

        executor = ExecutionAgent(web_pipeline=mock_web)
        results = await executor._execute_web_ingestion({"url": url})

        mock_web.ingest_url.assert_awaited_once_with(url)
        assert results[0]["url"] == url
        assert results[0]["chunks_ingested"] == 3
