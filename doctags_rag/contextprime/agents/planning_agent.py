"""
Planning Agent for strategic query decomposition and planning.

The planning agent:
- Analyzes query complexity
- Decomposes complex queries into sub-queries
- Creates dependency graphs
- Selects optimal retrieval strategies
- Generates execution plans
- Optimizes for cost, time, and quality
"""

import asyncio
import json
import os
import re
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

from .base_agent import BaseAgent, AgentRole, AgentMessage, AgentState

try:
    from contextprime.core.config import get_settings
except Exception:  # pragma: no cover - optional in unit tests
    get_settings = None  # type: ignore[assignment]

class StepType(Enum):
    """Types of plan steps."""
    RETRIEVAL = "retrieval"
    GRAPH_QUERY = "graph_query"
    SUMMARIZATION = "summarization"
    COMMUNITY_ANALYSIS = "community_analysis"
    RAPTOR_QUERY = "raptor_query"
    RERANKING = "reranking"
    SYNTHESIS = "synthesis"


class ExecutionMode(Enum):
    """Execution modes for plan steps."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


@dataclass
class PlanStep:
    """
    A single step in an execution plan.

    Each step represents an action to be taken,
    with dependencies and execution parameters.
    """
    step_id: str
    step_type: StepType
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    estimated_time_ms: float = 0.0
    estimated_cost: float = 0.0
    priority: int = 1
    required: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryPlan:
    """
    Complete execution plan for a query.

    Includes:
    - Query decomposition
    - Ordered steps
    - Resource estimates
    - Optimization strategies
    """
    plan_id: str
    original_query: str
    sub_queries: List[str]
    steps: List[PlanStep]
    total_estimated_time_ms: float
    total_estimated_cost: float
    optimization_strategy: str
    contingency_plans: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_parallel_steps(self) -> List[List[PlanStep]]:
        """
        Group steps that can be executed in parallel.

        Returns:
            List of parallel step groups
        """
        parallel_groups = []
        remaining_steps = self.steps.copy()
        completed_steps = set()

        while remaining_steps:
            # Find steps with no unmet dependencies
            parallel_group = []
            for step in remaining_steps:
                dependencies_met = all(
                    dep in completed_steps for dep in step.dependencies
                )
                if dependencies_met and step.execution_mode == ExecutionMode.PARALLEL:
                    parallel_group.append(step)

            # If no parallel steps, take the first sequential step
            if not parallel_group:
                for step in remaining_steps:
                    dependencies_met = all(
                        dep in completed_steps for dep in step.dependencies
                    )
                    if dependencies_met:
                        parallel_group.append(step)
                        break

            if parallel_group:
                parallel_groups.append(parallel_group)
                for step in parallel_group:
                    remaining_steps.remove(step)
                    completed_steps.add(step.step_id)
            else:
                # Break to avoid infinite loop if plan is malformed
                break

        return parallel_groups


class PlanningAgent(BaseAgent):
    """
    Agent responsible for strategic planning and query decomposition.

    Capabilities:
    - Query complexity analysis
    - Query decomposition
    - Strategy selection
    - Plan generation and optimization
    - Resource estimation
    """

    def __init__(self, agent_id: str = "planner"):
        """Initialize planning agent."""
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.PLANNER,
            capabilities={
                "query_analysis",
                "decomposition",
                "strategy_selection",
                "plan_generation",
                "optimization"
            }
        )

        # Planning statistics
        self.plans_generated = 0
        self.total_planning_time_ms = 0.0

        # Optional LLM client for decomposition (gated by env var)
        self._initialize_llm_client()

    def _initialize_llm_client(self) -> None:
        """
        Initialise an optional OpenAI client for LLM-backed query decomposition.

        Activated only when ``DOCTAGS_LLM_DECOMPOSITION=true``.  Requires
        ``OPENAI_API_KEY`` (or ``settings.llm.api_key``) to be set.  Falls back
        silently to heuristic decomposition if the client cannot be created.
        """
        enabled = os.getenv("DOCTAGS_LLM_DECOMPOSITION", "false").lower()
        self._llm_decomposition_enabled: bool = enabled in {"1", "true", "yes", "on"}
        self._llm_decomposition_client: Any = None
        self._llm_decomposition_model: str = "gpt-4o-mini"

        if not self._llm_decomposition_enabled or OpenAI is None:
            return

        try:
            api_key = os.getenv("OPENAI_API_KEY", "")
            if get_settings is not None:
                try:
                    settings = get_settings()
                    api_key = api_key or (settings.llm.api_key or "")
                    self._llm_decomposition_model = (
                        getattr(settings.llm, "model", None) or "gpt-4o-mini"
                    )
                except Exception:
                    pass
            if not api_key:
                logger.debug("LLM decomposition enabled but no API key found")
                return
            self._llm_decomposition_client = OpenAI(
                api_key=api_key,
                timeout=2.0,   # short — planning must not stall
                max_retries=0,
            )
            logger.info(
                f"LLM decomposition enabled (model: {self._llm_decomposition_model})"
            )
        except Exception as exc:
            logger.warning(f"LLM decomposition client init failed: {exc}")

    async def process_message(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """
        Process incoming messages.

        Args:
            message: Incoming message

        Returns:
            Response message if needed
        """
        content = message.content
        action = content.get("action")

        if action == "create_plan":
            query = content.get("query")
            context = content.get("context", {})
            plan = await self.create_plan(query, context)

            return await self.send_message(
                recipient_id=message.sender_id,
                content={
                    "action": "plan_created",
                    "plan": plan.__dict__
                },
                parent_message_id=message.id,
            )

        return None

    async def execute_action(
        self,
        action_type: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """
        Execute planning actions.

        Args:
            action_type: Type of action
            parameters: Action parameters

        Returns:
            Action result
        """
        if action_type == "analyze_query":
            return self._analyze_query_complexity(
                parameters["query"],
                parameters.get("context")
            )
        elif action_type == "decompose_query":
            return self._decompose_query(
                parameters["query"],
                parameters.get("context")
            )
        elif action_type == "select_strategy":
            return self._select_retrieval_strategy(
                parameters["query"],
                parameters.get("analysis")
            )
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    async def create_plan(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryPlan:
        """
        Create a complete execution plan for a query.

        Args:
            query: User query
            context: Additional context

        Returns:
            Complete query plan
        """
        start_time = time.time()
        self.update_state(AgentState.BUSY)

        logger.info(f"Creating plan for query: {query}")

        try:
            # Step 1: Analyze query complexity
            analysis = self._analyze_query_complexity(query, context)
            logger.debug(f"Query complexity: {analysis['complexity']}")

            # Step 2: Decompose query if complex
            sub_queries = []
            if analysis["complexity"] in ["complex", "very_complex"]:
                sub_queries = await self._decompose_query(query, context)
                logger.debug(f"Decomposed into {len(sub_queries)} sub-queries")

            # Step 3: Select retrieval strategy
            strategy = self._select_retrieval_strategy(query, analysis)
            logger.debug(f"Selected strategy: {strategy}")

            # Step 4: Generate plan steps
            steps = self._generate_plan_steps(
                query, sub_queries, strategy, analysis, context
            )

            # Step 5: Optimize plan
            optimized_steps = self._optimize_plan(steps, context)

            # Step 6: Add contingency plans
            contingencies = self._generate_contingency_plans(
                query, analysis, strategy
            )

            # Calculate estimates
            total_time = sum(s.estimated_time_ms for s in optimized_steps)
            total_cost = sum(s.estimated_cost for s in optimized_steps)

            # Derive query type for downstream synthesis
            complexity = analysis["complexity"]
            is_complex = complexity in ("complex", "very_complex")
            if is_complex and analysis.get("reasoning"):
                derived_query_type = "analytical"
            elif is_complex and (analysis.get("multi_part") or analysis.get("comparison")):
                derived_query_type = "multi_hop"
            else:
                derived_query_type = "simple"

            # Create plan
            plan = QueryPlan(
                plan_id=f"plan_{self.plans_generated}",
                original_query=query,
                sub_queries=sub_queries,
                steps=optimized_steps,
                total_estimated_time_ms=total_time,
                total_estimated_cost=total_cost,
                optimization_strategy="balanced",
                contingency_plans=contingencies,
                metadata={
                    "complexity": analysis["complexity"],
                    "query_type": derived_query_type,
                    "strategy": strategy,
                    "analysis": analysis
                }
            )

            # Record action
            duration_ms = (time.time() - start_time) * 1000
            self.record_action(
                action_type="create_plan",
                parameters={"query": query},
                result=plan,
                success=True,
                duration_ms=duration_ms
            )

            self.plans_generated += 1
            self.total_planning_time_ms += duration_ms

            logger.info(
                f"Plan created with {len(steps)} steps "
                f"(est. time: {total_time:.0f}ms)"
            )

            return plan

        finally:
            self.update_state(AgentState.IDLE)

    def _analyze_query_complexity(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze query complexity.

        Args:
            query: Query to analyze
            context: Additional context

        Returns:
            Analysis results
        """
        words = query.lower().split()
        word_count = len(words)

        # Check for complexity indicators
        multi_part = any(
            indicator in query.lower()
            for indicator in ["and", "or", "but", "also", "additionally"]
        )

        temporal = any(
            indicator in query.lower()
            for indicator in ["before", "after", "during", "when", "since"]
        )

        comparison = any(
            indicator in query.lower()
            for indicator in ["compare", "contrast", "difference", "versus", "vs"]
        )

        aggregation = any(
            indicator in query.lower()
            for indicator in ["summarize", "overview", "all", "total", "count"]
        )

        reasoning = any(
            indicator in query.lower()
            for indicator in ["why", "how", "explain", "reason", "cause"]
        )

        # Determine complexity
        complexity_score = 0
        if word_count > 15:
            complexity_score += 2
        elif word_count > 10:
            complexity_score += 1

        if multi_part:
            complexity_score += 1
        if temporal:
            complexity_score += 1
        if comparison:
            complexity_score += 2
        if aggregation:
            complexity_score += 1
        if reasoning:
            complexity_score += 1

        if complexity_score >= 5:
            complexity = "very_complex"
        elif complexity_score >= 3:
            complexity = "complex"
        elif complexity_score >= 1:
            complexity = "moderate"
        else:
            complexity = "simple"

        return {
            "complexity": complexity,
            "word_count": word_count,
            "multi_part": multi_part,
            "temporal": temporal,
            "comparison": comparison,
            "aggregation": aggregation,
            "reasoning": reasoning,
            "complexity_score": complexity_score
        }

    async def _decompose_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Decompose a complex query into sub-queries.

        Runs heuristic decomposition first (zero cost, zero latency).  If
        heuristics cannot split the query and the LLM client is available,
        falls back to an LLM call with a 2-second timeout.  Any LLM failure
        (parse error, timeout, API error) returns the original query unchanged.

        Args:
            query: Query to decompose
            context: Additional context

        Returns:
            List of sub-queries
        """
        sub_queries = self._heuristic_decompose(query)
        if sub_queries != [query]:
            return sub_queries  # structural split found — skip LLM

        if self._llm_decomposition_client is None:
            return sub_queries

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._llm_decompose_query, query, context),
                timeout=2.0,
            )
        except Exception as exc:
            logger.debug(f"LLM decomposition failed, using heuristics: {exc}")
            return sub_queries

    def _heuristic_decompose(self, query: str) -> List[str]:
        """Rule-based query decomposition (sync, zero cost)."""
        sub_queries = []

        # Check for explicit multiple questions (multiple question marks)
        question_count = query.count("?")
        if question_count > 1:
            parts = query.split("?")
            for part in parts:
                if part.strip():
                    sub_queries.append(part.strip() + "?")

        # Check for "and" separated queries
        elif " and " in query.lower():
            query_no_q = query.rstrip("?").strip()
            parts = query_no_q.split(" and ")
            sub_queries = [p.strip() + "?" for p in parts if p.strip()]

        # Check for comparison queries
        elif any(
            word in query.lower()
            for word in ["compare", "contrast", "differ", "difference", "versus"]
        ):
            if "differ" in query.lower() or "difference" in query.lower():
                words = query.lower().replace("?", "").split()
                if "from" in words:
                    from_idx = words.index("from")
                    entity1 = " ".join(words[max(0, from_idx - 2):from_idx])
                    entity2 = " ".join(words[from_idx + 1:from_idx + 3])
                    sub_queries.append(f"What is {entity1}?")
                    sub_queries.append(f"What is {entity2}?")
                    sub_queries.append(f"How do {entity1} and {entity2} differ?")
            else:
                words = query.split()
                sub_queries.append(
                    f"Information about {words[1] if len(words) > 1 else 'entity 1'}"
                )
                sub_queries.append(
                    f"Information about {words[-1].rstrip('?') if words else 'entity 2'}"
                )
                sub_queries.append("Comparison analysis")

        if not sub_queries:
            sub_queries = [query]

        return sub_queries

    def _llm_decompose_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> List[str]:
        """
        LLM-backed decomposition (sync; runs on a thread via asyncio.to_thread).

        Returns the original query as a single-element list on any parse failure.
        """
        system = (
            "You decompose research questions into 2–4 specific sub-questions "
            "suitable for document retrieval. Return a JSON array of strings only, "
            "no explanation, no markdown."
        )
        response = self._llm_decomposition_client.chat.completions.create(
            model=self._llm_decomposition_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Decompose: {query}"},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
                result = [s.strip() for s in parsed if s.strip()]
                if result:
                    return result
        except (json.JSONDecodeError, ValueError):
            pass
        return [query]

    def _select_retrieval_strategy(
        self,
        query: str,
        analysis: Dict[str, Any]
    ) -> str:
        """
        Select optimal retrieval strategy.

        Args:
            query: Query text
            analysis: Query analysis

        Returns:
            Strategy name
        """
        complexity = analysis.get("complexity", "simple")

        if analysis.get("aggregation", False):
            return "community_based"
        elif analysis.get("comparison", False):
            return "graph_hybrid"
        elif analysis.get("temporal", False):
            return "hybrid"
        elif complexity in ["very_complex", "complex"]:
            return "raptor_hierarchical"
        elif complexity == "moderate":
            return "hybrid"
        else:
            return "vector_only"

    def _generate_plan_steps(
        self,
        query: str,
        sub_queries: List[str],
        strategy: str,
        analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[PlanStep]:
        """
        Generate plan steps based on strategy.

        Args:
            query: Original query
            sub_queries: Decomposed sub-queries
            strategy: Retrieval strategy
            analysis: Query analysis
            context: Additional context

        Returns:
            List of plan steps
        """
        steps = []
        step_counter = 0

        # Track the first retrieval step ID so downstream steps (graph, RAPTOR, community)
        # can depend on retrieval completing.
        first_retrieval_step_id = f"step_{step_counter}"

        # Initial retrieval steps
        if sub_queries:
            # Parallel retrieval for sub-queries
            for i, sub_query in enumerate(sub_queries):
                step = PlanStep(
                    step_id=f"step_{step_counter}",
                    step_type=StepType.RETRIEVAL,
                    description=f"Retrieve results for: {sub_query}",
                    parameters={
                        "query": sub_query,
                        "strategy": strategy,
                        "top_k": 10
                    },
                    dependencies=[],
                    execution_mode=ExecutionMode.PARALLEL,
                    estimated_time_ms=500.0,
                    estimated_cost=0.01
                )
                steps.append(step)
                step_counter += 1
        else:
            # Single retrieval
            step = PlanStep(
                step_id=f"step_{step_counter}",
                step_type=StepType.RETRIEVAL,
                description=f"Retrieve results for: {query}",
                parameters={
                    "query": query,
                    "strategy": strategy,
                    "top_k": 10
                },
                dependencies=[],
                execution_mode=ExecutionMode.SEQUENTIAL,
                estimated_time_ms=500.0,
                estimated_cost=0.01
            )
            steps.append(step)
            step_counter += 1

        # Add graph query if needed
        if strategy in ["graph_hybrid", "hybrid"]:
            step = PlanStep(
                step_id=f"step_{step_counter}",
                step_type=StepType.GRAPH_QUERY,
                description="Execute graph query for context",
                parameters={"query": query},
                dependencies=[first_retrieval_step_id],
                execution_mode=ExecutionMode.SEQUENTIAL,
                estimated_time_ms=300.0,
                estimated_cost=0.005
            )
            steps.append(step)
            step_counter += 1

        # Add RAPTOR if needed
        if strategy == "raptor_hierarchical":
            step = PlanStep(
                step_id=f"step_{step_counter}",
                step_type=StepType.RAPTOR_QUERY,
                description="Query RAPTOR hierarchical summaries",
                parameters={"query": query},
                dependencies=[first_retrieval_step_id],
                execution_mode=ExecutionMode.SEQUENTIAL,
                estimated_time_ms=400.0,
                estimated_cost=0.02
            )
            steps.append(step)
            step_counter += 1

        # Add community analysis if needed
        if strategy == "community_based":
            step = PlanStep(
                step_id=f"step_{step_counter}",
                step_type=StepType.COMMUNITY_ANALYSIS,
                description="Analyze relevant communities",
                parameters={"query": query},
                dependencies=[first_retrieval_step_id],
                execution_mode=ExecutionMode.SEQUENTIAL,
                estimated_time_ms=600.0,
                estimated_cost=0.015
            )
            steps.append(step)
            step_counter += 1

        # Reranking step
        reranking_deps = [s.step_id for s in steps]
        step = PlanStep(
            step_id=f"step_{step_counter}",
            step_type=StepType.RERANKING,
            description="Rerank and aggregate results",
            parameters={"query": query},
            dependencies=reranking_deps,
            execution_mode=ExecutionMode.SEQUENTIAL,
            estimated_time_ms=200.0,
            estimated_cost=0.005
        )
        steps.append(step)
        step_counter += 1

        # Final synthesis step
        step = PlanStep(
            step_id=f"step_{step_counter}",
            step_type=StepType.SYNTHESIS,
            description="Synthesize final answer",
            parameters={"query": query},
            dependencies=[f"step_{step_counter-1}"],
            execution_mode=ExecutionMode.SEQUENTIAL,
            estimated_time_ms=300.0,
            estimated_cost=0.01
        )
        steps.append(step)

        return steps

    def _optimize_plan(
        self,
        steps: List[PlanStep],
        context: Optional[Dict[str, Any]] = None
    ) -> List[PlanStep]:
        """
        Optimize plan for performance.

        Args:
            steps: Original plan steps
            context: Optimization context

        Returns:
            Optimized steps
        """
        # For now, just return steps as-is
        # In production, would:
        # - Merge similar steps
        # - Reorder for better parallelization
        # - Remove redundant steps
        # - Adjust parameters based on context

        return steps

    def _generate_contingency_plans(
        self,
        query: str,
        analysis: Dict[str, Any],
        strategy: str
    ) -> List[str]:
        """
        Generate contingency plans for failure scenarios.

        Args:
            query: Query text
            analysis: Query analysis
            strategy: Selected strategy

        Returns:
            List of contingency plan descriptions
        """
        contingencies = []

        # Low confidence contingency
        contingencies.append(
            "If confidence < 0.5: Expand query and retry with increased top_k"
        )

        # No results contingency
        contingencies.append(
            "If no results: Try alternative strategy (fallback to vector-only)"
        )

        # Performance contingency
        contingencies.append(
            "If time > 5s: Switch to fast mode (disable heavy components)"
        )

        return contingencies
