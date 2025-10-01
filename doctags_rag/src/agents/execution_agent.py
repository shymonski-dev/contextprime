"""
Execution Agent for performing retrieval and processing actions.

The execution agent:
- Executes retrieval operations
- Performs graph queries
- Runs summarization tasks
- Monitors progress
- Handles errors and retries
- Collects results and provenance
"""

import time
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from .base_agent import BaseAgent, AgentRole, AgentMessage, AgentState
from .planning_agent import PlanStep, StepType


@dataclass
class ExecutionResult:
    """Result from executing a plan step."""
    step_id: str
    success: bool
    results: List[Dict[str, Any]]
    execution_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance: List[str] = field(default_factory=list)
    confidence: float = 0.0


class ExecutionAgent(BaseAgent):
    """
    Agent responsible for executing retrieval and processing actions.

    Capabilities:
    - Execute retrieval operations
    - Perform graph queries
    - Run summarization
    - Handle errors gracefully
    - Collect provenance information
    """

    def __init__(
        self,
        agent_id: str = "executor",
        retrieval_pipeline: Optional[Any] = None,
        graph_queries: Optional[Any] = None,
        raptor_pipeline: Optional[Any] = None,
        community_pipeline: Optional[Any] = None
    ):
        """
        Initialize execution agent.

        Args:
            agent_id: Agent identifier
            retrieval_pipeline: Advanced retrieval pipeline
            graph_queries: Graph query handler
            raptor_pipeline: RAPTOR pipeline
            community_pipeline: Community detection pipeline
        """
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.EXECUTOR,
            capabilities={
                "retrieval",
                "graph_query",
                "summarization",
                "community_analysis",
                "raptor_query",
                "retry_logic"
            }
        )

        # External components
        self.retrieval_pipeline = retrieval_pipeline
        self.graph_queries = graph_queries
        self.raptor_pipeline = raptor_pipeline
        self.community_pipeline = community_pipeline

        # Execution statistics
        self.steps_executed = 0
        self.steps_failed = 0
        self.total_execution_time_ms = 0.0

        # Retry configuration
        self.max_retries = 3
        self.retry_delay_ms = 500

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

        if action == "execute_step":
            step = content.get("step")
            result = await self.execute_step(step)

            return await self.send_message(
                recipient_id=message.sender_id,
                content={
                    "action": "step_completed",
                    "result": result.__dict__
                }
            )
        elif action == "execute_plan":
            steps = content.get("steps", [])
            results = await self.execute_plan(steps)

            return await self.send_message(
                recipient_id=message.sender_id,
                content={
                    "action": "plan_completed",
                    "results": [r.__dict__ for r in results]
                }
            )

        return None

    async def execute_action(
        self,
        action_type: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """
        Execute a specific action.

        Args:
            action_type: Type of action
            parameters: Action parameters

        Returns:
            Action result
        """
        if action_type == "retrieval":
            return await self._execute_retrieval(parameters)
        elif action_type == "graph_query":
            return await self._execute_graph_query(parameters)
        elif action_type == "summarization":
            return await self._execute_summarization(parameters)
        elif action_type == "community_analysis":
            return await self._execute_community_analysis(parameters)
        elif action_type == "raptor_query":
            return await self._execute_raptor_query(parameters)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    async def execute_step(
        self,
        step: PlanStep,
        retry_count: int = 0
    ) -> ExecutionResult:
        """
        Execute a single plan step with retry logic.

        Args:
            step: Plan step to execute
            retry_count: Current retry attempt

        Returns:
            Execution result
        """
        start_time = time.time()
        self.update_state(AgentState.BUSY)

        logger.info(f"Executing step: {step.description}")

        try:
            # Execute based on step type
            if step.step_type == StepType.RETRIEVAL:
                results = await self._execute_retrieval(step.parameters)
            elif step.step_type == StepType.GRAPH_QUERY:
                results = await self._execute_graph_query(step.parameters)
            elif step.step_type == StepType.SUMMARIZATION:
                results = await self._execute_summarization(step.parameters)
            elif step.step_type == StepType.COMMUNITY_ANALYSIS:
                results = await self._execute_community_analysis(step.parameters)
            elif step.step_type == StepType.RAPTOR_QUERY:
                results = await self._execute_raptor_query(step.parameters)
            elif step.step_type == StepType.RERANKING:
                results = await self._execute_reranking(step.parameters)
            elif step.step_type == StepType.SYNTHESIS:
                results = await self._execute_synthesis(step.parameters)
            else:
                raise ValueError(f"Unknown step type: {step.step_type}")

            # Calculate confidence
            confidence = self._calculate_confidence(results)

            # Record success
            execution_time = (time.time() - start_time) * 1000
            self.record_action(
                action_type=f"execute_{step.step_type.value}",
                parameters=step.parameters,
                result=results,
                success=True,
                duration_ms=execution_time
            )

            self.steps_executed += 1
            self.total_execution_time_ms += execution_time

            result = ExecutionResult(
                step_id=step.step_id,
                success=True,
                results=results,
                execution_time_ms=execution_time,
                confidence=confidence,
                metadata={"step_type": step.step_type.value}
            )

            logger.info(
                f"Step completed: {step.step_id} "
                f"({len(results)} results, {execution_time:.0f}ms)"
            )

            return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            logger.error(f"Step execution failed: {e}")

            # Retry logic
            if retry_count < self.max_retries and step.required:
                logger.info(f"Retrying step {step.step_id} (attempt {retry_count + 1})")
                await asyncio.sleep(self.retry_delay_ms / 1000)
                return await self.execute_step(step, retry_count + 1)

            # Record failure
            self.record_action(
                action_type=f"execute_{step.step_type.value}",
                parameters=step.parameters,
                result=None,
                success=False,
                duration_ms=execution_time,
                error_message=str(e)
            )

            self.steps_failed += 1

            return ExecutionResult(
                step_id=step.step_id,
                success=False,
                results=[],
                execution_time_ms=execution_time,
                error_message=str(e)
            )

        finally:
            self.update_state(AgentState.IDLE)

    async def execute_plan(
        self,
        steps: List[PlanStep]
    ) -> List[ExecutionResult]:
        """
        Execute multiple steps respecting dependencies.

        Args:
            steps: List of plan steps

        Returns:
            List of execution results
        """
        results: Dict[str, ExecutionResult] = {}
        completed_steps: Set[str] = set()

        while len(completed_steps) < len(steps):
            # Find steps ready to execute
            ready_steps = []
            for step in steps:
                if step.step_id in completed_steps:
                    continue

                dependencies_met = all(
                    dep in completed_steps for dep in step.dependencies
                )
                if dependencies_met:
                    ready_steps.append(step)

            if not ready_steps:
                logger.warning("No steps ready to execute, breaking")
                break

            # Execute ready steps
            step_results = await asyncio.gather(
                *[self.execute_step(step) for step in ready_steps],
                return_exceptions=True
            )

            # Record results
            for step, result in zip(ready_steps, step_results):
                if isinstance(result, Exception):
                    logger.error(f"Step {step.step_id} raised exception: {result}")
                    results[step.step_id] = ExecutionResult(
                        step_id=step.step_id,
                        success=False,
                        results=[],
                        execution_time_ms=0.0,
                        error_message=str(result)
                    )
                else:
                    results[step.step_id] = result

                completed_steps.add(step.step_id)

        return list(results.values())

    async def _execute_retrieval(
        self,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute retrieval operation."""
        query = parameters.get("query", "")
        strategy = parameters.get("strategy", "hybrid")
        top_k = parameters.get("top_k", 10)

        if not self.retrieval_pipeline:
            logger.warning("Retrieval pipeline not configured")
            return []

        # Simulate retrieval (in production would call actual pipeline)
        results = []
        for i in range(min(top_k, 5)):
            results.append({
                "content": f"Retrieved content {i} for query: {query}",
                "score": 0.9 - (i * 0.1),
                "metadata": {"source": f"doc_{i}", "strategy": strategy}
            })

        return results

    async def _execute_graph_query(
        self,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute graph query."""
        query = parameters.get("query", "")

        if not self.graph_queries:
            logger.warning("Graph queries not configured")
            return []

        # Simulate graph query
        results = [
            {
                "content": f"Graph context for: {query}",
                "relationships": ["related_to", "mentions"],
                "entities": ["entity1", "entity2"]
            }
        ]

        return results

    async def _execute_summarization(
        self,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute summarization."""
        content = parameters.get("content", [])

        # Simulate summarization
        summary = {
            "summary": f"Summary of {len(content)} items",
            "key_points": ["point1", "point2", "point3"]
        }

        return [summary]

    async def _execute_community_analysis(
        self,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute community analysis."""
        query = parameters.get("query", "")

        if not self.community_pipeline:
            logger.warning("Community pipeline not configured")
            return []

        # Simulate community analysis
        results = [
            {
                "community_id": "comm_1",
                "summary": f"Community summary for: {query}",
                "size": 10,
                "relevance": 0.85
            }
        ]

        return results

    async def _execute_raptor_query(
        self,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute RAPTOR hierarchical query."""
        query = parameters.get("query", "")

        if not self.raptor_pipeline:
            logger.warning("RAPTOR pipeline not configured")
            return []

        # Simulate RAPTOR query
        results = [
            {
                "level": "high",
                "content": f"High-level summary for: {query}",
                "score": 0.88
            },
            {
                "level": "mid",
                "content": f"Mid-level details for: {query}",
                "score": 0.75
            }
        ]

        return results

    async def _execute_reranking(
        self,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute result reranking."""
        query = parameters.get("query", "")
        results = parameters.get("results", [])

        # Simple reranking simulation
        # In production would use cross-encoder
        return sorted(
            results,
            key=lambda x: x.get("score", 0),
            reverse=True
        )[:10]

    async def _execute_synthesis(
        self,
        parameters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute final synthesis."""
        query = parameters.get("query", "")
        all_results = parameters.get("all_results", [])

        # Simulate synthesis
        synthesis = {
            "answer": f"Synthesized answer for: {query}",
            "sources": len(all_results),
            "confidence": 0.85
        }

        return [synthesis]

    def _calculate_confidence(
        self,
        results: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate confidence score for results.

        Args:
            results: Retrieved results

        Returns:
            Confidence score (0-1)
        """
        if not results:
            return 0.0

        # Average score from results
        scores = [r.get("score", 0.5) for r in results]
        avg_score = sum(scores) / len(scores)

        # Factor in result count
        count_factor = min(len(results) / 10, 1.0)

        return avg_score * 0.8 + count_factor * 0.2
