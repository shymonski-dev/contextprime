"""
Agent Coordinator for multi-agent orchestration.

The coordinator:
- Manages agent lifecycle
- Routes messages between agents
- Coordinates multi-agent workflows
- Resolves conflicts
- Builds consensus
"""

import asyncio
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from .base_agent import BaseAgent, AgentRole, AgentMessage, AgentState, MessagePriority


@dataclass
class CoordinationResult:
    """Result from coordinating multiple agents."""
    success: bool
    agent_results: Dict[str, Any] = field(default_factory=dict)
    consensus: Optional[Any] = None
    conflicts: List[str] = field(default_factory=list)
    coordination_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentCoordinator(BaseAgent):
    """
    Central coordinator for multi-agent system.

    Capabilities:
    - Agent lifecycle management
    - Message routing
    - Workflow orchestration
    - Conflict resolution
    - Consensus building
    """

    def __init__(self, agent_id: str = "coordinator"):
        """Initialize coordinator."""
        super().__init__(
            agent_id=agent_id,
            role=AgentRole.COORDINATOR,
            capabilities={
                "agent_management",
                "message_routing",
                "workflow_orchestration",
                "conflict_resolution",
                "consensus_building"
            }
        )

        # Agent registry
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_roles: Dict[AgentRole, List[str]] = {
            role: [] for role in AgentRole
        }

        # Message queue with priority
        self.message_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.message_handlers: Dict[str, asyncio.Task] = {}

    async def process_message(
        self,
        message: AgentMessage
    ) -> Optional[AgentMessage]:
        """Process incoming messages."""
        content = message.content
        action = content.get("action")

        if action == "coordinate_workflow":
            workflow_steps = content.get("steps", [])
            result = await self.coordinate_workflow(workflow_steps)

            return await self.send_message(
                recipient_id=message.sender_id,
                content={
                    "action": "workflow_complete",
                    "result": result.__dict__
                }
            )

        return None

    async def execute_action(
        self,
        action_type: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """Execute coordination actions."""
        if action_type == "coordinate":
            return await self.coordinate_workflow(parameters["steps"])
        elif action_type == "route_message":
            return await self.route_message(parameters["message"])
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent with the coordinator.

        Args:
            agent: Agent to register
        """
        self.agents[agent.agent_id] = agent
        self.agent_roles[agent.role].append(agent.agent_id)

        logger.info(
            f"Registered agent {agent.agent_id} with role {agent.role.value}"
        )

    def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent.

        Args:
            agent_id: Agent to unregister
        """
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            self.agent_roles[agent.role].remove(agent_id)
            del self.agents[agent_id]

            logger.info(f"Unregistered agent {agent_id}")

    async def route_message(
        self,
        message: AgentMessage
    ) -> bool:
        """
        Route a message to the appropriate agent.

        Args:
            message: Message to route

        Returns:
            True if routed successfully
        """
        recipient_id = message.recipient_id

        if recipient_id == "broadcast":
            # Broadcast to all agents
            for agent_id, agent in self.agents.items():
                if agent_id != message.sender_id:
                    await agent.receive_message(message)
            return True

        elif recipient_id.startswith("role:"):
            # Route to agents with specific role
            role_name = recipient_id.split(":")[1]
            try:
                role = AgentRole(role_name)
                agent_ids = self.agent_roles[role]
                for agent_id in agent_ids:
                    await self.agents[agent_id].receive_message(message)
                return True
            except (ValueError, KeyError):
                logger.error(f"Invalid role: {role_name}")
                return False

        elif recipient_id in self.agents:
            # Route to specific agent
            await self.agents[recipient_id].receive_message(message)
            return True

        else:
            logger.error(f"Unknown recipient: {recipient_id}")
            return False

    async def coordinate_workflow(
        self,
        workflow_steps: List[Dict[str, Any]]
    ) -> CoordinationResult:
        """
        Coordinate a multi-agent workflow.

        Args:
            workflow_steps: Steps defining the workflow

        Returns:
            Coordination result
        """
        import time
        start_time = time.time()
        self.update_state(AgentState.BUSY)

        logger.info(f"Coordinating workflow with {len(workflow_steps)} steps")

        try:
            agent_results = {}
            conflicts = []

            for step in workflow_steps:
                agent_role = step.get("agent_role")
                action = step.get("action")
                parameters = step.get("parameters", {})

                # Find agent with the role
                role = AgentRole(agent_role)
                agent_ids = self.agent_roles[role]

                if not agent_ids:
                    logger.warning(f"No agent found with role {agent_role}")
                    conflicts.append(f"Missing agent for role {agent_role}")
                    continue

                # Use first available agent (could implement load balancing)
                agent_id = agent_ids[0]
                agent = self.agents[agent_id]

                # Send message to agent
                message = await self.send_message(
                    recipient_id=agent_id,
                    content={
                        "action": action,
                        **parameters
                    },
                    requires_response=True
                )

                # Wait for response (simplified - in production use proper async)
                await asyncio.sleep(0.1)

                # Store result
                agent_results[agent_id] = {
                    "role": agent_role,
                    "action": action,
                    "status": "completed"
                }

            # Build consensus if multiple results
            consensus = None
            if len(agent_results) > 1:
                consensus = self._build_consensus(agent_results)

            coordination_time = (time.time() - start_time) * 1000

            result = CoordinationResult(
                success=len(conflicts) == 0,
                agent_results=agent_results,
                consensus=consensus,
                conflicts=conflicts,
                coordination_time_ms=coordination_time
            )

            logger.info(
                f"Workflow coordination complete: "
                f"{len(agent_results)} steps executed"
            )

            return result

        finally:
            self.update_state(AgentState.IDLE)

    def _build_consensus(
        self,
        agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build consensus from multiple agent results.

        Args:
            agent_results: Results from different agents

        Returns:
            Consensus result
        """
        # Simple voting/averaging approach
        # In production, would use more sophisticated methods

        consensus = {
            "method": "simple_aggregation",
            "participants": list(agent_results.keys()),
            "agreement_level": 1.0  # Simplified
        }

        return consensus

    async def broadcast_message(
        self,
        content: Dict[str, Any],
        exclude_agents: Optional[Set[str]] = None
    ) -> None:
        """
        Broadcast a message to all agents.

        Args:
            content: Message content
            exclude_agents: Agent IDs to exclude
        """
        exclude_agents = exclude_agents or set()

        for agent_id, agent in self.agents.items():
            if agent_id not in exclude_agents:
                message = AgentMessage(
                    sender_id=self.agent_id,
                    recipient_id=agent_id,
                    role=self.role,
                    content=content
                )
                await agent.receive_message(message)

    def get_agent_status_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all registered agents.

        Returns:
            Dictionary of agent statuses
        """
        return {
            agent_id: agent.get_status()
            for agent_id, agent in self.agents.items()
        }

    def get_agents_by_role(self, role: AgentRole) -> List[BaseAgent]:
        """
        Get all agents with a specific role.

        Args:
            role: Agent role to filter by

        Returns:
            List of agents with that role
        """
        agent_ids = self.agent_roles[role]
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]

    async def shutdown_all_agents(self) -> None:
        """Shutdown all registered agents."""
        logger.info("Shutting down all agents")

        for agent in self.agents.values():
            await agent.shutdown()

        self.agents.clear()
        self.agent_roles = {role: [] for role in AgentRole}
