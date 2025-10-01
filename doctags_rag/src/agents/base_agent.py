"""
Base agent classes and communication protocols.

Provides the foundational architecture for all agents in the system:
- State management
- Action history tracking
- Goal management
- Inter-agent communication
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger


class AgentRole(Enum):
    """Roles that agents can take in the system."""
    PLANNER = "planner"
    EXECUTOR = "executor"
    EVALUATOR = "evaluator"
    LEARNER = "learner"
    COORDINATOR = "coordinator"


class MessagePriority(Enum):
    """Priority levels for agent messages."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class AgentState(Enum):
    """Possible states for an agent."""
    IDLE = "idle"
    BUSY = "busy"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class AgentMessage:
    """
    Structured message for inter-agent communication.

    Messages follow a protocol that ensures:
    - Traceable communication
    - Priority handling
    - Response tracking
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str = ""
    role: AgentRole = AgentRole.EXECUTOR
    priority: MessagePriority = MessagePriority.NORMAL
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    requires_response: bool = False
    parent_message_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "role": self.role.value,
            "priority": self.priority.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "requires_response": self.requires_response,
            "parent_message_id": self.parent_message_id,
            "metadata": self.metadata
        }


@dataclass
class Action:
    """Record of an action taken by an agent."""
    action_type: str
    parameters: Dict[str, Any]
    result: Any
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    error_message: Optional[str] = None


@dataclass
class Goal:
    """Agent goal with tracking information."""
    description: str
    goal_type: str
    priority: int = 1
    status: str = "active"  # active, completed, failed, abandoned
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base class for all agents in the system.

    Provides:
    - State management
    - Action history
    - Goal tracking
    - Communication interface
    - Logging and monitoring
    """

    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        capabilities: Optional[Set[str]] = None
    ):
        """
        Initialize base agent.

        Args:
            agent_id: Unique identifier for this agent
            role: Role of this agent
            capabilities: Set of capabilities this agent has
        """
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities or set()

        # State management
        self.state = AgentState.IDLE
        self.created_at = datetime.now()
        self.last_activity = datetime.now()

        # Action and goal tracking
        self.action_history: List[Action] = []
        self.goals: List[Goal] = []
        self.current_goal: Optional[Goal] = None

        # Communication
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.sent_messages: List[AgentMessage] = []
        self.received_messages: List[AgentMessage] = []

        # Performance metrics
        self.metrics = {
            "actions_completed": 0,
            "actions_failed": 0,
            "goals_completed": 0,
            "goals_failed": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "total_processing_time_ms": 0.0,
            "average_action_time_ms": 0.0
        }

        logger.info(
            f"Agent {self.agent_id} initialized with role {self.role.value}"
        )

    @abstractmethod
    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process an incoming message.

        Args:
            message: Message to process

        Returns:
            Optional response message
        """
        pass

    @abstractmethod
    async def execute_action(
        self,
        action_type: str,
        parameters: Dict[str, Any]
    ) -> Any:
        """
        Execute a specific action.

        Args:
            action_type: Type of action to execute
            parameters: Action parameters

        Returns:
            Action result
        """
        pass

    async def send_message(
        self,
        recipient_id: str,
        content: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        requires_response: bool = False
    ) -> AgentMessage:
        """
        Send a message to another agent.

        Args:
            recipient_id: ID of recipient agent
            content: Message content
            priority: Message priority
            requires_response: Whether response is required

        Returns:
            The sent message
        """
        message = AgentMessage(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            role=self.role,
            priority=priority,
            content=content,
            requires_response=requires_response
        )

        self.sent_messages.append(message)
        self.metrics["messages_sent"] += 1
        self.last_activity = datetime.now()

        logger.debug(
            f"Agent {self.agent_id} sent message to {recipient_id} "
            f"(priority: {priority.value})"
        )

        return message

    async def receive_message(self, message: AgentMessage) -> None:
        """
        Receive a message into the inbox.

        Args:
            message: Message to receive
        """
        await self.inbox.put(message)
        self.received_messages.append(message)
        self.metrics["messages_received"] += 1
        self.last_activity = datetime.now()

        logger.debug(
            f"Agent {self.agent_id} received message from {message.sender_id}"
        )

    async def process_inbox(self) -> List[AgentMessage]:
        """
        Process all messages in inbox.

        Returns:
            List of response messages
        """
        responses = []

        while not self.inbox.empty():
            try:
                message = await self.inbox.get()
                response = await self.process_message(message)
                if response:
                    responses.append(response)
            except Exception as e:
                logger.error(
                    f"Agent {self.agent_id} error processing message: {e}"
                )

        return responses

    def record_action(
        self,
        action_type: str,
        parameters: Dict[str, Any],
        result: Any,
        success: bool,
        duration_ms: float,
        error_message: Optional[str] = None
    ) -> None:
        """
        Record an action in the history.

        Args:
            action_type: Type of action
            parameters: Action parameters
            result: Action result
            success: Whether action succeeded
            duration_ms: Duration in milliseconds
            error_message: Error message if failed
        """
        action = Action(
            action_type=action_type,
            parameters=parameters,
            result=result,
            success=success,
            duration_ms=duration_ms,
            error_message=error_message
        )

        self.action_history.append(action)
        self.last_activity = datetime.now()

        if success:
            self.metrics["actions_completed"] += 1
        else:
            self.metrics["actions_failed"] += 1

        # Update average action time
        self.metrics["total_processing_time_ms"] += duration_ms
        total_actions = (
            self.metrics["actions_completed"] + self.metrics["actions_failed"]
        )
        if total_actions > 0:
            self.metrics["average_action_time_ms"] = (
                self.metrics["total_processing_time_ms"] / total_actions
            )

    def add_goal(
        self,
        description: str,
        goal_type: str,
        priority: int = 1,
        success_criteria: Optional[Dict[str, Any]] = None
    ) -> Goal:
        """
        Add a new goal.

        Args:
            description: Goal description
            goal_type: Type of goal
            priority: Goal priority
            success_criteria: Success criteria dictionary

        Returns:
            The created goal
        """
        goal = Goal(
            description=description,
            goal_type=goal_type,
            priority=priority,
            success_criteria=success_criteria or {}
        )

        self.goals.append(goal)

        # Set as current goal if higher priority
        if not self.current_goal or goal.priority > self.current_goal.priority:
            self.current_goal = goal

        logger.info(
            f"Agent {self.agent_id} added goal: {description} "
            f"(priority: {priority})"
        )

        return goal

    def complete_goal(self, goal: Goal, success: bool = True) -> None:
        """
        Mark a goal as completed.

        Args:
            goal: Goal to complete
            success: Whether goal was successful
        """
        goal.status = "completed" if success else "failed"
        goal.completed_at = datetime.now()
        goal.progress = 1.0

        if success:
            self.metrics["goals_completed"] += 1
        else:
            self.metrics["goals_failed"] += 1

        # Set next goal if this was current
        if self.current_goal == goal:
            self.current_goal = self._get_next_goal()

        logger.info(
            f"Agent {self.agent_id} completed goal: {goal.description} "
            f"(success: {success})"
        )

    def _get_next_goal(self) -> Optional[Goal]:
        """Get the next active goal by priority."""
        active_goals = [g for g in self.goals if g.status == "active"]
        if not active_goals:
            return None
        return max(active_goals, key=lambda g: g.priority)

    def update_state(self, new_state: AgentState) -> None:
        """
        Update agent state.

        Args:
            new_state: New state
        """
        old_state = self.state
        self.state = new_state
        self.last_activity = datetime.now()

        logger.debug(
            f"Agent {self.agent_id} state changed: {old_state.value} -> "
            f"{new_state.value}"
        )

    def has_capability(self, capability: str) -> bool:
        """
        Check if agent has a capability.

        Args:
            capability: Capability to check

        Returns:
            True if agent has capability
        """
        return capability in self.capabilities

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive agent status.

        Returns:
            Status dictionary
        """
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "state": self.state.value,
            "capabilities": list(self.capabilities),
            "current_goal": (
                self.current_goal.description if self.current_goal else None
            ),
            "active_goals": len([g for g in self.goals if g.status == "active"]),
            "inbox_size": self.inbox.qsize(),
            "metrics": self.metrics,
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds(),
            "last_activity": self.last_activity.isoformat()
        }

    def get_action_history(
        self,
        limit: Optional[int] = None,
        action_type: Optional[str] = None
    ) -> List[Action]:
        """
        Get action history with optional filtering.

        Args:
            limit: Maximum number of actions to return
            action_type: Filter by action type

        Returns:
            List of actions
        """
        actions = self.action_history

        if action_type:
            actions = [a for a in actions if a.action_type == action_type]

        if limit:
            actions = actions[-limit:]

        return actions

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = {
            "actions_completed": 0,
            "actions_failed": 0,
            "goals_completed": 0,
            "goals_failed": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "total_processing_time_ms": 0.0,
            "average_action_time_ms": 0.0
        }
        logger.info(f"Agent {self.agent_id} metrics reset")

    async def shutdown(self) -> None:
        """Shutdown the agent gracefully."""
        self.update_state(AgentState.TERMINATED)
        logger.info(f"Agent {self.agent_id} shutting down")
