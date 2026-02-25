"""
Reinforcement Learning Module for continuous system improvement.

Implements:
- Reward functions
- Q-learning for discrete actions
- Policy learning
- Multi-armed bandits for exploration
"""

import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from loguru import logger


@dataclass
class RewardSignal:
    """Reward signal for RL."""
    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    done: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RLState:
    """State representation for RL."""
    query_complexity: str
    query_length: int
    available_strategies: List[str]
    previous_success: float
    context: Dict[str, Any] = field(default_factory=dict)

    def to_key(self) -> str:
        """Convert state to hashable key."""
        return f"{self.query_complexity}_{self.query_length}"


class RLModule:
    """
    Reinforcement Learning module for system optimization.

    Uses:
    - Q-learning for strategy selection
    - Multi-armed bandits for exploration
    - Reward shaping for multiple objectives
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize RL module.

        Args:
            learning_rate: Learning rate (alpha)
            discount_factor: Discount factor (gamma)
            epsilon: Exploration rate
            storage_path: Path to persist Q-table
        """
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.storage_path = storage_path or Path("data/rl_qtable.json")

        # Q-table: state-action values
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        # Action counts for exploration
        self.action_counts: Dict[str, int] = defaultdict(int)
        self.state_visits: Dict[str, int] = defaultdict(int)

        # Performance tracking
        self.episode_rewards: List[float] = []
        self.episode_count = 0

        # Load existing Q-table
        self._load_qtable()

    def select_action(
        self,
        state: RLState,
        available_actions: List[str]
    ) -> str:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            available_actions: List of possible actions

        Returns:
            Selected action
        """
        state_key = state.to_key()

        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.choice(available_actions)
            logger.debug(f"Exploring: selected random action {action}")
        else:
            # Exploit: best action
            q_values = {
                action: self.q_table[state_key][action]
                for action in available_actions
            }

            if not q_values:
                action = np.random.choice(available_actions)
            else:
                # Select action with highest Q-value
                max_q = max(q_values.values())
                best_actions = [
                    a for a, q in q_values.items() if q == max_q
                ]
                action = np.random.choice(best_actions)

            logger.debug(f"Exploiting: selected best action {action} (Q={max_q:.3f})")

        self.action_counts[action] += 1
        self.state_visits[state_key] += 1

        return action

    def update_q_value(
        self,
        reward_signal: RewardSignal
    ) -> None:
        """
        Update Q-value using Q-learning update rule.

        Q(s,a) = Q(s,a) + α * (r + γ * max Q(s',a') - Q(s,a))

        Args:
            reward_signal: Reward signal with state, action, reward
        """
        state_key = RLState(**reward_signal.state).to_key()
        next_state_key = RLState(**reward_signal.next_state).to_key()
        action = reward_signal.action
        reward = reward_signal.reward

        # Current Q-value
        current_q = self.q_table[state_key][action]

        # Max Q-value for next state
        if reward_signal.done:
            max_next_q = 0.0
        else:
            next_q_values = self.q_table[next_state_key]
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0

        # Q-learning update
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state_key][action] = new_q

        logger.debug(
            f"Updated Q({state_key}, {action}): {current_q:.3f} -> {new_q:.3f}"
        )

    def calculate_reward(
        self,
        quality_score: float,
        latency_ms: float,
        cost: float,
        user_satisfaction: Optional[float] = None
    ) -> float:
        """
        Calculate reward from multiple objectives.

        Args:
            quality_score: Result quality (0-1)
            latency_ms: Query latency
            cost: Query cost
            user_satisfaction: User feedback (0-1)

        Returns:
            Combined reward
        """
        # Quality reward (most important)
        quality_reward = quality_score * 10.0

        # Speed reward (faster is better)
        speed_reward = max(0, (5000 - latency_ms) / 5000) * 5.0

        # Cost penalty (lower cost is better)
        cost_penalty = -cost * 2.0

        # User satisfaction bonus
        satisfaction_bonus = (user_satisfaction or 0.5) * 3.0

        # Combined reward
        total_reward = (
            quality_reward * 0.5 +
            speed_reward * 0.2 +
            cost_penalty * 0.1 +
            satisfaction_bonus * 0.2
        )

        return total_reward

    def get_best_action(
        self,
        state: RLState,
        available_actions: List[str]
    ) -> Tuple[str, float]:
        """
        Get best action for state (pure exploitation).

        Args:
            state: Current state
            available_actions: Available actions

        Returns:
            Tuple of (best_action, q_value)
        """
        state_key = state.to_key()
        q_values = {
            action: self.q_table[state_key][action]
            for action in available_actions
        }

        if not q_values:
            return available_actions[0], 0.0

        best_action = max(q_values, key=q_values.get)
        best_q = q_values[best_action]

        return best_action, best_q

    def record_episode_reward(self, total_reward: float) -> None:
        """
        Record total reward for an episode.

        Args:
            total_reward: Total episode reward
        """
        self.episode_rewards.append(total_reward)
        self.episode_count += 1

        # Decay epsilon over time
        if self.episode_count % 10 == 0:
            self.epsilon = max(0.01, self.epsilon * 0.95)

    def get_statistics(self) -> Dict[str, Any]:
        """Get RL statistics."""
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else []

        return {
            "episode_count": self.episode_count,
            "q_table_size": len(self.q_table),
            "total_states": len(self.q_table),
            "total_actions": sum(len(actions) for actions in self.q_table.values()),
            "epsilon": self.epsilon,
            "average_reward_recent": (
                sum(recent_rewards) / len(recent_rewards)
                if recent_rewards else 0.0
            ),
            "total_reward": sum(self.episode_rewards),
            "action_distribution": dict(self.action_counts)
        }

    def _load_qtable(self) -> None:
        """Load Q-table from disk."""
        if not self.storage_path.exists():
            logger.info("No existing Q-table to load")
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Restore Q-table
            self.q_table = defaultdict(
                lambda: defaultdict(float),
                {
                    state: defaultdict(float, actions)
                    for state, actions in data.get("q_table", {}).items()
                }
            )

            # Restore metadata
            self.episode_count = data.get("episode_count", 0)
            self.epsilon = data.get("epsilon", self.epsilon)
            self.action_counts = defaultdict(
                int,
                data.get("action_counts", {})
            )

            logger.info(
                f"Loaded Q-table with {len(self.q_table)} states, "
                f"{self.episode_count} episodes"
            )

        except Exception as e:
            logger.error(f"Error loading Q-table: {e}")

    def save_qtable(self) -> None:
        """Save Q-table to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "q_table": {
                    state: dict(actions)
                    for state, actions in self.q_table.items()
                },
                "episode_count": self.episode_count,
                "epsilon": self.epsilon,
                "action_counts": dict(self.action_counts),
                "statistics": self.get_statistics()
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved Q-table with {len(self.q_table)} states")

        except Exception as e:
            logger.error(f"Error saving Q-table: {e}")


class MultiArmedBandit:
    """
    Multi-armed bandit for exploration-exploitation.

    Uses Upper Confidence Bound (UCB) algorithm.
    """

    def __init__(self, exploration_factor: float = 2.0):
        """
        Initialize bandit.

        Args:
            exploration_factor: UCB exploration parameter (c)
        """
        self.exploration_factor = exploration_factor

        # Arm statistics
        self.arm_rewards: Dict[str, List[float]] = defaultdict(list)
        self.arm_counts: Dict[str, int] = defaultdict(int)
        self.total_pulls = 0

    def select_arm(self, available_arms: List[str]) -> str:
        """
        Select arm using UCB algorithm.

        Args:
            available_arms: List of available arms

        Returns:
            Selected arm
        """
        # Pull each arm at least once
        for arm in available_arms:
            if self.arm_counts[arm] == 0:
                return arm

        # Calculate UCB for each arm
        ucb_values = {}
        for arm in available_arms:
            avg_reward = np.mean(self.arm_rewards[arm])
            exploration_bonus = self.exploration_factor * np.sqrt(
                np.log(self.total_pulls) / self.arm_counts[arm]
            )
            ucb_values[arm] = avg_reward + exploration_bonus

        # Select arm with highest UCB
        best_arm = max(ucb_values, key=ucb_values.get)
        return best_arm

    def update(self, arm: str, reward: float) -> None:
        """
        Update arm statistics.

        Args:
            arm: Arm that was pulled
            reward: Reward received
        """
        self.arm_rewards[arm].append(reward)
        self.arm_counts[arm] += 1
        self.total_pulls += 1

    def get_best_arm(self) -> Tuple[str, float]:
        """
        Get current best arm.

        Returns:
            Tuple of (best_arm, average_reward)
        """
        avg_rewards = {
            arm: np.mean(rewards)
            for arm, rewards in self.arm_rewards.items()
        }

        if not avg_rewards:
            return "", 0.0

        best_arm = max(avg_rewards, key=avg_rewards.get)
        return best_arm, avg_rewards[best_arm]
