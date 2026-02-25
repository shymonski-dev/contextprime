"""
Memory System for agent persistence and knowledge retention.

Implements:
- Short-term memory (current session)
- Long-term memory (persistent storage)
- Episodic memory (complete interactions)
- Memory management (forgetting, consolidation)
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from loguru import logger


@dataclass
class MemoryEntry:
    """Single memory entry."""
    entry_id: str
    memory_type: str  # short, long, episodic
    content: Dict[str, Any]
    importance: float = 0.5
    access_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat()
        }


class ShortTermMemory:
    """
    Short-term working memory for current session.

    Holds recent context with limited capacity.
    """

    def __init__(self, capacity: int = 50):
        """
        Initialize short-term memory.

        Args:
            capacity: Maximum number of entries
        """
        self.capacity = capacity
        self.memory: deque = deque(maxlen=capacity)

    def add(self, content: Dict[str, Any], importance: float = 0.5) -> None:
        """
        Add entry to short-term memory.

        Args:
            content: Memory content
            importance: Importance score
        """
        entry = MemoryEntry(
            entry_id=f"stm_{len(self.memory)}",
            memory_type="short",
            content=content,
            importance=importance
        )
        self.memory.append(entry)

    def get_recent(self, n: int = 10) -> List[MemoryEntry]:
        """
        Get n most recent entries.

        Args:
            n: Number of entries

        Returns:
            List of recent entries
        """
        return list(self.memory)[-n:]

    def search(self, query: str) -> List[MemoryEntry]:
        """
        Search short-term memory.

        Args:
            query: Search query

        Returns:
            Matching entries
        """
        query_lower = query.lower()
        matches = []

        for entry in self.memory:
            content_str = str(entry.content).lower()
            if query_lower in content_str:
                matches.append(entry)

        return matches

    def clear(self) -> None:
        """Clear all short-term memory."""
        self.memory.clear()

    def size(self) -> int:
        """Get current memory size."""
        return len(self.memory)


class LongTermMemory:
    """
    Long-term persistent memory.

    Stores important information that survives sessions.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize long-term memory.

        Args:
            storage_path: Path to persist memory
        """
        self.storage_path = storage_path or Path("data/long_term_memory.json")
        self.memory: Dict[str, MemoryEntry] = {}
        self._load()

    def add(
        self,
        entry_id: str,
        content: Dict[str, Any],
        importance: float = 0.5
    ) -> None:
        """
        Add entry to long-term memory.

        Args:
            entry_id: Unique entry ID
            content: Memory content
            importance: Importance score
        """
        entry = MemoryEntry(
            entry_id=entry_id,
            memory_type="long",
            content=content,
            importance=importance
        )
        self.memory[entry_id] = entry

        # Persist if high importance
        if importance >= 0.7:
            self._save()

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve entry by ID.

        Args:
            entry_id: Entry ID

        Returns:
            Memory entry if found
        """
        entry = self.memory.get(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now()
        return entry

    def search(
        self,
        query: str,
        min_importance: float = 0.0
    ) -> List[MemoryEntry]:
        """
        Search long-term memory.

        Args:
            query: Search query
            min_importance: Minimum importance threshold

        Returns:
            Matching entries
        """
        query_lower = query.lower()
        matches = []

        for entry in self.memory.values():
            if entry.importance < min_importance:
                continue

            content_str = str(entry.content).lower()
            if query_lower in content_str:
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                matches.append(entry)

        return matches

    def consolidate(self, min_importance: float = 0.3) -> int:
        """
        Consolidate memory by removing low-importance entries.

        Args:
            min_importance: Minimum importance to keep

        Returns:
            Number of entries removed
        """
        before_count = len(self.memory)

        # Remove low-importance, infrequently accessed entries
        to_remove = []
        for entry_id, entry in self.memory.items():
            if (entry.importance < min_importance and
                entry.access_count < 2):
                to_remove.append(entry_id)

        for entry_id in to_remove:
            del self.memory[entry_id]

        removed_count = before_count - len(self.memory)
        if removed_count > 0:
            self._save()
            logger.info(f"Consolidated memory: removed {removed_count} entries")

        return removed_count

    def _load(self) -> None:
        """Load memory from disk."""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            for entry_data in data.get("entries", []):
                entry = MemoryEntry(
                    entry_id=entry_data["entry_id"],
                    memory_type=entry_data["memory_type"],
                    content=entry_data["content"],
                    importance=entry_data["importance"],
                    access_count=entry_data.get("access_count", 0),
                    created_at=datetime.fromisoformat(entry_data["created_at"]),
                    last_accessed=datetime.fromisoformat(entry_data["last_accessed"]),
                    metadata=entry_data.get("metadata", {})
                )
                self.memory[entry.entry_id] = entry

            logger.info(f"Loaded {len(self.memory)} long-term memories")

        except Exception as e:
            logger.error(f"Error loading long-term memory: {e}")

    def _save(self) -> None:
        """Save memory to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "entries": [
                    entry.to_dict()
                    for entry in self.memory.values()
                ]
            }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving long-term memory: {e}")


class MemorySystem:
    """
    Complete memory system integrating short and long-term memory.
    """

    def __init__(
        self,
        short_term_capacity: int = 50,
        storage_path: Optional[Path] = None
    ):
        """
        Initialize memory system.

        Args:
            short_term_capacity: Capacity for short-term memory
            storage_path: Path for long-term storage
        """
        self.short_term = ShortTermMemory(capacity=short_term_capacity)
        self.long_term = LongTermMemory(storage_path=storage_path)

        # Episodic memory (complete interactions)
        self.episodes: List[Dict[str, Any]] = []

    def remember(
        self,
        content: Dict[str, Any],
        importance: float = 0.5,
        persist: bool = False
    ) -> str:
        """
        Store a memory.

        Args:
            content: Memory content
            importance: Importance score
            persist: Whether to persist to long-term memory

        Returns:
            Memory entry ID
        """
        # Always add to short-term
        self.short_term.add(content, importance)

        # Add to long-term if important or explicitly requested
        if persist or importance >= 0.7:
            entry_id = f"ltm_{len(self.long_term.memory)}"
            self.long_term.add(entry_id, content, importance)
            return entry_id

        return f"stm_{self.short_term.size() - 1}"

    def recall(
        self,
        query: str,
        use_long_term: bool = True
    ) -> List[MemoryEntry]:
        """
        Recall memories matching query.

        Args:
            query: Search query
            use_long_term: Whether to search long-term memory

        Returns:
            Matching memory entries
        """
        results = []

        # Search short-term memory
        results.extend(self.short_term.search(query))

        # Search long-term memory
        if use_long_term:
            results.extend(self.long_term.search(query))

        # Sort by importance and recency
        results.sort(
            key=lambda e: (e.importance, e.last_accessed),
            reverse=True
        )

        return results

    def remember_episode(
        self,
        query: str,
        plan: Dict[str, Any],
        results: List[Dict[str, Any]],
        assessment: Dict[str, Any]
    ) -> None:
        """
        Store a complete query episode.

        Args:
            query: Original query
            plan: Execution plan
            results: Results
            assessment: Quality assessment
        """
        episode = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "plan": plan,
            "results": results,
            "assessment": assessment
        }

        self.episodes.append(episode)

        # Store in long-term if successful
        if assessment.get("overall_score", 0) >= 0.8:
            self.long_term.add(
                entry_id=f"episode_{len(self.episodes)}",
                content=episode,
                importance=0.8
            )

    def consolidate_memories(self) -> Dict[str, int]:
        """
        Consolidate memories by promoting important short-term memories
        and forgetting low-value long-term memories.

        Returns:
            Consolidation statistics
        """
        # Promote important recent memories to long-term
        recent_memories = self.short_term.get_recent(10)
        promoted = 0

        for memory in recent_memories:
            if memory.importance >= 0.7:
                self.long_term.add(
                    entry_id=f"promoted_{len(self.long_term.memory)}",
                    content=memory.content,
                    importance=memory.importance
                )
                promoted += 1

        # Consolidate long-term memory
        removed = self.long_term.consolidate()

        return {
            "promoted_to_long_term": promoted,
            "removed_from_long_term": removed,
            "short_term_size": self.short_term.size(),
            "long_term_size": len(self.long_term.memory)
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            "short_term_size": self.short_term.size(),
            "short_term_capacity": self.short_term.capacity,
            "long_term_size": len(self.long_term.memory),
            "episodes_count": len(self.episodes),
            "total_memories": (
                self.short_term.size() + len(self.long_term.memory)
            )
        }
