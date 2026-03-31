"""Multi-agent memory pool with priority-based eviction.

Manages compressed KV caches for multiple concurrent agents within a
shared memory budget. When the pool is full, the lowest-priority,
least-recently-used agent is evicted.
"""

import time
from dataclasses import dataclass, field

import numpy as np

from turbomoq.compressor import TurboMOQCompressor, CompressedMOQCache

__all__ = ["MemoryPool", "AgentSlot", "PoolStats"]


@dataclass
class AgentSlot:
    """Memory slot for a single agent."""
    agent_id: str
    agent_type: str
    cache: CompressedMOQCache | None = None
    priority: float = 1.0
    last_access: float = field(default_factory=time.monotonic)
    access_count: int = 0

    def touch(self):
        self.last_access = time.monotonic()
        self.access_count += 1

    @property
    def memory_bytes(self) -> int:
        return self.cache.memory_bytes if self.cache else 0


@dataclass
class PoolStats:
    """Aggregate pool statistics."""
    total_agents: int = 0
    total_memory_bytes: int = 0
    budget_bytes: int = 0
    utilization: float = 0.0

    def to_dict(self) -> dict:
        return {
            "agents": self.total_agents,
            "memory_mb": round(self.total_memory_bytes / 1e6, 1),
            "budget_mb": round(self.budget_bytes / 1e6, 1),
            "utilization_pct": round(self.utilization * 100, 1),
        }


class MemoryPool:
    """Multi-agent compressed KV cache pool.

    Usage:
        pool = MemoryPool(budget_mb=8192)
        pool.allocate("agent-1", "research", priority=1.0)
        pool.store("agent-1", compressed_cache)
        cache = pool.get("agent-1")
        pool.evict_if_needed()
    """

    def __init__(self, budget_mb: int = 4096):
        self.budget_bytes = budget_mb * 1024 * 1024
        self._slots: dict[str, AgentSlot] = {}

    def allocate(self, agent_id: str, agent_type: str = "research",
                 priority: float = 1.0) -> AgentSlot:
        if agent_id in self._slots:
            self._slots[agent_id].touch()
            return self._slots[agent_id]
        slot = AgentSlot(agent_id=agent_id, agent_type=agent_type, priority=priority)
        self._slots[agent_id] = slot
        return slot

    def store(self, agent_id: str, cache: CompressedMOQCache) -> None:
        slot = self._slots.get(agent_id)
        if slot is None:
            slot = self.allocate(agent_id)
        slot.cache = cache
        slot.touch()

    def get(self, agent_id: str) -> CompressedMOQCache | None:
        slot = self._slots.get(agent_id)
        if slot:
            slot.touch()
            return slot.cache
        return None

    def release(self, agent_id: str) -> bool:
        return self._slots.pop(agent_id, None) is not None

    def evict_if_needed(self) -> list[str]:
        evicted = []
        while self._total_memory() > self.budget_bytes and self._slots:
            victim = min(self._slots.values(),
                         key=lambda s: (s.priority, -s.last_access))
            evicted.append(victim.agent_id)
            self._slots.pop(victim.agent_id)
        return evicted

    def _total_memory(self) -> int:
        return sum(s.memory_bytes for s in self._slots.values())

    @property
    def stats(self) -> PoolStats:
        total = self._total_memory()
        return PoolStats(
            total_agents=len(self._slots),
            total_memory_bytes=total,
            budget_bytes=self.budget_bytes,
            utilization=total / self.budget_bytes if self.budget_bytes > 0 else 0,
        )
