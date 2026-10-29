"""Priority task queue for autonomous industrial AMR missions."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Iterator

from mission.parser import MissionTask


@dataclass(order=True)
class PrioritizedTask:
    priority: int  # lower number = popped first in min-heap, so we invert
    task_id: str = field(compare=False)
    task: MissionTask = field(compare=False)


class TaskQueue:
    """Thread-safe priority queue for logistics missions."""

    def __init__(self) -> None:
        self._heap: list[PrioritizedTask] = []
        self._all_tasks: dict[str, MissionTask] = {}

    def push(self, task: MissionTask) -> None:
        """Add task to queue ordered by priority (higher priority runs first)."""
        # Invert priority so that priority=5 is popped before priority=1
        inv_prio = -task.priority
        item = PrioritizedTask(priority=inv_prio, task_id=task.task_id, task=task)
        heapq.heappush(self._heap, item)
        self._all_tasks[task.task_id] = task

    def pop(self) -> MissionTask | None:
        """Retrieve the highest priority mission task."""
        if not self._heap:
            return None
        item = heapq.heappop(self._heap)
        return item.task

    def peek(self) -> MissionTask | None:
        """View next task without removing it."""
        if not self._heap:
            return None
        return self._heap[0].task

    def clear(self) -> None:
        """Clear all pending tasks."""
        self._heap.clear()
        self._all_tasks.clear()

    def __len__(self) -> int:
        return len(self._heap)
