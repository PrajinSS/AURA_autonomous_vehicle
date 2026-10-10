"""The simulated world: map bounds, goal, and all obstacles."""

from __future__ import annotations

import math

from simulation.geometry import Bounds
from simulation.obstacles import Obstacle, ObstacleFrame


class Environment:
    """Owns the ground truth of the world and positions it for a given time."""

    def __init__(
        self,
        bounds: Bounds,
        goal: tuple[float, float],
        obstacles: list[Obstacle],
        goal_tolerance: float = 1.5,
    ) -> None:
        if goal_tolerance <= 0:
            raise ValueError("goal_tolerance must be > 0.")
        ids = [o.obstacle_id for o in obstacles]
        if len(ids) != len(set(ids)):
            raise ValueError("Obstacle ids must be unique.")
        self.bounds = bounds
        self.goal = goal
        self.goal_tolerance = goal_tolerance
        self.obstacles = obstacles
        self.time = 0.0
        self.update(0.0)

    # ---- time -------------------------------------------------------------
    def update(self, t: float) -> None:
        """Move every obstacle to its position at absolute time ``t``."""
        for obstacle in self.obstacles:
            obstacle.update(t)
        self.time = t

    # ---- queries ----------------------------------------------------------
    @property
    def static_obstacles(self) -> list[Obstacle]:
        return [o for o in self.obstacles if o.is_static]

    @property
    def dynamic_obstacles(self) -> list[Obstacle]:
        return [o for o in self.obstacles if not o.is_static]

    def obstacle_gaps(self, x: float, y: float, radius: float) -> list[float]:
        """Clearance (surface to surface) to every obstacle; <= 0 means contact."""
        return [o.gap(x, y, radius) for o in self.obstacles]

    def min_gap(self, x: float, y: float, radius: float) -> float:
        gaps = self.obstacle_gaps(x, y, radius)
        return min(gaps) if gaps else math.inf

    def in_collision(self, x: float, y: float, radius: float) -> bool:
        return self.min_gap(x, y, radius) <= 0.0

    def goal_distance(self, x: float, y: float) -> float:
        return math.hypot(self.goal[0] - x, self.goal[1] - y)

    def goal_reached(self, x: float, y: float) -> bool:
        return self.goal_distance(x, y) <= self.goal_tolerance

    def frames(self) -> tuple[ObstacleFrame, ...]:
        return tuple(o.frame() for o in self.obstacles)
