"""The simulated world: map bounds, goal, and all obstacles."""

from __future__ import annotations

import math

from simulation.geometry import Bounds
from simulation.obstacles import Obstacle


class Environment:
    """Owns the ground truth of the world and advances it in time."""

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

    def step(self, dt: float) -> None:
        for obstacle in self.obstacles:
            obstacle.step(dt, self.bounds)
        self.time += dt

    def obstacle_gaps(self, x: float, y: float, radius: float) -> list[float]:
        """Clearance (surface to surface) to every obstacle; <= 0 means contact."""
        return [
            math.hypot(o.x - x, o.y - y) - o.radius - radius for o in self.obstacles
        ]

    def min_gap(self, x: float, y: float, radius: float) -> float:
        gaps = self.obstacle_gaps(x, y, radius)
        return min(gaps) if gaps else math.inf

    def in_collision(self, x: float, y: float, radius: float) -> bool:
        return self.min_gap(x, y, radius) <= 0.0

    def goal_distance(self, x: float, y: float) -> float:
        return math.hypot(self.goal[0] - x, self.goal[1] - y)

    def goal_reached(self, x: float, y: float) -> bool:
        return self.goal_distance(x, y) <= self.goal_tolerance

    def ground_truth(self) -> list[dict[str, float | str]]:
        return [o.ground_truth() for o in self.obstacles]
