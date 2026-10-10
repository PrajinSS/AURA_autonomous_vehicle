"""World obstacles = a shape + a motion model + a category label."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simulation.motion import LinearMotion, Motion, StaticMotion
from simulation.shapes import Circle, Shape


@dataclass(frozen=True)
class ObstacleFrame:
    """Lightweight pose snapshot used for logging and plotting."""

    obstacle_id: str
    kind: str  # "circle" or "rectangle"
    x: float
    y: float
    width: float
    height: float


@dataclass
class Obstacle:
    obstacle_id: str
    shape: Shape
    motion: Motion
    category: str = "static"  # free label: static / worker / vehicle ...
    x: float = field(init=False, default=0.0)
    y: float = field(init=False, default=0.0)
    vx: float = field(init=False, default=0.0)
    vy: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.update(0.0)

    # ---- construction helpers -------------------------------------------
    @classmethod
    def static(cls, obstacle_id: str, x: float, y: float, radius: float) -> Obstacle:
        return cls(obstacle_id, Circle(radius), StaticMotion(x, y), "static")

    @classmethod
    def moving(
        cls,
        obstacle_id: str,
        x: float,
        y: float,
        radius: float,
        vx: float,
        vy: float,
    ) -> Obstacle:
        return cls(obstacle_id, Circle(radius), LinearMotion(x, y, vx, vy), "worker")

    # ---- state ------------------------------------------------------------
    @property
    def is_static(self) -> bool:
        return isinstance(self.motion, StaticMotion)

    def update(self, t: float) -> None:
        """Place the obstacle where its motion model says it is at time ``t``."""
        state = self.motion.state_at(t)
        self.x, self.y, self.vx, self.vy = state.x, state.y, state.vx, state.vy

    # ---- geometry ---------------------------------------------------------
    def closest_point(self, px: float, py: float) -> tuple[float, float]:
        rx, ry = self.shape.closest_point_rel(px - self.x, py - self.y)
        return (self.x + rx, self.y + ry)

    def distance(self, px, py):
        """Distance from a point (or arrays of points) to the obstacle surface."""
        return self.shape.distance_rel(px - self.x, py - self.y)

    def gap(self, px: float, py: float, radius: float) -> float:
        """Surface-to-surface clearance for a circular body; <= 0 is contact."""
        return float(self.distance(px, py)) - radius

    # ---- ground truth -----------------------------------------------------
    def frame(self) -> ObstacleFrame:
        w, h = self.shape.extent
        return ObstacleFrame(self.obstacle_id, self.shape.kind, self.x, self.y, w, h)

    def future_positions(self, t0: float, horizon: float, step: float) -> np.ndarray:
        """True centre positions at t0+step, t0+2*step, ... t0+horizon (n, 2)."""
        times = t0 + step * np.arange(1, int(round(horizon / step)) + 1)
        return np.array([[s.x, s.y] for s in map(self.motion.state_at, times)])
