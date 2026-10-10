"""Static and moving obstacles (ground-truth objects in the world)."""

from __future__ import annotations

from dataclasses import dataclass

from simulation.geometry import Bounds


@dataclass
class Obstacle:
    """A circular obstacle with constant velocity.

    Static obstacles simply have zero velocity. Moving obstacles (workers,
    other vehicles) bounce off the map boundary so they never leave the world.
    """

    obstacle_id: str
    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("Obstacle radius must be > 0.")

    @classmethod
    def static(cls, obstacle_id: str, x: float, y: float, radius: float) -> Obstacle:
        return cls(obstacle_id, x, y, radius)

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
        return cls(obstacle_id, x, y, radius, vx, vy)

    @property
    def is_static(self) -> bool:
        return self.vx == 0.0 and self.vy == 0.0

    def step(self, dt: float, bounds: Bounds) -> None:
        """Advance one time step with constant velocity and wall reflection."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.x - self.radius < bounds.x_min:
            self.x = bounds.x_min + self.radius
            self.vx = abs(self.vx)
        elif self.x + self.radius > bounds.x_max:
            self.x = bounds.x_max - self.radius
            self.vx = -abs(self.vx)
        if self.y - self.radius < bounds.y_min:
            self.y = bounds.y_min + self.radius
            self.vy = abs(self.vy)
        elif self.y + self.radius > bounds.y_max:
            self.y = bounds.y_max - self.radius
            self.vy = -abs(self.vy)

    def ground_truth(self) -> dict[str, float | str]:
        """Exact state; Phase 3 sensors will corrupt this with noise/faults."""
        return {
            "id": self.obstacle_id,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "radius": self.radius,
        }
