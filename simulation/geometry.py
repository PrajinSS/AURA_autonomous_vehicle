"""Small, dependency-free geometry helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


def wrap_angle(angle: float) -> float:
    """Wrap an angle in radians to the interval [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clamp(value: float, low: float, high: float) -> float:
    """Limit ``value`` to the closed interval [low, high]."""
    return max(low, min(high, value))


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned rectangular map boundary."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("Bounds must have positive width and height.")

    def contains_circle(self, x: float, y: float, radius: float) -> bool:
        """True if a circle lies completely inside the boundary."""
        return (
            self.x_min + radius <= x <= self.x_max - radius
            and self.y_min + radius <= y <= self.y_max - radius
        )
