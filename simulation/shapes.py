"""Obstacle footprints. Shapes are defined around their own centre (0, 0).
All distance methods accept scalars *or* NumPy arrays, so the same code
serves single collision checks and whole-grid occupancy computation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Shape(Protocol):
    kind: str

    @property
    def extent(self) -> tuple[float, float]:
        """Full (width, height) of the bounding box."""
        ...

    def closest_point_rel(self, dx: float, dy: float) -> tuple[float, float]:
        """Closest point on/in the shape to a point given relative to the centre."""
        ...

    def distance_rel(self, dx, dy):
        """Distance from a relative point to the shape; 0 if inside."""
        ...


@dataclass(frozen=True)
class Circle:
    radius: float
    kind: str = "circle"

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError("Circle radius must be > 0.")

    @property
    def extent(self) -> tuple[float, float]:
        return (2.0 * self.radius, 2.0 * self.radius)

    def closest_point_rel(self, dx: float, dy: float) -> tuple[float, float]:
        dist = math.hypot(dx, dy)
        if dist <= self.radius:
            return (dx, dy)
        scale = self.radius / dist
        return (dx * scale, dy * scale)

    def distance_rel(self, dx, dy):
        return np.maximum(np.hypot(dx, dy) - self.radius, 0.0)


@dataclass(frozen=True)
class Rectangle:
    """Axis-aligned rectangle (shelves, walls, machines, AGV bodies)."""

    width: float
    height: float
    kind: str = "rectangle"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Rectangle width and height must be > 0.")

    @property
    def extent(self) -> tuple[float, float]:
        return (self.width, self.height)

    def closest_point_rel(self, dx: float, dy: float) -> tuple[float, float]:
        hw, hh = self.width / 2.0, self.height / 2.0
        return (min(max(dx, -hw), hw), min(max(dy, -hh), hh))

    def distance_rel(self, dx, dy):
        hw, hh = self.width / 2.0, self.height / 2.0
        outside_x = np.abs(dx) - hw
        outside_y = np.abs(dy) - hh
        return np.hypot(np.maximum(outside_x, 0.0), np.maximum(outside_y, 0.0))
