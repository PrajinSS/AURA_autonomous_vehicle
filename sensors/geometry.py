"""Geometry helpers for sensors. They read only ``WorldState`` truth records."""

from __future__ import annotations

import math

import numpy as np

from simulation.geometry import wrap_angle
from simulation.world_state import ObstacleTruth


def polar(
    ox: float, oy: float, heading: float, tx: float, ty: float
) -> tuple[float, float]:
    """Range and bearing (relative to ``heading``) from origin to target."""
    dx, dy = tx - ox, ty - oy
    return math.hypot(dx, dy), wrap_angle(math.atan2(dy, dx) - heading)


def polar_to_world(
    ox: float, oy: float, heading: float, rng: float, bearing: float
) -> tuple[float, float]:
    angle = heading + bearing
    return ox + rng * math.cos(angle), oy + rng * math.sin(angle)


def obstacle_distance(obstacle: ObstacleTruth, px, py):
    """Distance from point(s) to the obstacle surface (0 inside). Array-safe."""
    dx, dy = px - obstacle.x, py - obstacle.y
    if obstacle.shape == "circle":
        return np.maximum(np.hypot(dx, dy) - obstacle.width / 2.0, 0.0)
    out_x = np.abs(dx) - obstacle.width / 2.0
    out_y = np.abs(dy) - obstacle.height / 2.0
    return np.hypot(np.maximum(out_x, 0.0), np.maximum(out_y, 0.0))


def line_of_sight_blocked(
    ox: float,
    oy: float,
    target: ObstacleTruth,
    blockers: list[ObstacleTruth],
    step: float = 0.5,
) -> bool:
    """True if another obstacle sits on the straight line sensor -> target."""
    dist = math.hypot(target.x - ox, target.y - oy)
    extent = max(target.width, target.height) / 2.0  # stop before the target body
    usable = dist - extent
    if usable <= step or not blockers:
        return False
    along = np.arange(step, usable, step)
    px = ox + (target.x - ox) / dist * along
    py = oy + (target.y - oy) / dist * along
    return any(bool(np.any(obstacle_distance(b, px, py) <= 0.0)) for b in blockers)
