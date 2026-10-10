"""Analytic obstacle motion models.
Motion is a pure function of time, ``state_at(t)``. That gives us:
 * zero numerical drift, whatever the time step;
 * exact ground truth for any future time (needed to score the Phase 6
   trajectory predictor with ADE / FDE);
 * perfectly repeatable scenarios.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class MotionState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0


class Motion(Protocol):
    def state_at(self, t: float) -> MotionState: ...

    def shifted(self, dx: float, dy: float) -> Motion:
        """Return a copy translated by (dx, dy); used for seeded jitter."""
        ...


@dataclass(frozen=True)
class StaticMotion:
    x: float
    y: float

    def state_at(self, t: float) -> MotionState:
        return MotionState(self.x, self.y)

    def shifted(self, dx: float, dy: float) -> StaticMotion:
        return StaticMotion(self.x + dx, self.y + dy)


@dataclass(frozen=True)
class LinearMotion:
    """Constant velocity in a straight line (a worker crossing a corridor)."""

    x: float
    y: float
    vx: float
    vy: float

    def state_at(self, t: float) -> MotionState:
        t = max(t, 0.0)
        return MotionState(self.x + self.vx * t, self.y + self.vy * t, self.vx, self.vy)

    def shifted(self, dx: float, dy: float) -> LinearMotion:
        return LinearMotion(self.x + dx, self.y + dy, self.vx, self.vy)


class WaypointMotion:
    """Follow a polyline at constant speed (patrolling forklift, AGV route).
    mode = "loop" : after the last point, return to the first and repeat
           "pingpong" : go to the end, then walk the path backwards, repeat
           "once" : stop at the last point
    """

    MODES = ("loop", "pingpong", "once")

    def __init__(
        self,
        points: list[tuple[float, float]],
        speed: float,
        mode: str = "pingpong",
    ) -> None:
        if len(points) < 2:
            raise ValueError("WaypointMotion needs at least 2 points.")
        if speed <= 0:
            raise ValueError("WaypointMotion speed must be > 0.")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got '{mode}'.")
        self.points = [(float(x), float(y)) for x, y in points]
        self.speed = float(speed)
        self.mode = mode
        path = self.points + [self.points[0]] if mode == "loop" else self.points
        lengths = [math.dist(a, b) for a, b in zip(path[:-1], path[1:], strict=True)]
        if min(lengths) <= 0.0:
            raise ValueError("Consecutive waypoints must be distinct.")
        self._path = path
        self._cum = np.concatenate(([0.0], np.cumsum(lengths)))
        self._length = float(self._cum[-1])

    def state_at(self, t: float) -> MotionState:
        s = self.speed * max(t, 0.0)
        direction = 1.0
        if self.mode == "loop":
            s %= self._length
        elif self.mode == "pingpong":
            s %= 2.0 * self._length
            if s > self._length:
                s = 2.0 * self._length - s
                direction = -1.0
        elif s >= self._length:  # "once" and finished
            x, y = self._path[-1]
            return MotionState(x, y)

        idx = int(np.searchsorted(self._cum, s, side="right")) - 1
        idx = min(max(idx, 0), len(self._path) - 2)
        (x0, y0), (x1, y1) = self._path[idx], self._path[idx + 1]
        seg_len = float(self._cum[idx + 1] - self._cum[idx])
        ux, uy = (x1 - x0) / seg_len, (y1 - y0) / seg_len
        offset = s - float(self._cum[idx])
        return MotionState(
            x0 + ux * offset,
            y0 + uy * offset,
            direction * self.speed * ux,
            direction * self.speed * uy,
        )

    def shifted(self, dx: float, dy: float) -> WaypointMotion:
        moved = [(x + dx, y + dy) for x, y in self.points]
        return WaypointMotion(moved, self.speed, self.mode)
