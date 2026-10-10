"""Deterministic simulation clock based on an integer tick counter."""

from __future__ import annotations


class SimulationClock:
    """Time = ticks * dt. Using an integer counter avoids float drift."""

    def __init__(self, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        self.dt = dt
        self.ticks = 0

    @property
    def time(self) -> float:
        return self.ticks * self.dt

    def tick(self) -> float:
        self.ticks += 1
        return self.time

    def reset(self) -> None:
        self.ticks = 0

    def steps_for(self, duration: float) -> int:
        return int(round(duration / self.dt))
