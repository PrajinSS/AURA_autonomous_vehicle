"""Occupancy grid of the *static* world (input for the Phase 9 route planner)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simulation.environment import Environment


@dataclass(frozen=True, eq=False)
class OccupancyGrid:
    """data[row, col] is True when the cell is blocked. Row = y, col = x."""

    data: np.ndarray
    resolution: float
    x_min: float
    y_min: float

    @classmethod
    def from_environment(
        cls, env: Environment, resolution: float = 0.5, inflation: float = 0.0
    ) -> OccupancyGrid:
        """Rasterise static obstacles; ``inflation`` grows them by e.g. the
        vehicle radius so a planner can treat the vehicle as a single point."""
        if resolution <= 0 or inflation < 0:
            raise ValueError("resolution must be > 0 and inflation >= 0.")
        b = env.bounds
        cols = int(np.ceil((b.x_max - b.x_min) / resolution))
        rows = int(np.ceil((b.y_max - b.y_min) / resolution))
        xs = b.x_min + (np.arange(cols) + 0.5) * resolution
        ys = b.y_min + (np.arange(rows) + 0.5) * resolution
        grid_x, grid_y = np.meshgrid(xs, ys)
        blocked = np.zeros((rows, cols), dtype=bool)
        for obstacle in env.static_obstacles:
            blocked |= obstacle.distance(grid_x, grid_y) <= inflation
        return cls(blocked, resolution, b.x_min, b.y_min)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # (rows, cols)

    @property
    def occupied_fraction(self) -> float:
        return float(self.data.mean())

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        col = int(np.floor((x - self.x_min) / self.resolution))
        row = int(np.floor((y - self.y_min) / self.resolution))
        rows, cols = self.shape
        if 0 <= row < rows and 0 <= col < cols:
            return (row, col)
        return None

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.x_min + (col + 0.5) * self.resolution,
            self.y_min + (row + 0.5) * self.resolution,
        )

    def is_occupied(self, x: float, y: float) -> bool:
        """Points outside the map count as occupied (unsafe to enter)."""
        cell = self.world_to_cell(x, y)
        return True if cell is None else bool(self.data[cell])
