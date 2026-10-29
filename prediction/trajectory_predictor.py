"""Trajectory prediction engine for AURA.

Generates forward trajectory predictions and future bounding-box horizons
for actively tracked dynamic obstacles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from perception.tracker import TrackedObstacle


@dataclass
class TrajectoryPoint:
    t: float  # time relative to now (s)
    x: float  # ego frame X (m)
    y: float  # ego frame Y (m)
    vx: float
    vy: float
    uncertainty_radius: float


@dataclass
class PredictedTrajectory:
    track_id: str
    category: str
    waypoints: list[TrajectoryPoint]
    crosses_path: bool = False
    time_to_cross_s: float | None = None

    @property
    def final_point(self) -> TrajectoryPoint:
        return self.waypoints[-1]


class TrajectoryPredictor:
    """Predicts future motion using constant-velocity model with expanding uncertainty cones."""

    def __init__(
        self,
        horizon_s: float = 3.5,
        dt_step: float = 0.2,
        vehicle_corridor_half_width: float = 1.6,
    ) -> None:
        self.horizon = horizon_s
        self.dt_step = dt_step
        self.corridor_w = vehicle_corridor_half_width

    def predict(self, tracks: Sequence[TrackedObstacle]) -> list[PredictedTrajectory]:
        """Compute future trajectory waypoints for all tracked obstacles."""
        predictions = []
        num_steps = int(self.horizon / self.dt_step)

        for track in tracks:
            pts: list[TrajectoryPoint] = []
            crosses = False
            time_cross: float | None = None

            # Base speed
            vx = track.vx
            vy = track.vy
            speed = math.hypot(vx, vy)

            # Damp static obstacles
            if speed < 0.15:
                vx = 0.0
                vy = 0.0

            for i in range(1, num_steps + 1):
                t = i * self.dt_step
                px = track.x + vx * t
                py = track.y + vy * t
                # Uncertainty grows with prediction time: sigma_0 + 0.15 * t
                unc = 0.3 + 0.15 * t

                pts.append(
                    TrajectoryPoint(
                        t=round(t, 2),
                        x=round(px, 2),
                        y=round(py, 2),
                        vx=round(vx, 2),
                        vy=round(vy, 2),
                        uncertainty_radius=round(unc, 2),
                    )
                )

                # Check if enters vehicle forward corridor: px > 0 and |py| < corridor_w
                if not crosses and px > 0.5 and abs(py) < self.corridor_w:
                    crosses = True
                    time_cross = round(t, 2)

            predictions.append(
                PredictedTrajectory(
                    track_id=track.track_id,
                    category=track.category,
                    waypoints=pts,
                    crosses_path=crosses,
                    time_to_cross_s=time_cross,
                )
            )

        return predictions
