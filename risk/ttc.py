"""Analytical Time-to-Collision (TTC) calculator for AURA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from perception.tracker import TrackedObstacle


@dataclass
class TTCEstimate:
    obstacle_id: str
    ttc_sec: float  # Time to collision in seconds (inf if divergent)
    min_distance_m: float  # Predicted closest point of approach (CPA)
    time_at_cpa_s: float
    is_imminent: bool  # ttc < 2.5s


class TTCCalculator:
    """Computes exact time to collision using relative kinematics."""

    def __init__(self, ego_radius: float = 1.3, default_ttc_threshold: float = 3.0) -> None:
        self.ego_radius = ego_radius
        self.threshold = default_ttc_threshold

    def compute_ttc(
        self,
        obstacle: TrackedObstacle,
        ego_speed: float,
    ) -> TTCEstimate:
        """Calculate TTC between ego vehicle moving along +X and obstacle."""
        rx = obstacle.x
        ry = obstacle.y

        # Obstacles behind the vehicle are not an imminent forward collision threat
        if rx < -0.5:
            return TTCEstimate(obstacle.track_id, math.inf, math.hypot(rx, ry), 0.0, False)

        # In ego frame, obstacle.vx from tracker is already relative rate of change of ego distance.
        # If obstacle.vx is nonzero, use it directly; if zero (stationary/fresh), relative speed is -ego_speed
        if abs(obstacle.vx) > 0.05:
            rvx = obstacle.vx
        else:
            rvx = -ego_speed
        rvy = obstacle.vy

        combined_radius = self.ego_radius + max(obstacle.length, obstacle.width) / 2.0

        # Current distance
        current_dist = math.hypot(rx, ry)
        if current_dist <= combined_radius:
            return TTCEstimate(obstacle.track_id, 0.0, current_dist, 0.0, True)

        # Relative speed squared
        v_rel_sq = rvx * rvx + rvy * rvy
        if v_rel_sq < 1e-4:
            # Vehicles not moving relative to each other
            return TTCEstimate(obstacle.track_id, math.inf, current_dist, math.inf, False)

        # Dot product: p . v
        p_dot_v = rx * rvx + ry * rvy
        if p_dot_v >= 0:
            # Diverging or moving away
            return TTCEstimate(obstacle.track_id, math.inf, current_dist, 0.0, False)

        # Time to closest point of approach (t_cpa = - (p . v) / |v|^2)
        t_cpa = -p_dot_v / v_rel_sq

        # Closest distance squared: d_cpa^2 = |p|^2 - (p . v)^2 / |v|^2
        d_cpa_sq = (rx * rx + ry * ry) - (p_dot_v * p_dot_v) / v_rel_sq
        d_cpa = math.sqrt(max(0.0, d_cpa_sq))

        if d_cpa > combined_radius:
            # Misses the vehicle clearance boundary
            return TTCEstimate(
                obstacle.track_id,
                math.inf,
                round(d_cpa, 2),
                round(t_cpa, 2),
                False,
            )

        # Collision time (entry into circle of radius combined_radius)
        # solve |p + v * t|^2 = R^2
        # |v|^2 * t^2 + 2(p . v) t + (|p|^2 - R^2) = 0
        c = (rx * rx + ry * ry) - combined_radius * combined_radius
        discriminant = (2.0 * p_dot_v) ** 2 - 4.0 * v_rel_sq * c
        if discriminant < 0:
            return TTCEstimate(obstacle.track_id, math.inf, round(d_cpa, 2), round(t_cpa, 2), False)

        t_entry = (-2.0 * p_dot_v - math.sqrt(discriminant)) / (2.0 * v_rel_sq)
        ttc = max(0.0, t_entry)

        return TTCEstimate(
            obstacle_id=obstacle.track_id,
            ttc_sec=round(ttc, 1),
            min_distance_m=round(d_cpa, 2),
            time_at_cpa_s=round(t_cpa, 2),
            is_imminent=(ttc < self.threshold),
        )
