"""Phase 1 reactive baseline: go to the goal and push away from obstacles.

This is intentionally simple. It reacts only to *current* obstacle positions,
which makes it a fair "Baseline A" to beat in later phases.
"""

from __future__ import annotations

import math

from config import ControllerParams, VehicleParams
from simulation.geometry import clamp, wrap_angle
from simulation.obstacles import Obstacle
from simulation.vehicle import VehicleCommand, VehicleState


class ReactiveGoalController:
    """Potential-field style steering plus clearance-based speed limiting."""

    def __init__(
        self,
        vehicle_params: VehicleParams | None = None,
        params: ControllerParams | None = None,
    ) -> None:
        self.vp = vehicle_params or VehicleParams()
        self.cp = params or ControllerParams()

    def compute(
        self,
        state: VehicleState,
        goal: tuple[float, float],
        obstacles: list[Obstacle],
    ) -> VehicleCommand:
        cp, vp = self.cp, self.vp
        dx, dy = goal[0] - state.x, goal[1] - state.y
        goal_dist = math.hypot(dx, dy)
        if goal_dist < 1e-9:
            return VehicleCommand(0.0, 0.0)

        # Attraction towards the goal (unit vector).
        vec_x, vec_y = dx / goal_dist, dy / goal_dist
        cos_t, sin_t = math.cos(cp.tangential_angle), math.sin(cp.tangential_angle)
        front_gap = math.inf

        for obs in obstacles:
            away_x, away_y = state.x - obs.x, state.y - obs.y
            centre = math.hypot(away_x, away_y)
            if centre < 1e-9:
                continue
            gap = centre - obs.radius - vp.radius

            # Is the obstacle in front of us? (used for speed limiting)
            bearing = wrap_angle(math.atan2(-away_y, -away_x) - state.heading)
            if abs(bearing) <= cp.front_cone:
                front_gap = min(front_gap, gap)

            if gap >= cp.influence_gap:
                continue
            strength = cp.repulsion_gain * (cp.influence_gap - max(gap, 0.0))
            strength /= cp.influence_gap
            ux, uy = away_x / centre, away_y / centre
            # Rotating the push slightly breaks head-on symmetry.
            vec_x += strength * (ux * cos_t - uy * sin_t)
            vec_y += strength * (ux * sin_t + uy * cos_t)

        desired = math.atan2(vec_y, vec_x)
        error = wrap_angle(desired - state.heading)
        steering = clamp(cp.k_steer * error, -vp.max_steer, vp.max_steer)

        # Speed limits: cruise, smooth stop at goal, clearance, sharp turns.
        stop_speed = math.sqrt(2.0 * cp.comfort_decel * goal_dist)
        clearance_speed = vp.max_speed
        if front_gap < cp.slow_gap:
            ratio = clamp(front_gap / cp.slow_gap, 0.0, 1.0)
            clearance_speed = max(cp.min_speed_near_obstacle, vp.max_speed * ratio)
        turn_speed = vp.max_speed * max(0.3, math.cos(min(abs(error), 1.2)))

        target = min(vp.max_speed, stop_speed, clearance_speed, turn_speed)
        return VehicleCommand(target_speed=target, steering=steering)
