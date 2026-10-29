"""Local obstacle avoidance and trajectory tracking planner for AURA."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from perception.tracker import TrackedObstacle
from planning.global_planner import Waypoint
from simulation.geometry import wrap_angle


class LocalSplinePlanner:
    """Computes steering and acceleration commands to follow waypoints while evading obstacles."""

    def __init__(
        self,
        lookahead_distance: float = 3.5,
        k_p_steer: float = 1.8,
        safety_clearance: float = 2.2,
    ) -> None:
        self.lookahead = lookahead_distance
        self.kp_steer = k_p_steer
        self.safety_clearance = safety_clearance

    def compute_control(
        self,
        current_pose: tuple[float, float, float],  # x, y, heading
        current_speed: float,
        waypoints: Sequence[Waypoint],
        tracked_obstacles: Sequence[TrackedObstacle],
        speed_limit: float = 3.0,
    ) -> tuple[float, float]:
        """Generate (accel_cmd, steer_cmd)."""
        x, y, heading = current_pose

        if not waypoints:
            return -2.0, 0.0  # Brake to stop if no path

        # 1. Find closest waypoint index within immediate forward horizon
        closest_idx = 0
        min_d = math.hypot(waypoints[0].x - x, waypoints[0].y - y)
        search_limit = min(8, len(waypoints))
        for i in range(1, search_limit):
            d = math.hypot(waypoints[i].x - x, waypoints[i].y - y)
            if d < min_d:
                min_d = d
                closest_idx = i

        # Adaptive speed-dependent lookahead distance (1.8m at low speed/turns, up to 4.0m cruising)
        effective_lookahead = max(1.8, min(self.lookahead, 1.2 * max(0.5, current_speed)))

        target_wp = waypoints[-1]
        for i in range(closest_idx, len(waypoints)):
            wp = waypoints[i]
            d = math.hypot(wp.x - x, wp.y - y)
            if d >= effective_lookahead:
                target_wp = wp
                break

        # Vector to target in world coordinates
        dx = target_wp.x - x
        dy = target_wp.y - y
        target_angle = math.atan2(dy, dx)
        heading_error = wrap_angle(target_angle - heading)

        # 2. Obstacle repulsion bias (reactive lateral push relative to ego position)
        repulsion_bias = 0.0
        for obs in tracked_obstacles:
            # TrackedObstacle is already in ego vehicle frame (x forward, y left)
            ox_ego = obs.x
            oy_ego = obs.y

            # Only react to obstacles in forward driving corridor
            if 0.5 < ox_ego < 8.0 and abs(oy_ego) < self.safety_clearance:
                push_dir = -1.0 if oy_ego >= 0 else 1.0
                clearance_deficit = self.safety_clearance - abs(oy_ego)
                repulsion_bias += push_dir * clearance_deficit * 0.35

        # Steering command = Pure Pursuit with damping + obstacle avoidance
        steer_cmd = self.kp_steer * heading_error + repulsion_bias
        steer_cmd = max(-0.75, min(0.75, steer_cmd))

        # 3. Acceleration command & Speed Profiling
        dist_to_final = math.hypot(waypoints[-1].x - x, waypoints[-1].y - y)
        if dist_to_final < 1.2:
            target_v = 0.3  # Precise docking halt
        elif dist_to_final < 4.0:
            target_v = 1.2  # Dock approach speed
        else:
            target_v = min(speed_limit, target_wp.target_speed_mps)

        # Cornering speed governor: reduce speed smoothly when steering sharply
        abs_steer = abs(steer_cmd)
        if abs_steer > 0.15:
            turn_factor = max(0.25, 1.0 - 1.1 * (abs_steer - 0.15))
            target_v = min(target_v, max(0.9, target_v * turn_factor))

        # Check for obstacles directly ahead in the vehicle's driving corridor
        for obs in tracked_obstacles:
            ox_ego = obs.x
            oy_ego = obs.y
            if 0.5 < ox_ego < 6.0 and abs(oy_ego) < 1.4:
                if ox_ego < 2.5:
                    target_v = 0.0  # Stop safely before obstacle
                else:
                    target_v = min(target_v, 1.0)  # Slow down smoothly

        v_err = target_v - current_speed
        accel_gain = 1.8 if v_err > 0 else 2.5
        accel_cmd = max(-3.5, min(2.0, accel_gain * v_err))

        return accel_cmd, steer_cmd

