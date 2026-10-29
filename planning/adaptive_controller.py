"""Integrated Adaptive Planning Controller for AURA.

Executes real-time decision selection (CONTINUE, SLOW_DOWN, STOP, REROUTE, SAFE_RETURN)
based on risk evaluation, sensor reliability, and vehicle health.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import numpy as np

from perception.tracker import TrackedObstacle
from planning.global_planner import AStarGlobalPlanner, Waypoint
from planning.local_planner import LocalSplinePlanner
from risk.risk_assessor import AssessmentReport, RiskLevel
from simulation.amr_vehicle import AMRState
from simulation.environment import Environment


class AdaptiveAction(str, Enum):
    CONTINUE = "CONTINUE"
    SLOW_DOWN_AND_REROUTE = "SLOW DOWN + REROUTE"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    SAFE_RETURN = "SAFE_RETURN"


@dataclass
class PlannerOutput:
    action: AdaptiveAction
    accel_cmd: float
    steer_cmd: float
    normal_path: list[Waypoint]
    alternative_path: list[Waypoint] | None
    is_rerouting: bool


class AdaptivePlannerController:
    """Master closed-loop planner integrating global route, local avoidance, and risk adaptation."""

    def __init__(
        self,
        global_planner: AStarGlobalPlanner | None = None,
        local_planner: LocalSplinePlanner | None = None,
    ) -> None:
        self.global_planner = global_planner or AStarGlobalPlanner()
        self.local_planner = local_planner or LocalSplinePlanner()
        self.current_normal_path: list[Waypoint] = []
        self.current_alt_path: list[Waypoint] | None = None
        self.is_rerouting = False
        self._step_counter = 0

    def plan_step(
        self,
        vehicle_state: AMRState,
        goal: tuple[float, float],
        env: Environment,
        tracks: Sequence[TrackedObstacle],
        risk_report: AssessmentReport,
    ) -> PlannerOutput:
        """Execute one complete planning cycle."""
        curr_pose = (vehicle_state.x, vehicle_state.y, vehicle_state.heading)

        # 1. Evaluate Risk Action
        action_str = risk_report.recommended_action
        if action_str == "EMERGENCY_STOP":
            action = AdaptiveAction.EMERGENCY_STOP
            return PlannerOutput(
                action=action,
                accel_cmd=-3.5,  # Maximum safe emergency braking
                steer_cmd=0.0,
                normal_path=self.current_normal_path,
                alternative_path=self.current_alt_path,
                is_rerouting=self.is_rerouting,
            )

        # 2. Plan or refresh Normal Path
        start_pt = (vehicle_state.x, vehicle_state.y)
        if not self.current_normal_path or len(self.current_normal_path) < 2:
            self.current_normal_path = self.global_planner.plan_path(start_pt, goal, env)
        else:
            # Prune waypoints already passed by ego (search forward in immediate horizon)
            closest_i = 0
            min_d = math.hypot(self.current_normal_path[0].x - vehicle_state.x, self.current_normal_path[0].y - vehicle_state.y)
            search_limit = min(8, len(self.current_normal_path))
            for i in range(1, search_limit):
                d = math.hypot(self.current_normal_path[i].x - vehicle_state.x, self.current_normal_path[i].y - vehicle_state.y)
                if d < min_d:
                    min_d = d
                    closest_i = i
            if closest_i > 0 and len(self.current_normal_path) - closest_i >= 2:
                self.current_normal_path = self.current_normal_path[closest_i:]

        # 3. Reroute evaluation if HIGH / MEDIUM risk or blocking obstacles
        speed_limit = 4.5  # Cruising speed on clear arterial roads
        self._step_counter += 1
        if "REROUTE" in action_str or risk_report.overall_risk_level in (RiskLevel.HIGH, RiskLevel.MEDIUM):
            self.is_rerouting = True
            action = AdaptiveAction.SLOW_DOWN_AND_REROUTE
            speed_limit = 2.0  # Throttle speed for safety during evasive reroute
            if not self.current_alt_path:
                avoid_regions = [(t.x, t.y, 2.0) for t in tracks] if tracks else [(curr_pose[0] + 3.0, curr_pose[1], 1.5)]
                self.current_alt_path = self.global_planner.plan_path(start_pt, goal, env, avoid_regions=avoid_regions)
            active_waypoints = self.current_normal_path
        else:
            self.is_rerouting = False
            self.current_alt_path = None
            action = AdaptiveAction.CONTINUE
            active_waypoints = self.current_normal_path

        # 4. Mechanical health and battery adjustments
        if vehicle_state.battery_soc_pct < 20.0 or vehicle_state.motor_health < 0.7:
            speed_limit = min(speed_limit, 1.8)

        # 5. Local tracking control
        accel_cmd, steer_cmd = self.local_planner.compute_control(
            curr_pose,
            vehicle_state.speed,
            active_waypoints,
            tracks,
            speed_limit=speed_limit,
        )

        return PlannerOutput(
            action=action,
            accel_cmd=accel_cmd,
            steer_cmd=steer_cmd,
            normal_path=self.current_normal_path,
            alternative_path=self.current_alt_path,
            is_rerouting=self.is_rerouting,
        )
