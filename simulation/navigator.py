"""Goal-directed autonomous navigator for AURA vehicles.

Connects the mission state machine's current goal to the adaptive planner,
producing real accel/steer commands that drive the vehicle toward its
mission waypoints instead of hard-coded constant values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mission.state_machine import MissionProgress, MissionStatus
from planning.adaptive_controller import AdaptivePlannerController, PlannerOutput
from planning.global_planner import AStarGlobalPlanner, Waypoint
from planning.local_planner import LocalSplinePlanner
from perception.tracker import TrackedObstacle
from risk.risk_assessor import AssessmentReport
from simulation.amr_vehicle import AMRState
from simulation.environment import Environment


@dataclass
class NavigationCommand:
    """Output of the navigator for one tick."""

    accel_cmd: float
    steer_cmd: float
    normal_path: list[Waypoint]
    alt_path: list[Waypoint] | None
    is_rerouting: bool
    is_waiting: bool  # True during pick/unload dwell


class AutonomousNavigator:
    """Bridges the mission goal to adaptive path planning and control.

    Replaces the previous hard-coded accel/steer with goal-directed navigation.
    """

    def __init__(self) -> None:
        self.planner = AdaptivePlannerController(
            global_planner=AStarGlobalPlanner(resolution=0.5, vehicle_radius=1.25),
            local_planner=LocalSplinePlanner(
                lookahead_distance=4.0,
                k_p_steer=1.8,
                safety_clearance=2.2,
            ),
        )

        self._last_goal: tuple[float, float] | None = None
        self._path_stale_counter = 0

    def compute(
        self,
        vehicle_state: AMRState,
        mission_progress: MissionProgress,
        env: Environment,
        tracks: list[TrackedObstacle],
        risk_report: AssessmentReport,
    ) -> NavigationCommand:
        """Compute one navigation tick based on the current mission state."""
        goal = mission_progress.current_goal

        # If mission is completed, aborted, paused, charging, or in dwell/approval states, stop
        waiting_states = {
            MissionStatus.PICKING,
            MissionStatus.LOADING,
            MissionStatus.WAITING_FOR_LOAD_APPROVAL,
            MissionStatus.UNLOADING,
            MissionStatus.WAITING_FOR_UNLOAD_APPROVAL,
            MissionStatus.COMPLETED,
            MissionStatus.ABORTED,
            MissionStatus.EMERGENCY_STOP,
            MissionStatus.PAUSED,
            MissionStatus.CANCELLED,
            MissionStatus.FAULT,
            MissionStatus.IDLE,
            MissionStatus.IDLE_AT_CHARGER,
            MissionStatus.CHARGING,
            MissionStatus.AVAILABLE,
        }
        if mission_progress.status in waiting_states:
            return NavigationCommand(
                accel_cmd=-2.0,  # gentle brake to stop
                steer_cmd=0.0,
                normal_path=self.planner.current_normal_path,
                alt_path=self.planner.current_alt_path,
                is_rerouting=False,
                is_waiting=True,
            )

        # Invalidate cached path when goal changes
        if self._last_goal is None or math.hypot(
            goal[0] - self._last_goal[0], goal[1] - self._last_goal[1]
        ) > 1.0:
            self.planner.current_normal_path = []
            self.planner.current_alt_path = None
            self._last_goal = goal

        plan = self.planner.plan_step(
            vehicle_state=vehicle_state,
            goal=goal,
            env=env,
            tracks=tracks,
            risk_report=risk_report,
        )

        return NavigationCommand(
            accel_cmd=plan.accel_cmd,
            steer_cmd=plan.steer_cmd,
            normal_path=plan.normal_path,
            alt_path=plan.alternative_path,
            is_rerouting=plan.is_rerouting,
            is_waiting=False,
        )
