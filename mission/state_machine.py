"""Mission execution state machine and real-time progress tracker for AURA.

Implements the authoritative mission lifecycle:
IDLE_AT_CHARGER / CHARGING
-> ASSIGNED
-> TRAVELLING_TO_PICKUP / EN_ROUTE_PICKUP
-> DOCKING_AT_PICKUP
-> LOADING / PICKING (at least 5 simulation seconds progress in simulation)
-> WAITING_FOR_LOAD_APPROVAL (explicit worker approval required)
-> TRAVELLING_TO_DELIVERY / EN_ROUTE_DELIVERY
-> DOCKING_AT_DELIVERY
-> UNLOADING (at least 5 simulation seconds progress in simulation)
-> WAITING_FOR_UNLOAD_APPROVAL (explicit worker approval required)
-> RETURNING_TO_CHARGER / RETURNING_TO_CHARGE
-> DOCKING_AT_CHARGER
-> CHARGING
-> AVAILABLE / COMPLETED

Special states:
- PAUSED
- BLOCKED
- EMERGENCY_STOP / ABORTED
- CANCELLED
- FAULT
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from mission.parser import MissionTask, TaskType
from simulation.factory_world import DockLocation


class MissionStatus(str, Enum):
    # Lifecycle states with backward-compatible aliases
    IDLE = "IDLE"
    IDLE_AT_CHARGER = "IDLE_AT_CHARGER"
    ASSIGNED = "ASSIGNED"
    EN_ROUTE_PICKUP = "EN_ROUTE_PICKUP"
    TRAVELLING_TO_PICKUP = "TRAVELLING_TO_PICKUP"
    DOCKING_AT_PICKUP = "DOCKING_AT_PICKUP"
    PICKING = "PICKING"
    LOADING = "LOADING"
    WAITING_FOR_LOAD_APPROVAL = "WAITING_FOR_LOAD_APPROVAL"
    EN_ROUTE_DELIVERY = "EN_ROUTE_DELIVERY"
    TRAVELLING_TO_DELIVERY = "TRAVELLING_TO_DELIVERY"
    DOCKING_AT_DELIVERY = "DOCKING_AT_DELIVERY"
    UNLOADING = "UNLOADING"
    WAITING_FOR_UNLOAD_APPROVAL = "WAITING_FOR_UNLOAD_APPROVAL"
    RETURNING_TO_CHARGE = "RETURNING_TO_CHARGE"
    RETURNING_TO_CHARGER = "RETURNING_TO_CHARGER"
    DOCKING_AT_CHARGER = "DOCKING_AT_CHARGER"
    CHARGING = "CHARGING"
    AVAILABLE = "AVAILABLE"
    COMPLETED = "COMPLETED"

    # Contingency / control states
    PAUSED = "PAUSED"
    BLOCKED = "BLOCKED"
    ABORTED = "ABORTED"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    CANCELLED = "CANCELLED"
    FAULT = "FAULT"


@dataclass
class MissionProgress:
    status: MissionStatus
    percent_complete: float  # [0, 100]
    estimated_remaining_time_str: str  # e.g. "03:15"
    subtext: str  # e.g. "IN PROGRESS • MOVING TO DOCK A"
    active_task: MissionTask | None
    current_goal: tuple[float, float]
    cargo_loaded: bool = False
    dwell_progress_pct: float = 0.0  # [0, 100] during loading/unloading
    awaiting_approval: bool = False
    approval_type: str = ""  # "load", "unload", or ""
    current_dock_name: str = ""


class MissionStateMachine:
    """Manages the full autonomous lifecycle of an industrial logistics mission."""

    def __init__(
        self,
        waypoint_tolerance: float = 1.8,
        target_dwell: float = 2.0,
        require_worker_approval: bool = False,
        home_charger: tuple[float, float] = (-14.0, -14.0),
    ) -> None:
        self.tolerance = waypoint_tolerance
        self.target_dwell = target_dwell
        self.require_worker_approval = require_worker_approval
        self.home_charger = home_charger

        self.status = MissionStatus.IDLE
        self.active_task: MissionTask | None = None
        self.current_goal: tuple[float, float] = home_charger
        self.dwell_time = 0.0
        self.cargo_loaded = False
        self.total_mission_distance = 1.0
        self.distance_traveled = 0.0
        self._last_pose: tuple[float, float] | None = None
        self._pre_pause_status: MissionStatus | None = None
        self.current_dock_name: str = ""
        self.event_log: list[str] = []

    def set_task(self, task: MissionTask, current_pose: tuple[float, float]) -> None:
        """Initialize and start a new mission task."""
        self.active_task = task
        self.dwell_time = 0.0
        self.distance_traveled = 0.0
        self._last_pose = current_pose

        if task.task_type == TaskType.ABORT:
            self.status = MissionStatus.ABORTED
            self.current_goal = current_pose
            self.event_log.append("Emergency stop triggered")
            return

        if task.task_type == TaskType.PICKUP_AND_DELIVER:
            self.status = MissionStatus.EN_ROUTE_PICKUP
            assert task.pickup_location is not None
            assert task.delivery_location is not None
            self.current_goal = (task.pickup_location.x, task.pickup_location.y)
            self.current_dock_name = task.pickup_location.name
            d1 = float(np.hypot(task.pickup_location.x - current_pose[0], task.pickup_location.y - current_pose[1]))
            d2 = float(np.hypot(task.delivery_location.x - task.pickup_location.x, task.delivery_location.y - task.pickup_location.y))
            d3 = float(np.hypot(self.home_charger[0] - task.delivery_location.x, self.home_charger[1] - task.delivery_location.y))
            self.total_mission_distance = max(d1 + d2 + d3, 1.0)
            self.cargo_loaded = False
            self.event_log.append(f"Mission started: {task.display_text}")

        elif task.task_type == TaskType.RETURN_TO_CHARGE:
            self.status = MissionStatus.RETURNING_TO_CHARGE
            target = task.delivery_location
            self.current_goal = (target.x, target.y) if target else self.home_charger
            self.current_dock_name = target.name if target else "Home Charger"
            self.total_mission_distance = max(float(np.hypot(self.current_goal[0] - current_pose[0], self.current_goal[1] - current_pose[1])), 1.0)
            self.event_log.append("Returning to charging bay")

        elif task.task_type == TaskType.NAVIGATE_TO:
            self.status = MissionStatus.EN_ROUTE_DELIVERY
            target = task.delivery_location
            self.current_goal = (target.x, target.y) if target else current_pose
            self.current_dock_name = target.name if target else "Destination"
            self.total_mission_distance = max(float(np.hypot(self.current_goal[0] - current_pose[0], self.current_goal[1] - current_pose[1])), 1.0)
            self.event_log.append(f"Navigating to {self.current_dock_name}")

    def pause(self) -> None:
        """Pause mission execution safely."""
        if self.status not in (MissionStatus.PAUSED, MissionStatus.EMERGENCY_STOP, MissionStatus.ABORTED):
            self._pre_pause_status = self.status
            self.status = MissionStatus.PAUSED
            self.event_log.append("Mission paused")

    def resume(self) -> None:
        """Resume paused mission."""
        if self.status == MissionStatus.PAUSED and self._pre_pause_status is not None:
            self.status = self._pre_pause_status
            self._pre_pause_status = None
            self.event_log.append("Mission resumed")

    def abort(self) -> None:
        """Trigger immediate emergency mission abort."""
        self.status = MissionStatus.ABORTED
        self.event_log.append("Mission aborted via E-STOP")

    def emergency_stop(self) -> None:
        """Trigger immediate emergency stop."""
        self.status = MissionStatus.EMERGENCY_STOP
        self.event_log.append("Emergency stop engaged")


    def cancel(self) -> None:
        """Cancel current active mission and set destination to home charger."""
        self.status = MissionStatus.CANCELLED
        self.current_goal = self.home_charger
        self.cargo_loaded = False
        self.event_log.append("Mission cancelled; returning home")

    def approve_load(self, dock_id: str | None = None) -> bool:
        """Explicit worker approval to depart pickup dock after loading."""
        if self.status in (MissionStatus.WAITING_FOR_LOAD_APPROVAL, MissionStatus.PICKING, MissionStatus.LOADING):
            self.status = MissionStatus.EN_ROUTE_DELIVERY
            self.cargo_loaded = True
            if self.active_task and self.active_task.delivery_location:
                loc = self.active_task.delivery_location
                self.current_goal = (loc.x, loc.y)
                self.current_dock_name = loc.name
            self.dwell_time = 0.0
            self.event_log.append(f"Loading approved at {self.current_dock_name}. Departing.")
            return True
        return False

    def approve_unload(self, dock_id: str | None = None) -> bool:
        """Explicit worker approval after unloading complete."""
        if self.status in (MissionStatus.WAITING_FOR_UNLOAD_APPROVAL, MissionStatus.UNLOADING):
            self.status = MissionStatus.RETURNING_TO_CHARGE
            self.cargo_loaded = False
            self.current_goal = self.home_charger
            self.current_dock_name = "Charging Bay"
            self.dwell_time = 0.0
            self.event_log.append("Unloading approved. Returning to charger.")
            return True
        return False

    def update(self, current_pose: tuple[float, float], speed: float = 1.5, dt: float = 0.05) -> MissionProgress:
        """Update mission state machine based on vehicle current position."""
        if self.status in (MissionStatus.PAUSED, MissionStatus.EMERGENCY_STOP, MissionStatus.ABORTED):
            return self._build_progress(current_pose, speed)

        if self._last_pose is not None:
            step_dist = float(np.hypot(current_pose[0] - self._last_pose[0], current_pose[1] - self._last_pose[1]))
            self.distance_traveled += step_dist
        self._last_pose = current_pose

        dist_to_goal = float(np.hypot(self.current_goal[0] - current_pose[0], self.current_goal[1] - current_pose[1]))

        # --- State Transitions ---
        # 1. Heading to pickup
        if self.status in (MissionStatus.EN_ROUTE_PICKUP, MissionStatus.TRAVELLING_TO_PICKUP, MissionStatus.ASSIGNED):
            if dist_to_goal <= self.tolerance:
                self.status = MissionStatus.PICKING
                self.dwell_time = 0.0

        # 2. Loading / Picking
        elif self.status in (MissionStatus.PICKING, MissionStatus.LOADING):
            self.dwell_time += dt
            if self.dwell_time >= self.target_dwell:
                self.cargo_loaded = True
                if self.require_worker_approval:
                    self.status = MissionStatus.WAITING_FOR_LOAD_APPROVAL
                else:
                    self.status = MissionStatus.EN_ROUTE_DELIVERY
                    if self.active_task and self.active_task.delivery_location:
                        loc = self.active_task.delivery_location
                        self.current_goal = (loc.x, loc.y)
                        self.current_dock_name = loc.name
                self.dwell_time = 0.0

        # 3. Waiting for Load Approval
        elif self.status == MissionStatus.WAITING_FOR_LOAD_APPROVAL:
            # Remains stationary until approve_load() called
            pass

        # 4. Heading to delivery
        elif self.status in (MissionStatus.EN_ROUTE_DELIVERY, MissionStatus.TRAVELLING_TO_DELIVERY):
            if dist_to_goal <= self.tolerance:
                self.status = MissionStatus.UNLOADING
                self.dwell_time = 0.0

        # 5. Unloading
        elif self.status == MissionStatus.UNLOADING:
            self.dwell_time += dt
            if self.dwell_time >= self.target_dwell:
                self.cargo_loaded = False
                if self.require_worker_approval:
                    self.status = MissionStatus.WAITING_FOR_UNLOAD_APPROVAL
                else:
                    self.status = MissionStatus.COMPLETED
                self.dwell_time = 0.0

        # 6. Waiting for Unload Approval
        elif self.status == MissionStatus.WAITING_FOR_UNLOAD_APPROVAL:
            # Remains stationary until approve_unload() called
            pass

        # 7. Returning to Charger
        elif self.status in (MissionStatus.RETURNING_TO_CHARGE, MissionStatus.RETURNING_TO_CHARGER):
            if dist_to_goal <= self.tolerance:
                self.status = MissionStatus.COMPLETED
                self.dwell_time = 0.0

        return self._build_progress(current_pose, speed)

    def _build_progress(self, current_pose: tuple[float, float], speed: float) -> MissionProgress:
        """Construct MissionProgress data transfer object."""
        dwell_pct = 0.0
        if self.status in (MissionStatus.PICKING, MissionStatus.LOADING, MissionStatus.UNLOADING):
            dwell_pct = min(100.0, (self.dwell_time / max(self.target_dwell, 0.1)) * 100.0)

        awaiting_approval = self.status in (
            MissionStatus.WAITING_FOR_LOAD_APPROVAL,
            MissionStatus.WAITING_FOR_UNLOAD_APPROVAL,
        )
        approval_type = "load" if self.status == MissionStatus.WAITING_FOR_LOAD_APPROVAL else \
                        "unload" if self.status == MissionStatus.WAITING_FOR_UNLOAD_APPROVAL else ""

        if self.status in (MissionStatus.COMPLETED, MissionStatus.CHARGING):
            pct = 100.0
            time_str = "00:00"
            subtext = "CHARGING • AT CHARGING BAY" if self.status == MissionStatus.CHARGING else "COMPLETED • MISSION FINISHED"
        elif self.status in (MissionStatus.EMERGENCY_STOP, MissionStatus.ABORTED):
            pct = min(100.0, (self.distance_traveled / max(self.total_mission_distance, 1.0)) * 100.0)
            time_str = "--:--"
            subtext = "STOPPED • MISSION ABORTED"
        elif self.status == MissionStatus.PAUSED:
            pct = min(100.0, (self.distance_traveled / max(self.total_mission_distance, 1.0)) * 100.0)
            time_str = "--:--"
            subtext = "PAUSED • AWAITING RESUME"
        elif self.status == MissionStatus.CANCELLED:
            pct = 100.0
            time_str = "--:--"
            subtext = "CANCELLED • MISSION CANCELLED"
        elif self.status in (MissionStatus.IDLE, MissionStatus.IDLE_AT_CHARGER, MissionStatus.AVAILABLE):
            pct = 0.0
            time_str = "00:00"
            subtext = "READY • AWAITING COMMAND"
        elif self.status == MissionStatus.WAITING_FOR_LOAD_APPROVAL:
            pct = 50.0
            time_str = "--:--"
            subtext = f"LOADED • AWAITING WORKER APPROVAL ({self.current_dock_name})"
        elif self.status == MissionStatus.WAITING_FOR_UNLOAD_APPROVAL:
            pct = 85.0
            time_str = "--:--"
            subtext = f"UNLOADED • AWAITING WORKER APPROVAL ({self.current_dock_name})"
        elif self.status in (MissionStatus.PICKING, MissionStatus.LOADING):
            pct = 40.0 + (dwell_pct * 0.1)
            rem = max(0.0, self.target_dwell - self.dwell_time)
            time_str = f"00:{int(rem):02d}"
            subtext = f"LOADING PALLET ({int(dwell_pct)}%) • {self.current_dock_name}"
        elif self.status == MissionStatus.UNLOADING:
            pct = 75.0 + (dwell_pct * 0.1)
            rem = max(0.0, self.target_dwell - self.dwell_time)
            time_str = f"00:{int(rem):02d}"
            subtext = f"UNLOADING PALLET ({int(dwell_pct)}%) • {self.current_dock_name}"
        else:
            pct = min(98.0, max(5.0, (self.distance_traveled / max(self.total_mission_distance, 1.0)) * 100.0))
            rem_dist = max(0.0, self.total_mission_distance - self.distance_traveled)
            est_sec = int(rem_dist / max(speed, 0.5))
            m, s = divmod(est_sec, 60)
            time_str = f"{m:02d}:{s:02d}"

            dest_label = self.current_dock_name or "DESTINATION"
            subtext = f"IN PROGRESS • MOVING TO {dest_label.upper()}"

        return MissionProgress(
            status=self.status,
            percent_complete=round(pct, 1),
            estimated_remaining_time_str=time_str,
            subtext=subtext,
            active_task=self.active_task,
            current_goal=self.current_goal,
            cargo_loaded=self.cargo_loaded,
            dwell_progress_pct=round(dwell_pct, 1),
            awaiting_approval=awaiting_approval,
            approval_type=approval_type,
            current_dock_name=self.current_dock_name,
        )
