"""Fixed-step simulation loop, termination logic, logging and metrics."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Protocol

import pandas as pd

from config import SimulationParams
from simulation.environment import Environment
from simulation.obstacles import Obstacle
from simulation.vehicle import Vehicle, VehicleCommand, VehicleState

logger = logging.getLogger(__name__)


class Controller(Protocol):
    """Anything that maps (state, goal, obstacles) to a command.

    The Phase 1 reactive controller satisfies this; the adaptive planner of
    Phase 9 will satisfy it too, so the simulator never has to change.
    """

    def compute(
        self,
        state: VehicleState,
        goal: tuple[float, float],
        obstacles: list[Obstacle],
    ) -> VehicleCommand: ...


class TerminationReason(str, Enum):
    GOAL_REACHED = "goal_reached"
    COLLISION = "collision"
    OUT_OF_BOUNDS = "out_of_bounds"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class StepRecord:
    """One logged row per simulation step."""

    t: float
    x: float
    y: float
    heading: float
    speed: float
    target_speed: float
    steering: float
    min_gap: float
    goal_distance: float


ObstacleSnapshot = tuple[tuple[float, float, float], ...]  # (x, y, radius)


@dataclass
class SimulationResult:
    scenario: str
    reason: TerminationReason
    path_length: float
    near_miss_gap: float
    records: list[StepRecord] = field(default_factory=list)
    obstacle_history: list[ObstacleSnapshot] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.records])

    def near_miss_events(self) -> int:
        """Number of separate occasions the clearance fell below the threshold."""
        events, inside = 0, False
        for record in self.records:
            now_inside = 0.0 < record.min_gap < self.near_miss_gap
            if now_inside and not inside:
                events += 1
            inside = now_inside
        return events

    def summary(self) -> dict[str, float | int | str]:
        finite_gaps = [r.min_gap for r in self.records if math.isfinite(r.min_gap)]
        duration = self.records[-1].t if self.records else 0.0
        speeds = [r.speed for r in self.records]
        return {
            "scenario": self.scenario,
            "outcome": self.reason.value,
            "duration_s": round(duration, 2),
            "path_length_m": round(self.path_length, 2),
            "mean_speed_mps": round(sum(speeds) / len(speeds), 3) if speeds else 0.0,
            "min_clearance_m": round(min(finite_gaps), 3) if finite_gaps else math.inf,
            "near_miss_events": self.near_miss_events(),
            "steps": len(self.records),
        }


class Simulator:
    """Runs one episode: world + vehicle + controller until a stop condition."""

    def __init__(
        self,
        environment: Environment,
        vehicle: Vehicle,
        controller: Controller,
        params: SimulationParams | None = None,
        name: str = "custom",
    ) -> None:
        self.env = environment
        self.vehicle = vehicle
        self.controller = controller
        self.params = params or SimulationParams()
        self.name = name

    def _snapshot(self, command: VehicleCommand) -> StepRecord:
        s, r = self.vehicle.state, self.vehicle.params.radius
        return StepRecord(
            t=round(self.env.time, 6),
            x=s.x,
            y=s.y,
            heading=s.heading,
            speed=s.speed,
            target_speed=command.target_speed,
            steering=command.steering,
            min_gap=self.env.min_gap(s.x, s.y, r),
            goal_distance=self.env.goal_distance(s.x, s.y),
        )

    def _obstacles_now(self) -> ObstacleSnapshot:
        return tuple((o.x, o.y, o.radius) for o in self.env.obstacles)

    def _check_termination(self) -> TerminationReason | None:
        s, r = self.vehicle.state, self.vehicle.params.radius
        if self.env.in_collision(s.x, s.y, r):
            return TerminationReason.COLLISION
        if not self.env.bounds.contains_circle(s.x, s.y, r):
            return TerminationReason.OUT_OF_BOUNDS
        if self.env.goal_reached(s.x, s.y):
            return TerminationReason.GOAL_REACHED
        return None

    def run(self) -> SimulationResult:
        dt = self.params.dt
        max_steps = int(self.params.max_time / dt)
        result = SimulationResult(
            scenario=self.name,
            reason=TerminationReason.TIMEOUT,
            path_length=0.0,
            near_miss_gap=self.params.near_miss_gap,
        )
        result.records.append(self._snapshot(VehicleCommand(0.0, 0.0)))
        result.obstacle_history.append(self._obstacles_now())

        for _ in range(max_steps):
            command = self.controller.compute(
                self.vehicle.state, self.env.goal, self.env.obstacles
            )
            self.vehicle.step(command, dt)
            self.env.step(dt)
            result.records.append(self._snapshot(command))
            result.obstacle_history.append(self._obstacles_now())

            reason = self._check_termination()
            if reason is not None:
                result.reason = reason
                break

        result.path_length = self.vehicle.distance_travelled
        logger.info("Run finished: %s", result.summary())
        return result
