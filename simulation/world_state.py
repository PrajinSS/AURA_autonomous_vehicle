"""Ground-truth snapshot of the whole world at one instant.
This is the *only* thing the Phase 3 virtual sensors will be allowed to read.
Sensors turn this perfect truth into noisy, faulty measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from simulation.obstacles import Obstacle
from simulation.vehicle import VehicleState


@dataclass(frozen=True)
class VehicleTruth:
    x: float
    y: float
    heading: float
    speed: float
    acceleration: float


@dataclass(frozen=True)
class ObstacleTruth:
    obstacle_id: str
    category: str
    shape: str
    x: float
    y: float
    vx: float
    vy: float
    width: float
    height: float


@dataclass(frozen=True)
class WorldState:
    time: float
    vehicle: VehicleTruth
    obstacles: tuple[ObstacleTruth, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def build_world_state(
    time: float, vehicle: VehicleState, obstacles: list[Obstacle]
) -> WorldState:
    truths = []
    for o in obstacles:
        w, h = o.shape.extent
        truths.append(
            ObstacleTruth(
                o.obstacle_id, o.category, o.shape.kind, o.x, o.y, o.vx, o.vy, w, h
            )
        )
    return WorldState(
        time=time,
        vehicle=VehicleTruth(
            vehicle.x, vehicle.y, vehicle.heading, vehicle.speed, vehicle.acceleration
        ),
        obstacles=tuple(truths),
    )
