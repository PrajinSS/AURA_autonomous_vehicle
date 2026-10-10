"""Central, typed and validated configuration for the Phase 1 simulator.

Every tunable number lives here so experiments stay repeatable and no
"magic numbers" are scattered through the code base.
"""

from __future__ import annotations

from dataclasses import dataclass


def _require_positive(owner: str, **values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{owner}.{name} must be > 0, got {value}")


@dataclass(frozen=True)
class VehicleParams:
    """Physical limits of the industrial vehicle (SI units: m, s, rad)."""

    wheelbase: float = 2.5
    radius: float = 1.2  # circular footprint used for collision checks
    max_speed: float = 4.0
    max_accel: float = 1.5
    max_brake: float = 4.0
    max_steer: float = 0.6

    def __post_init__(self) -> None:
        _require_positive(
            "VehicleParams",
            wheelbase=self.wheelbase,
            radius=self.radius,
            max_speed=self.max_speed,
            max_accel=self.max_accel,
            max_brake=self.max_brake,
            max_steer=self.max_steer,
        )


@dataclass(frozen=True)
class ControllerParams:
    """Tuning of the Phase 1 reactive baseline controller."""

    k_steer: float = 1.8
    influence_gap: float = 8.0  # obstacles farther than this are ignored
    repulsion_gain: float = 2.5
    tangential_angle: float = 0.5  # rad; makes the vehicle pick a side
    slow_gap: float = 5.0  # start slowing when clearance drops below this
    min_speed_near_obstacle: float = 0.8
    comfort_decel: float = 1.0  # used to brake smoothly towards the goal
    front_cone: float = 1.75  # rad; obstacles outside this cone do not slow us

    def __post_init__(self) -> None:
        _require_positive(
            "ControllerParams",
            k_steer=self.k_steer,
            influence_gap=self.influence_gap,
            repulsion_gain=self.repulsion_gain,
            slow_gap=self.slow_gap,
            comfort_decel=self.comfort_decel,
        )


@dataclass(frozen=True)
class SimulationParams:
    """Time stepping, termination and metric settings."""

    dt: float = 0.05
    max_time: float = 120.0
    near_miss_gap: float = 1.0  # clearance (m) below which we count a near miss
    seed: int = 42

    def __post_init__(self) -> None:
        _require_positive(
            "SimulationParams",
            dt=self.dt,
            max_time=self.max_time,
            near_miss_gap=self.near_miss_gap,
        )
