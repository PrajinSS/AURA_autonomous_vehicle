"""Kinematic bicycle-model vehicle with acceleration limits."""

from __future__ import annotations

import math
from dataclasses import dataclass

from config import VehicleParams
from simulation.geometry import clamp, wrap_angle


@dataclass
class VehicleState:
    """Pose and speed of the vehicle (heading in radians, speed in m/s)."""

    x: float
    y: float
    heading: float = 0.0
    speed: float = 0.0
    acceleration: float = 0.0  # last applied, m/s^2


@dataclass(frozen=True)
class VehicleCommand:
    """High-level command produced by a controller or planner."""

    target_speed: float
    steering: float


class Vehicle:
    """Integrates the kinematic bicycle model one step at a time."""

    def __init__(self, state: VehicleState, params: VehicleParams | None = None):
        self.state = state
        self.params = params or VehicleParams()
        self.distance_travelled = 0.0

    def step(self, command: VehicleCommand, dt: float) -> None:
        if dt <= 0:
            raise ValueError("dt must be > 0.")
        p, s = self.params, self.state
        target = clamp(command.target_speed, 0.0, p.max_speed)
        steering = clamp(command.steering, -p.max_steer, p.max_steer)
        previous_speed = s.speed
        delta_v = target - s.speed
        if delta_v >= 0.0:
            delta_v = min(delta_v, p.max_accel * dt)
        else:
            delta_v = max(delta_v, -p.max_brake * dt)
        s.speed = max(0.0, s.speed + delta_v)
        s.acceleration = (s.speed - previous_speed) / dt
        s.x += s.speed * math.cos(s.heading) * dt
        s.y += s.speed * math.sin(s.heading) * dt
        s.heading = wrap_angle(
            s.heading + (s.speed / p.wheelbase) * math.tan(steering) * dt
        )
        self.distance_travelled += s.speed * dt
