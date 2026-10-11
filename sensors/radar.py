"""Virtual radar: long range, direct radial-velocity measurement and works in
poor visibility, but coarse angular accuracy and prone to ghost targets."""

from __future__ import annotations

import math
from typing import Any

from sensors.base import BaseSensor, SensorConfig
from sensors.faults import FaultEffects
from simulation.world_state import ObstacleTruth, WorldState

RADAR_CONFIG = SensorConfig(
    name="radar",
    rate_hz=20.0,
    max_range=80.0,
    fov_deg=120.0,
    range_sigma0=0.4,
    range_sigma_per_m=0.005,
    bearing_sigma_deg=3.0,
    base_dropout=0.03,
    far_dropout=0.1,
    base_ghost_rate=0.02,
    occludable=False,  # radar waves partly pass or bounce around obstacles
)


class RadarSensor(BaseSensor):
    DEFAULT_CONFIG = RADAR_CONFIG
    RADIAL_VELOCITY_SIGMA = 0.15  # m/s

    def _extra_fields(
        self,
        target: ObstacleTruth,
        true_range: float,
        fx: FaultEffects,
        world: WorldState,
    ) -> dict[str, Any]:
        veh = world.vehicle
        # Relative velocity of the target as seen from the moving vehicle.
        rel_vx = target.vx - veh.speed * math.cos(veh.heading)
        rel_vy = target.vy - veh.speed * math.sin(veh.heading)
        los_x = (target.x - veh.x) / max(true_range, 1e-9)
        los_y = (target.y - veh.y) / max(true_range, 1e-9)
        radial = rel_vx * los_x + rel_vy * los_y  # + = moving away
        sigma = self.RADIAL_VELOCITY_SIGMA * fx.noise_multiplier
        return {"radial_velocity": float(radial + self.rng.normal(0.0, sigma))}

    def _ghost_fields(self) -> dict[str, Any]:
        return {"radial_velocity": float(self.rng.normal(0.0, 3.0))}
