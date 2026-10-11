"""Common sensor behaviour: scheduling, visibility, noise, faults, delay.

Concrete sensors (camera, LiDAR, radar) only define their configuration and
the extra fields they report (class label, radial velocity, ...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from sensors.detection import Detection, SensorFrame
from sensors.faults import FaultEffects, FaultSchedule
from sensors.geometry import line_of_sight_blocked, polar, polar_to_world
from simulation.geometry import wrap_angle
from simulation.world_state import ObstacleTruth, WorldState

OUTLIER_RANGE_ERROR = (5.0, 15.0)  # metres, gross range error of an outlier
OUTLIER_BEARING_ERROR = 0.26  # rad (about 15 degrees)


@dataclass(frozen=True)
class SensorConfig:
    name: str
    rate_hz: float
    max_range: float
    fov_deg: float
    range_sigma0: float  # range noise std at zero distance (m)
    range_sigma_per_m: float  # extra range noise std per metre of distance
    bearing_sigma_deg: float
    base_dropout: float = 0.0  # chance of missing a visible object
    far_dropout: float = 0.0  # extra miss chance near the range limit
    base_ghost_rate: float = 0.0  # false alarms per frame, even when healthy
    occludable: bool = True  # blocked by objects in the line of sight

    def __post_init__(self) -> None:
        if min(self.rate_hz, self.max_range, self.fov_deg) <= 0:
            raise ValueError("rate_hz, max_range and fov_deg must be > 0")
        if self.fov_deg > 360:
            raise ValueError("fov_deg must be <= 360")
        if min(self.range_sigma0, self.range_sigma_per_m, self.bearing_sigma_deg) < 0:
            raise ValueError("noise parameters must be >= 0")
        if not 0 <= self.base_dropout <= 1 or not 0 <= self.far_dropout <= 1:
            raise ValueError("dropout probabilities must be in [0, 1]")


class BaseSensor:
    DEFAULT_CONFIG: SensorConfig

    def __init__(
        self,
        faults: FaultSchedule | None = None,
        rng: np.random.Generator | None = None,
        config: SensorConfig | None = None,
    ) -> None:
        self.config = config or self.DEFAULT_CONFIG
        self.faults = faults or FaultSchedule()
        self.rng = rng or np.random.default_rng(0)
        self._period = 1.0 / self.config.rate_hz
        self._next_time = 0.0
        self._last_detections: tuple[Detection, ...] | None = None

    @property
    def name(self) -> str:
        return self.config.name

    # ---- hooks for concrete sensors ----------------------------------------
    def _extra_fields(
        self,
        target: ObstacleTruth,
        true_range: float,
        fx: FaultEffects,
        world: WorldState,
    ) -> dict[str, Any]:
        return {}

    def _ghost_fields(self) -> dict[str, Any]:
        return {}

    # ---- main entry point ---------------------------------------------------
    def sample(self, world: WorldState) -> SensorFrame | None:
        """Return a frame if the sensor is due to measure at ``world.time``."""
        t = world.time
        if t + 1e-9 < self._next_time:
            return None
        while self._next_time <= t + 1e-9:
            self._next_time += self._period

        cfg, veh = self.config, world.vehicle
        fx = self.faults.effects_at(self.name, t)
        max_range = cfg.max_range * (1.0 - fx.range_loss)
        half_fov = math.radians(cfg.fov_deg) / 2.0
        eff_half_fov = half_fov * (1.0 - fx.fov_loss)

        expected: list[str] = []
        detections: list[Detection] = []
        for target in world.obstacles:
            true_range, bearing = polar(veh.x, veh.y, veh.heading, target.x, target.y)
            if true_range > cfg.max_range or abs(bearing) > half_fov:
                continue
            if cfg.occludable:
                blockers = [o for o in world.obstacles if o is not target]
                if line_of_sight_blocked(veh.x, veh.y, target, blockers):
                    continue
            expected.append(target.obstacle_id)  # visible if the sensor were healthy
            if true_range > max_range or abs(bearing) > eff_half_fov:
                continue
            p_base = cfg.base_dropout + cfg.far_dropout * (true_range / max_range) ** 4
            p_miss = 1.0 - (1.0 - min(p_base, 1.0)) * (1.0 - fx.dropout_prob)
            if self.rng.random() < p_miss:
                continue
            detections.append(self._measure(target, true_range, bearing, fx, world))

        detections.extend(self._ghosts(fx, max_range, eff_half_fov, world))

        frozen = False
        if (
            self._last_detections is not None
            and fx.freeze_prob > 0.0
            and self.rng.random() < fx.freeze_prob
        ):
            detections = list(self._last_detections)  # stale data, fresh header
            frozen = True
        else:
            self._last_detections = tuple(detections)

        return SensorFrame(
            sensor=self.name,
            timestamp=t,
            arrival_time=t + fx.delay_s,
            detections=tuple(detections),
            truth_expected_ids=tuple(expected),
            truth_frozen=frozen,
            truth_active_faults=fx.active,
            truth_severity=fx.severity,
        )

    # ---- internals -----------------------------------------------------------
    def _measure(
        self,
        target: ObstacleTruth,
        true_range: float,
        bearing: float,
        fx: FaultEffects,
        world: WorldState,
    ) -> Detection:
        cfg, veh, rng = self.config, world.vehicle, self.rng
        sigma_r = (
            cfg.range_sigma0 + cfg.range_sigma_per_m * true_range
        ) * fx.noise_multiplier
        sigma_b = math.radians(cfg.bearing_sigma_deg) * fx.noise_multiplier
        meas_range = max(0.1, true_range + rng.normal(0.0, sigma_r))
        meas_bearing = bearing + rng.normal(0.0, sigma_b) + fx.bearing_bias_rad
        fault = ""
        if rng.random() < fx.outlier_prob:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            meas_range = max(0.1, meas_range + sign * rng.uniform(*OUTLIER_RANGE_ERROR))
            meas_bearing += rng.uniform(-OUTLIER_BEARING_ERROR, OUTLIER_BEARING_ERROR)
            fault = "outlier"
        meas_bearing = wrap_angle(meas_bearing)
        x, y = polar_to_world(veh.x, veh.y, veh.heading, meas_range, meas_bearing)
        return Detection(
            sensor=self.name,
            timestamp=world.time,
            x=x,
            y=y,
            range_m=meas_range,
            bearing_rad=meas_bearing,
            truth_id=target.obstacle_id,
            truth_x=target.x,
            truth_y=target.y,
            truth_fault=fault,
            **self._extra_fields(target, true_range, fx, world),
        )

    def _ghosts(
        self, fx: FaultEffects, max_range: float, half_fov: float, world: WorldState
    ) -> list[Detection]:
        rate = self.config.base_ghost_rate + fx.ghost_rate
        if rate <= 0.0 or max_range <= 2.0:
            return []
        veh, rng = world.vehicle, self.rng
        ghosts = []
        for _ in range(int(rng.poisson(rate))):
            g_range = rng.uniform(2.0, max_range)
            g_bearing = rng.uniform(-half_fov, half_fov)
            x, y = polar_to_world(veh.x, veh.y, veh.heading, g_range, g_bearing)
            ghosts.append(
                Detection(
                    sensor=self.name,
                    timestamp=world.time,
                    x=x,
                    y=y,
                    range_m=g_range,
                    bearing_rad=g_bearing,
                    truth_fault="ghost",
                    **self._ghost_fields(),
                )
            )
        return ghosts
