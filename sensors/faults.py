"""Fault injection: describe *what* goes wrong, *when*, and *how badly*.

A FaultSpec has a severity that ramps from 0 to 1, so sensors can degrade
gradually (dust building up on a lens) or suddenly (ramp = 0).

    severity(t) = 0                       before ``start`` or after ``end``
                = min(1, (t-start)/ramp)  while active
    effect      = severity * level

Meaning of ``level`` for each fault type:
    noise     noise standard deviation is multiplied by (1 + level*severity)
    dropout   probability that a visible object is NOT reported (0..1)
    outlier   probability that a reported object has a gross error (0..1)
    delay     extra delivery delay in seconds
    fov       fraction of the field of view lost (0..1)
    range     fraction of the maximum range lost (0..1)
    bias      bearing mis-calibration in degrees
    ghost     mean number of false detections added per frame
    freeze    probability the sensor repeats its previous (stale) frame (0..1)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

SENSOR_NAMES = ("camera", "lidar", "radar")


class FaultType(str, Enum):
    NOISE = "noise"
    DROPOUT = "dropout"
    OUTLIER = "outlier"
    DELAY = "delay"
    FOV = "fov"
    RANGE = "range"
    BIAS = "bias"
    GHOST = "ghost"
    FREEZE = "freeze"


_LEVEL_LIMITS: dict[FaultType, tuple[float, float]] = {
    FaultType.NOISE: (0.0, 50.0),
    FaultType.DROPOUT: (0.0, 1.0),
    FaultType.OUTLIER: (0.0, 1.0),
    FaultType.DELAY: (0.0, 5.0),
    FaultType.FOV: (0.0, 1.0),
    FaultType.RANGE: (0.0, 1.0),
    FaultType.BIAS: (-30.0, 30.0),
    FaultType.GHOST: (0.0, 20.0),
    FaultType.FREEZE: (0.0, 1.0),
}


@dataclass(frozen=True)
class FaultSpec:
    sensor: str
    fault: FaultType
    level: float
    start: float = 0.0
    end: float | None = None
    ramp: float = 0.0

    def __post_init__(self) -> None:
        if self.sensor not in SENSOR_NAMES:
            raise ValueError(
                f"sensor must be one of {SENSOR_NAMES}, got '{self.sensor}'"
            )
        low, high = _LEVEL_LIMITS[self.fault]
        if not low <= self.level <= high:
            raise ValueError(
                f"level for '{self.fault.value}' must be in [{low}, {high}], "
                f"got {self.level}"
            )
        if self.start < 0 or self.ramp < 0:
            raise ValueError("start and ramp must be >= 0")
        if self.end is not None and self.end <= self.start:
            raise ValueError("end must be greater than start")

    def severity(self, t: float) -> float:
        if t < self.start or (self.end is not None and t >= self.end):
            return 0.0
        if self.ramp <= 0:
            return 1.0
        return min(1.0, (t - self.start) / self.ramp)


@dataclass(frozen=True)
class FaultEffects:
    """Combined effect of every fault active on one sensor at one time."""

    noise_multiplier: float = 1.0
    dropout_prob: float = 0.0
    outlier_prob: float = 0.0
    delay_s: float = 0.0
    fov_loss: float = 0.0
    range_loss: float = 0.0
    bearing_bias_rad: float = 0.0
    ghost_rate: float = 0.0
    freeze_prob: float = 0.0
    severity: float = 0.0  # strongest active severity (0..1)
    active: tuple[str, ...] = ()


def _union(probabilities: list[float]) -> float:
    """P(at least one event) for independent probabilities."""
    remaining = 1.0
    for p in probabilities:
        remaining *= 1.0 - p
    return 1.0 - remaining


class FaultSchedule:
    """A named collection of FaultSpecs (one 'fault profile')."""

    def __init__(
        self,
        specs: list[FaultSpec] | None = None,
        name: str = "nominal",
        description: str = "",
    ) -> None:
        self.specs = list(specs or [])
        self.name = name
        self.description = description

    def effects_at(self, sensor: str, t: float) -> FaultEffects:
        noise = 1.0
        buckets: dict[FaultType, list[float]] = {ft: [] for ft in FaultType}
        delay = bias = ghost = severity = 0.0
        active: list[str] = []
        for spec in self.specs:
            if spec.sensor != sensor:
                continue
            sev = spec.severity(t)
            if sev <= 0.0:
                continue
            active.append(spec.fault.value)
            severity = max(severity, sev)
            amount = spec.level * sev
            if spec.fault is FaultType.NOISE:
                noise *= 1.0 + amount
            elif spec.fault is FaultType.DELAY:
                delay += amount
            elif spec.fault is FaultType.BIAS:
                bias += amount
            elif spec.fault is FaultType.GHOST:
                ghost += amount
            else:
                buckets[spec.fault].append(amount)
        return FaultEffects(
            noise_multiplier=noise,
            dropout_prob=_union(buckets[FaultType.DROPOUT]),
            outlier_prob=_union(buckets[FaultType.OUTLIER]),
            delay_s=delay,
            fov_loss=_union(buckets[FaultType.FOV]),
            range_loss=_union(buckets[FaultType.RANGE]),
            bearing_bias_rad=math.radians(bias),
            ghost_rate=ghost,
            freeze_prob=_union(buckets[FaultType.FREEZE]),
            severity=severity,
            active=tuple(sorted(set(active))),
        )
