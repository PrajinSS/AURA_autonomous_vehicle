"""Result objects of the health engine (what other modules consume)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass(frozen=True)
class SensorHealth:
    """Health of one sensor at one time.

    reliability: smoothed score in [0, 1]; later phases use it as a fusion weight.
    confidence: how much evidence supports the score (0 = pure guess).
    """

    sensor: str
    reliability: float
    raw_reliability: float
    confidence: float
    status: HealthStatus
    scores: dict[str, float | None] = field(default_factory=dict)
    values: dict[str, float | None] = field(default_factory=dict)
    evidence: dict[str, int] = field(default_factory=dict)
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class HealthReport:
    time: float
    sensors: dict[str, SensorHealth]

    def reliabilities(self) -> dict[str, float]:
        return {name: h.reliability for name, h in self.sensors.items()}

    def statuses(self) -> dict[str, HealthStatus]:
        return {name: h.status for name, h in self.sensors.items()}
