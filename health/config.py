"""Settings of the sensor-health engine.

Every number that decides "healthy" or "faulty" lives here. The calibration
values were measured on healthy (nominal) simulation runs with
``calibrate_health.py``; re-run that script if you change a sensor model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

INDICATORS = (
    "completeness",
    "support",
    "persistence",
    "noise",
    "bias",
    "freeze",
    "latency",
    "stream",
    "intrinsic",
)


@dataclass(frozen=True)
class Calibration:
    """Maps a raw indicator value to a score in [0, 1].

    value == nominal -> score 1 (healthy); value == floor -> score 0 (failed).
    It works for both directions: "higher is better" (nominal > floor) and
    "lower is better" (nominal < floor).
    """

    nominal: float
    floor: float
    evidence_scale: float = 10.0  # observations that give ~63 % confidence
    prior_strength: float = 0.0  # pseudo-observations at 'nominal' (shrinkage)

    def __post_init__(self) -> None:
        if self.nominal == self.floor:
            raise ValueError("nominal and floor must differ")

    def shrink(self, value: float, evidence: int) -> float:
        """Pull a value towards nominal when it rests on few observations.

        With n observations: (n*value + k*nominal) / (n + k), k = prior_strength.
        A single unlucky reading can then not condemn a healthy sensor.
        """
        k = self.prior_strength
        return (
            (evidence * value + k * self.nominal) / (evidence + k) if k > 0 else value
        )

    def score(self, value: float) -> float:
        return float(
            np.clip((value - self.floor) / (self.nominal - self.floor), 0.0, 1.0)
        )


def _default_calibrations() -> dict[str, Calibration]:
    """Measured on healthy runs of all five scenarios (see calibrate_health.py)."""
    c = Calibration
    return {
        "completeness": c(
            nominal=0.90, floor=0.30, evidence_scale=15, prior_strength=8
        ),
        "support": c(nominal=0.90, floor=0.40, evidence_scale=25, prior_strength=12),
        "persistence": c(
            nominal=0.90, floor=0.62, evidence_scale=25, prior_strength=10
        ),
        "noise": c(nominal=1.3, floor=4.0, evidence_scale=15, prior_strength=6),
        "bias": c(nominal=0.5, floor=3.0, evidence_scale=15, prior_strength=6),
        "freeze": c(nominal=0.0, floor=0.3, evidence_scale=8, prior_strength=4),
        "latency": c(nominal=0.0, floor=0.25, evidence_scale=8, prior_strength=4),
        "stream": c(nominal=2.0, floor=10.0, evidence_scale=1),
        "intrinsic": c(nominal=0.88, floor=0.55, evidence_scale=15, prior_strength=6),
    }


def _default_weights() -> dict[str, float]:
    return {
        "completeness": 0.25,
        "support": 0.15,
        "persistence": 0.10,
        "noise": 0.12,
        "bias": 0.08,
        "freeze": 0.12,
        "latency": 0.10,
        "stream": 0.08,
        "intrinsic": 0.10,
    }


@dataclass(frozen=True)
class HealthConfig:
    window_s: float = 2.0  # how much recent history is judged
    eval_rate_hz: float = 5.0  # how often a new report is produced
    coverage_margin: float = 0.9  # only judge objects well inside a sensor's range/FOV
    min_gate_m: float = 1.5  # smallest matching distance between two detections
    gate_sigmas: float = 3.0  # matching distance in standard deviations
    min_evidence: int = 3  # fewer observations than this: indicator ignored
    single_fault_margin: float = 0.35  # reliability <= weakest score + margin
    degraded_below: float = 0.65
    failed_below: float = 0.30
    hysteresis: float = 0.10  # extra margin needed to recover to a better status
    tau_down_s: float = 0.4  # reliability falls quickly ...
    tau_up_s: float = 1.5  # ... and recovers slowly
    min_confidence: float = 0.15  # below this the previous reliability is held
    score_floor: float = 0.05  # avoids division by zero in the harmonic mean
    calibrations: dict[str, Calibration] = field(default_factory=_default_calibrations)
    weights: dict[str, float] = field(default_factory=_default_weights)

    def __post_init__(self) -> None:
        if set(self.calibrations) != set(INDICATORS) or set(self.weights) != set(
            INDICATORS
        ):
            raise ValueError(f"calibrations and weights need exactly: {INDICATORS}")
        if not 0 < self.failed_below < self.degraded_below < 1:
            raise ValueError("need 0 < failed_below < degraded_below < 1")
        if min(self.window_s, self.eval_rate_hz, self.tau_down_s, self.tau_up_s) <= 0:
            raise ValueError("window, rate and time constants must be > 0")
