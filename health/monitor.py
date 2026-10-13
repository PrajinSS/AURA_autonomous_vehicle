"""Stateful sensor-health monitor: indicators -> scores -> reliability -> status."""

from __future__ import annotations

import bisect
import math
from typing import Any

from health.config import INDICATORS, HealthConfig
from health.indicators import EgoPose, FrameStore, compute_indicators
from health.report import HealthReport, HealthStatus, SensorHealth
from sensors.base import SensorConfig
from sensors.camera import CAMERA_CONFIG
from sensors.detection import SensorFrame
from sensors.lidar import LIDAR_CONFIG
from sensors.radar import RADAR_CONFIG

DEFAULT_SPECS = {c.name: c for c in (CAMERA_CONFIG, LIDAR_CONFIG, RADAR_CONFIG)}

REASONS = {
    "completeness": "misses objects other sensors agree on (hit rate {v:.0%})",
    "support": "{u:.0%} of its detections are not confirmed by other sensors",
    "persistence": "only {v:.0%} of detections persist from frame to frame",
    "noise": "position noise is {v:.1f}x the nominal level",
    "bias": "systematic bearing offset of about {v:.1f} degrees",
    "freeze": "repeats identical data in {v:.0%} of frames",
    "latency": "data arrives {v:.2f} s late",
    "stream": "no new data for {v:.1f} sensor periods",
    "intrinsic": "own detection confidence is low ({v:.2f})",
}


class SensorHealthMonitor:
    """Feed it delivered frames and the vehicle's own pose; read health reports.

    It never sees ``truth_*`` fields or the simulator's ground truth.
    """

    def __init__(
        self,
        config: HealthConfig | None = None,
        specs: dict[str, SensorConfig] | None = None,
        static_map: Any = None,
    ) -> None:
        self.config = config or HealthConfig()
        self.specs = specs or DEFAULT_SPECS
        self.static_map = static_map  # OccupancyGrid of known static obstacles, or None
        self._stores = {name: FrameStore() for name in self.specs}
        self._ego: list[tuple[float, float, float, float]] = []  # t, x, y, heading
        self._ego_times: list[float] = []
        self._reliability = {name: 1.0 for name in self.specs}
        self._status = {name: HealthStatus.HEALTHY for name in self.specs}
        self._last_reasons: dict[str, tuple[str, ...]] = {n: () for n in self.specs}
        self._next_eval = 0.0
        self._last_eval = 0.0
        self.latest: HealthReport | None = None
        self.history: list[HealthReport] = []

    # ---- public API -----------------------------------------------------------
    def update(
        self, frames: list[SensorFrame], ego: Any, now: float
    ) -> HealthReport | None:
        """Ingest data. Returns a new report when one is due, else None."""
        for frame in frames:
            if frame.sensor in self._stores:
                self._stores[frame.sensor].add(frame)
        self._ego.append((now, ego.x, ego.y, ego.heading))
        self._ego_times.append(now)

        if now + 1e-9 < self._next_eval:
            return None
        self._next_eval += 1.0 / self.config.eval_rate_hz
        self._prune(now)
        report = self._evaluate(now)
        self.latest = report
        self.history.append(report)
        return report

    # ---- internals ----------------------------------------------------------------
    def _prune(self, now: float) -> None:
        cutoff = now - self.config.window_s - 1.0
        for store in self._stores.values():
            store.prune(cutoff)
        cut = bisect.bisect_left(self._ego_times, cutoff)
        del self._ego[:cut], self._ego_times[:cut]

    def _ego_at(self, t: float) -> EgoPose | None:
        if not self._ego:
            return None
        i = bisect.bisect_left(self._ego_times, t)
        j = min(
            range(max(0, i - 1), min(len(self._ego), i + 1)),
            key=lambda k: abs(self._ego_times[k] - t),
        )
        _, x, y, heading = self._ego[j]
        return (x, y, heading)

    def _evaluate(self, now: float) -> HealthReport:
        dt = max(now - self._last_eval, 1e-6)
        self._last_eval = now
        sensors = {}
        # Only sensors that are currently healthy may act as witnesses for others.
        trusted = {n for n, st in self._status.items() if st is HealthStatus.HEALTHY}
        strong = {n for n in trusted if self._reliability[n] >= 0.9}
        for name in self.specs:
            ind = compute_indicators(
                name,
                self._stores,
                self.specs,
                self._ego_at,
                now,
                self.config,
                trusted,
                self.static_map,
                strong,
            )
            sensors[name] = self._assess(name, ind, dt)
        return HealthReport(now, sensors)

    def _assess(self, name: str, ind: dict, dt: float) -> SensorHealth:
        cfg = self.config
        scores: dict[str, float | None] = {}
        confidences: dict[str, float] = {}
        for key in INDICATORS:
            cal = cfg.calibrations[key]
            value = ind[key].value
            ok = value is not None and (
                key == "stream" or ind[key].evidence >= cfg.min_evidence
            )
            scores[key] = (
                cal.score(cal.shrink(value, ind[key].evidence)) if ok else None
            )
            confidences[key] = (
                1.0 - math.exp(-ind[key].evidence / cal.evidence_scale) if ok else 0.0
            )
        used = {k: s for k, s in scores.items() if s is not None and cfg.weights[k] > 0}
        weight_all = sum(cfg.weights.values())
        confidence = sum(cfg.weights[k] * confidences[k] for k in used) / weight_all
        prev = self._reliability[name]
        if used and confidence >= cfg.min_confidence:
            w_used = sum(cfg.weights[k] for k in used)
            harmonic = w_used / sum(
                cfg.weights[k] / max(s, cfg.score_floor) for k, s in used.items()
            )
            raw = min(harmonic, min(used.values()) + cfg.single_fault_margin, 1.0)
            tau = cfg.tau_down_s if raw < prev else cfg.tau_up_s
            reliability = prev + (1.0 - math.exp(-dt / tau)) * (raw - prev)
        else:
            raw, reliability = prev, prev  # not enough evidence: hold the last value
        self._reliability[name] = reliability
        status = self._next_status(self._status[name], reliability)
        self._status[name] = status
        reasons = self._reasons(scores, ind)
        if not reasons and status is not HealthStatus.HEALTHY:
            # No fresh evidence: keep explaining the held assessment.
            reasons = self._last_reasons[name] or ("holding last assessment",)
            reasons = tuple(
                r if r.startswith("(held)") else f"(held) {r}" for r in reasons
            )
        if reasons and not reasons[0].startswith("(held)"):
            self._last_reasons[name] = reasons
        return SensorHealth(
            sensor=name,
            reliability=reliability,
            raw_reliability=raw,
            confidence=confidence,
            status=status,
            scores=scores,
            values={k: ind[k].value for k in INDICATORS},
            evidence={k: ind[k].evidence for k in INDICATORS},
            reasons=reasons,
        )

    def _next_status(self, prev: HealthStatus, q: float) -> HealthStatus:
        cfg = self.config
        if q < cfg.failed_below:
            return HealthStatus.FAILED
        if prev is HealthStatus.FAILED and q < cfg.failed_below + cfg.hysteresis:
            return HealthStatus.FAILED
        if q < cfg.degraded_below:
            return HealthStatus.DEGRADED
        if prev is not HealthStatus.HEALTHY and q < cfg.degraded_below + cfg.hysteresis:
            return HealthStatus.DEGRADED
        return HealthStatus.HEALTHY

    @staticmethod
    def _reasons(scores: dict, ind: dict) -> tuple[str, ...]:
        weak = sorted((s, k) for k, s in scores.items() if s is not None and s < 0.6)
        out = []
        for _, key in weak[:3]:
            value = ind[key].value
            unsupported = 1.0 - value if key == "support" else 0.0
            out.append(REASONS[key].format(v=value, u=unsupported))
        return tuple(out)
