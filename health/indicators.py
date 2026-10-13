"""Raw health indicators, computed ONLY from delivered sensor data.

Nothing here may read ``truth_*`` fields. The engine judges a sensor the way
a human would: by comparing it with the other sensors and by checking whether
its data stream behaves sensibly over time.

Indicators (value, number of observations):
 completeness share of objects that two OTHER sensors agree on (inside this
 sensor's coverage) that this sensor also reported
 support share of this sensor's detections confirmed by another sensor
 persistence share of detections that also appeared in the previous frame
 (needs no other sensor: ghosts and wild jumps do not persist)
 noise size of position disagreement with consensus, in units of the
 nominal expected disagreement (about 1 when healthy)
 bias significant average bearing offset versus consensus (degrees)
 freeze share of consecutive frames that repeat identical data
 latency average delivery delay (seconds)
 stream time since the last frame, in sensor periods
 intrinsic sensor's own confidence (camera class confidence)
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from health.config import HealthConfig
from sensors.base import SensorConfig
from sensors.detection import Detection, SensorFrame
from sensors.geometry import polar
from simulation.geometry import wrap_angle

EgoPose = tuple[float, float, float]  # x, y, heading


@dataclass(frozen=True)
class Indicator:
    value: float | None
    evidence: int = 0


class FrameStore:
    """Time-ordered frames of one sensor with fast nearest-in-time lookup."""

    def __init__(self) -> None:
        self.frames: list[SensorFrame] = []
        self.times: list[float] = []
        self.last_arrival: float | None = None

    def add(self, frame: SensorFrame) -> None:
        idx = bisect.bisect_right(
            self.times, frame.timestamp
        )  # delayed frames may be late
        self.times.insert(idx, frame.timestamp)
        self.frames.insert(idx, frame)
        if self.last_arrival is None or frame.arrival_time > self.last_arrival:
            self.last_arrival = frame.arrival_time

    def prune(self, cutoff: float) -> None:
        cut = bisect.bisect_left(self.times, cutoff)
        del self.times[:cut], self.frames[:cut]

    @property
    def last_time(self) -> float | None:
        return self.times[-1] if self.times else None

    def since(self, t0: float) -> list[SensorFrame]:
        return self.frames[bisect.bisect_left(self.times, t0) :]

    def nearest(self, t: float, tol: float) -> SensorFrame | None:
        if not self.times:
            return None
        i = bisect.bisect_left(self.times, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.times) and abs(self.times[j] - t) <= tol:
                if best is None or abs(self.times[j] - t) < abs(self.times[best] - t):
                    best = j
        return None if best is None else self.frames[best]


# ---- geometry helpers ---------------------------------------------------------
def sigma_pos(spec: SensorConfig, r: float) -> float:
    """Nominal position standard deviation (m) of a sensor at distance r."""
    along = spec.range_sigma0 + spec.range_sigma_per_m * r
    across = r * math.radians(spec.bearing_sigma_deg)
    return math.hypot(along, across)


def in_coverage(
    spec: SensorConfig, ego: EgoPose, x: float, y: float, margin: float
) -> bool:
    rng, bearing = polar(ego[0], ego[1], ego[2], x, y)
    if rng > spec.max_range * margin:
        return False
    return (
        spec.fov_deg >= 360.0 or abs(bearing) <= math.radians(spec.fov_deg) / 2 * margin
    )


def gate(a: SensorConfig, b: SensorConfig, r: float, cfg: HealthConfig) -> float:
    return max(
        cfg.min_gate_m, cfg.gate_sigmas * math.hypot(sigma_pos(a, r), sigma_pos(b, r))
    )


def nearest_within(
    x: float, y: float, detections: tuple[Detection, ...], limit: float
) -> Detection | None:
    best, best_d = None, limit
    for det in detections:
        dist = math.hypot(det.x - x, det.y - y)
        if dist <= best_d:
            best, best_d = det, dist
    return best


# ---- line of sight against the vehicle's static map ------------------------------
def los_clear(grid: Any, ego: EgoPose, x: float, y: float, step: float = 0.5) -> bool:
    """True if nothing in the known static map blocks the view from ego to (x, y).

    A static object is allowed to be its own target: if (x, y) lies inside a
    blocked region, only blocked cells *before* the last free cell count.
    """
    if grid is None:
        return True
    dist = math.hypot(x - ego[0], y - ego[1])
    n = int(dist / step)
    if n < 2:
        return True
    fractions = np.append(np.arange(1, n + 1) * step / dist, 1.0)
    xs = ego[0] + fractions * (x - ego[0])
    ys = ego[1] + fractions * (y - ego[1])
    cols = np.floor((xs - grid.x_min) / grid.resolution).astype(int)
    rows = np.floor((ys - grid.y_min) / grid.resolution).astype(int)
    rows_n, cols_n = grid.data.shape
    inside = (rows >= 0) & (rows < rows_n) & (cols >= 0) & (cols < cols_n)
    blocked = np.zeros(len(xs), dtype=bool)
    blocked[inside] = grid.data[rows[inside], cols[inside]]
    if blocked[-1]:  # target is a static object: ignore its own body
        free = np.flatnonzero(~blocked[:-1])
        return not (free.size and blocked[: free[-1]].any())
    return not blocked.any()


def can_observe(
    spec: SensorConfig, ego: EgoPose, x: float, y: float, cfg: HealthConfig, grid: Any
) -> bool:
    """Could a healthy sensor with this spec see a point at (x, y) right now?"""
    if not in_coverage(spec, ego, x, y, cfg.coverage_margin):
        return False
    return not spec.occludable or los_clear(grid, ego, x, y)


# ---- the indicators -----------------------------------------------------------------
@dataclass
class _Acc:
    support_n: int = 0
    support_ok: int = 0
    ref_n: int = 0
    ref_hit: int = 0
    pair_z2: dict[str, list[float]] = field(default_factory=dict)
    pair_bearing: dict[str, list[float]] = field(default_factory=dict)


def compute_indicators(
    sensor: str,
    stores: dict[str, FrameStore],
    specs: dict[str, SensorConfig],
    ego_at: Callable[[float], EgoPose | None],
    now: float,
    cfg: HealthConfig,
    trusted: set[str] | None = None,
    static_map: Any = None,
    strong: set[str] | None = None,
) -> dict[str, Indicator]:
    """Raw indicators for one sensor.

    ``trusted`` = sensors allowed to act as witnesses (default: all others). A
    degraded sensor must not judge the others, or innocent sensors would be
    blamed for its faults. ``static_map`` (an occupancy grid of the known static
    obstacles) tells us what a camera or LiDAR could physically see.
    """
    spec, store = specs[sensor], stores[sensor]
    strong = strong if strong is not None else trusted
    witnesses = [n for n in specs if n != sensor and (trusted is None or n in trusted)]
    tol = {n: 0.6 / specs[n].rate_hz for n in witnesses}
    frames = store.since(store.last_time - cfg.window_s) if store.frames else []
    acc = _Acc()
    for frame in frames:
        ego = ego_at(frame.timestamp)
        if ego is None:
            continue
        near = {w: stores[w].nearest(frame.timestamp, tol[w]) for w in witnesses}
        near = {w: fr for w, fr in near.items() if fr is not None}
        _support(frame, near, spec, specs, ego, cfg, static_map, acc)
        _pairwise(frame, near, spec, specs, ego, cfg, static_map, acc)
        _references(
            frame, near, witnesses, spec, specs, ego, cfg, static_map, acc, strong
        )
    return {
        "completeness": _ratio(acc.ref_hit, acc.ref_n, cfg),
        "support": _ratio(acc.support_ok, acc.support_n, cfg),
        "persistence": _persistence(frames, spec, cfg),
        "noise": _noise(acc.pair_z2, cfg),
        "bias": _bias(acc.pair_bearing, cfg),
        "freeze": _freeze(frames, cfg),
        "latency": _latency(frames, cfg),
        "stream": _stream(store, spec, now),
        "intrinsic": _intrinsic(frames, sensor, cfg),
    }


def _support(frame, near, spec, specs, ego, cfg, grid, acc) -> None:
    """Is each detection confirmed by a witness that could have seen it?"""
    for det in frame.detections:
        covering = [
            w for w in near if can_observe(specs[w], ego, det.x, det.y, cfg, grid)
        ]
        if not covering:
            continue  # nobody could confirm it: no evidence either way
        acc.support_n += 1
        for w in covering:
            limit = gate(spec, specs[w], det.range_m, cfg)
            if nearest_within(det.x, det.y, near[w].detections, limit) is not None:
                acc.support_ok += 1
                break


def _pairwise(frame, near, spec, specs, ego, cfg, grid, acc) -> None:
    """Position and bearing disagreement with each witness separately.

    Judging against each witness on its own lets us find the odd one out: a
    healthy sensor agrees with at least one partner, a faulty one with none.
    """
    for det in frame.detections:
        for w, fr in near.items():
            if not can_observe(specs[w], ego, det.x, det.y, cfg, grid):
                continue
            ref = nearest_within(
                det.x, det.y, fr.detections, gate(spec, specs[w], det.range_m, cfg)
            )
            if ref is None:
                continue
            sigma = math.hypot(
                sigma_pos(spec, det.range_m), sigma_pos(specs[w], det.range_m)
            )
            acc.pair_z2.setdefault(w, []).append(
                (math.hypot(det.x - ref.x, det.y - ref.y) / sigma) ** 2
            )
            _, ref_bearing = polar(ego[0], ego[1], ego[2], ref.x, ref.y)
            acc.pair_bearing.setdefault(w, []).append(
                wrap_angle(det.bearing_rad - ref_bearing)
            )


def _references(
    frame, near, witnesses, spec, specs, ego, cfg, grid, acc, strong
) -> None:
    """Did this sensor report objects that trusted witnesses agree exist?

    Two trusted witnesses: reference = objects BOTH report (strong evidence).
    One trusted witness: reference = its detections (weaker, but still useful
    when the other witness is itself degraded).
    """
    if len(near) == 2 and len(witnesses) == 2:
        w1, w2 = witnesses
        points = []
        for a in near[w1].detections:
            limit = gate(specs[w1], specs[w2], a.range_m, cfg)
            b = nearest_within(a.x, a.y, near[w2].detections, limit)
            if b is None:
                continue
            k1 = 1.0 / sigma_pos(specs[w1], a.range_m) ** 2
            k2 = 1.0 / sigma_pos(specs[w2], b.range_m) ** 2
            points.append(
                ((a.x * k1 + b.x * k2) / (k1 + k2), (a.y * k1 + b.y * k2) / (k1 + k2))
            )
    elif len(near) == 1 and (strong is None or next(iter(near)) in strong):
        (only,) = near.values()
        points = [(d.x, d.y) for d in only.detections]
    else:
        return
    for px, py in points:
        if not can_observe(spec, ego, px, py, cfg, grid):
            continue
        acc.ref_n += 1
        r, _ = polar(ego[0], ego[1], ego[2], px, py)
        limit = max(cfg.min_gate_m, cfg.gate_sigmas * 2.0 * sigma_pos(spec, r))
        if nearest_within(px, py, frame.detections, limit) is not None:
            acc.ref_hit += 1


def _ratio(hits: int, total: int, cfg: HealthConfig) -> Indicator:
    if total < cfg.min_evidence:
        return Indicator(None, total)
    return Indicator(hits / total, total)


def _persistence(
    frames: list[SensorFrame], spec: SensorConfig, cfg: HealthConfig
) -> Indicator:
    """Do detections reappear near the same place in the next frame?

    Needs no other sensor, so it still works when the witnesses are degraded.
    """
    total = kept = 0
    for prev, cur in zip(frames[:-1], frames[1:], strict=True):
        if not prev.detections or cur.timestamp - prev.timestamp > 2.5 / spec.rate_hz:
            continue
        for det in cur.detections:
            total += 1
            sigma = sigma_pos(spec, det.range_m)
            limit = max(cfg.min_gate_m, cfg.gate_sigmas * math.sqrt(2.0) * sigma)
            if nearest_within(det.x, det.y, prev.detections, limit) is not None:
                kept += 1
    return _ratio(kept, total, cfg)


def _noise(per_witness: dict[str, list[float]], cfg: HealthConfig) -> Indicator:
    """RMS disagreement in nominal units, from the witness it agrees with best."""
    best = None
    for z2 in per_witness.values():
        if len(z2) >= cfg.min_evidence:
            value = math.sqrt(sum(z2) / len(z2))
            if best is None or value < best.value:
                best = Indicator(value, len(z2))
    return best or Indicator(
        None, max((len(v) for v in per_witness.values()), default=0)
    )


def _bias(per_witness: dict[str, list[float]], cfg: HealthConfig) -> Indicator:
    """Significant mean bearing offset (degrees), from the best-agreeing witness."""
    best = None
    for errors in per_witness.values():
        n = len(errors)
        if n < max(cfg.min_evidence, 5):
            continue
        mean = float(np.mean(errors))
        sem = float(np.std(errors, ddof=1)) / math.sqrt(n)
        value = math.degrees(max(0.0, abs(mean) - 2.0 * sem))
        if best is None or value < best.value:
            best = Indicator(value, n)
    return best or Indicator(
        None, max((len(v) for v in per_witness.values()), default=0)
    )


def _freeze(frames: list[SensorFrame], cfg: HealthConfig) -> Indicator:
    pairs = same = 0
    for prev, cur in zip(frames[:-1], frames[1:], strict=True):
        if not prev.detections or not cur.detections:
            continue  # empty frames carry no information
        pairs += 1
        if [(d.x, d.y) for d in prev.detections] == [
            (d.x, d.y) for d in cur.detections
        ]:
            same += 1
    if pairs < cfg.min_evidence:
        return Indicator(None, pairs)
    return Indicator(same / pairs, pairs)


def _latency(frames: list[SensorFrame], cfg: HealthConfig) -> Indicator:
    if len(frames) < cfg.min_evidence:
        return Indicator(None, len(frames))
    return Indicator(sum(f.latency for f in frames) / len(frames), len(frames))


def _stream(store: FrameStore, spec: SensorConfig, now: float) -> Indicator:
    age = now - (store.last_arrival if store.last_arrival is not None else 0.0)
    return Indicator(max(0.0, age) * spec.rate_hz, 1)


def _intrinsic(frames: list[SensorFrame], sensor: str, cfg: HealthConfig) -> Indicator:
    values = [
        d.label_confidence
        for f in frames
        for d in f.detections
        if d.label_confidence is not None
    ]
    if sensor != "camera" or len(values) < cfg.min_evidence:
        return Indicator(None, len(values))
    return Indicator(float(np.mean(values)), len(values))
