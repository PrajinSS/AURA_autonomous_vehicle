"""Turn sensor frames into tables and quality metrics (uses truth_* fields)."""

from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd

from sensors.detection import SensorFrame


def frames_dataframe(frames: Iterable[SensorFrame]) -> pd.DataFrame:
    """One row per delivered frame."""
    rows = [
        {
            "sensor": f.sensor,
            "timestamp": f.timestamp,
            "arrival_time": f.arrival_time,
            "latency_s": f.latency,
            "n_detections": len(f.detections),
            "n_expected": len(f.truth_expected_ids),
            "frozen": f.truth_frozen,
            "severity": f.truth_severity,
            "active_faults": "|".join(f.truth_active_faults),
        }
        for f in frames
    ]
    return pd.DataFrame(rows)


def detections_dataframe(frames: Iterable[SensorFrame]) -> pd.DataFrame:
    """One row per detection, with the true position error for matched ones."""
    rows = []
    for f in frames:
        for d in f.detections:
            error = (
                math.hypot(d.x - d.truth_x, d.y - d.truth_y)
                if d.truth_x is not None and d.truth_y is not None
                else math.nan
            )
            rows.append(
                {
                    "sensor": d.sensor,
                    "timestamp": d.timestamp,
                    "arrival_time": f.arrival_time,
                    "x": d.x,
                    "y": d.y,
                    "range_m": d.range_m,
                    "bearing_rad": d.bearing_rad,
                    "radial_velocity": d.radial_velocity,
                    "label": d.label,
                    "label_confidence": d.label_confidence,
                    "truth_id": d.truth_id,
                    "truth_fault": d.truth_fault,
                    "error_m": error,
                    "frame_frozen": f.truth_frozen,
                }
            )
    return pd.DataFrame(rows)


def _rmse(values: list[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values)) if values else math.nan


def sensor_metrics(frames: Iterable[SensorFrame]) -> pd.DataFrame:
    """Per-sensor quality: miss rate, position RMSE, outliers, ghosts, latency."""
    stats: dict[str, dict] = {}
    for f in frames:
        s = stats.setdefault(
            f.sensor,
            {
                "frames": 0,
                "expected": 0,
                "hits": 0,
                "errors": [],
                "inlier_errors": [],
                "outliers": 0,
                "ghosts": 0,
                "latency": 0.0,
                "frozen": 0,
            },
        )
        s["frames"] += 1
        s["latency"] += f.latency
        s["expected"] += len(f.truth_expected_ids)
        if f.truth_frozen:
            s["frozen"] += 1  # stale data: counts as no usable detections
        else:
            found = {d.truth_id for d in f.detections if d.truth_id is not None}
            s["hits"] += len(found & set(f.truth_expected_ids))
            for d in f.detections:
                if d.truth_fault == "ghost":
                    s["ghosts"] += 1
                elif d.truth_x is not None and d.truth_y is not None:
                    err = math.hypot(d.x - d.truth_x, d.y - d.truth_y)
                    s["errors"].append(err)
                    if d.truth_fault == "outlier":
                        s["outliers"] += 1
                    else:
                        s["inlier_errors"].append(err)

    rows = []
    for name, s in sorted(stats.items()):
        matched = len(s["errors"])
        rows.append(
            {
                "sensor": name,
                "frames": s["frames"],
                "miss_rate": round(1.0 - s["hits"] / s["expected"], 3)
                if s["expected"]
                else math.nan,
                "pos_rmse_m": round(_rmse(s["errors"]), 3),
                "pos_rmse_inliers_m": round(_rmse(s["inlier_errors"]), 3),
                "outlier_rate": round(s["outliers"] / matched, 3)
                if matched
                else math.nan,
                "ghosts_per_frame": round(s["ghosts"] / s["frames"], 3),
                "frozen_frames": s["frozen"],
                "mean_latency_s": round(s["latency"] / s["frames"], 3),
            }
        )
    return pd.DataFrame(rows)
