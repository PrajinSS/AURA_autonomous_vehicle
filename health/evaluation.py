"""Grade the health engine against the recorded fault labels (truth_* fields).

The labels are used ONLY here, to score the engine. The engine itself never
sees them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

from experiments.runner import run_episode
from health.config import INDICATORS
from simulation.simulator import SimulationResult


def _since_fault(frames: list, idx: int, now: float) -> float:
    """Seconds since this sensor last had an active fault (inf = never)."""
    for frame in reversed(frames[: idx + 1]):
        if frame.truth_severity > 0.0:
            return max(0.0, now - frame.timestamp)
    return math.inf


def health_dataframe(result: SimulationResult) -> pd.DataFrame:
    """One row per (report, sensor) with indicators and the true fault state."""
    arrivals: dict[str, np.ndarray] = {}
    frames_by_sensor: dict[str, list] = {}
    for frame in result.sensor_frames:
        frames_by_sensor.setdefault(frame.sensor, []).append(frame)
    for name, frames in frames_by_sensor.items():
        arrivals[name] = np.array([f.arrival_time for f in frames])

    rows = []
    for report in result.health_reports:
        for name, h in report.sensors.items():
            frames = frames_by_sensor.get(name, [])
            idx = (
                int(
                    np.searchsorted(
                        arrivals.get(name, np.array([])), report.time + 1e-9, "right"
                    )
                )
                - 1
            )
            latest = frames[idx] if idx >= 0 else None
            row = {
                "time": report.time,
                "since_fault_s": _since_fault(frames, idx, report.time),
                "sensor": name,
                "reliability": h.reliability,
                "raw_reliability": h.raw_reliability,
                "confidence": h.confidence,
                "status": h.status.value,
                "truth_severity": latest.truth_severity if latest else 0.0,
                "truth_faults": "|".join(latest.truth_active_faults) if latest else "",
            }
            for key in INDICATORS:
                row[f"{key}_value"] = h.values[key]
                row[f"{key}_score"] = h.scores[key]
                row[f"{key}_evidence"] = h.evidence[key]
            rows.append(row)
    return pd.DataFrame(rows)


def truth_class(
    severity: float,
    threshold: float,
    since_fault_s: float = math.inf,
    grace_s: float = 3.0,
) -> str:
    """healthy: no fault for a while. faulty: clearly faulty.

    ambiguous (ignored): a faint fault, or the first ``grace_s`` seconds after a
    fault ended, because recovery is deliberately gradual.
    """
    if severity > 0.0:
        return "faulty" if severity >= threshold else "ambiguous"
    return "healthy" if since_fault_s >= grace_s else "ambiguous"


def classification_metrics(
    df: pd.DataFrame,
    severity_threshold: float = 0.5,
    warmup_s: float = 2.0,
    predicted_faulty: pd.Series | None = None,
) -> pd.DataFrame:
    """Per-sensor accuracy, precision, recall and false-alarm rate.

    The engine says "faulty" when status != healthy (or use ``predicted_faulty``
    to grade another model on the same rows).
    """
    work = df[df["time"] >= warmup_s].copy()
    work["pred"] = (
        (work["status"] != "healthy")
        if predicted_faulty is None
        else predicted_faulty.loc[work.index]
    )
    work["cls"] = [
        truth_class(sev, severity_threshold, since)
        for sev, since in zip(
            work["truth_severity"], work["since_fault_s"], strict=True
        )
    ]
    work = work[work["cls"] != "ambiguous"]
    rows = []
    for sensor, g in work.groupby("sensor"):
        truth = g["cls"] == "faulty"
        pred = g["pred"].astype(bool)
        tp = int((truth & pred).sum())
        fp = int((~truth & pred).sum())
        fn = int((truth & ~pred).sum())
        tn = int((~truth & ~pred).sum())
        rows.append(
            {
                "sensor": sensor,
                "healthy_rows": tn + fp,
                "faulty_rows": tp + fn,
                "accuracy": round((tp + tn) / max(tp + tn + fp + fn, 1), 3),
                "precision": round(tp / (tp + fp), 3) if tp + fp else math.nan,
                "recall": round(tp / (tp + fn), 3) if tp + fn else math.nan,
                "false_alarm_rate": round(fp / (fp + tn), 3) if fp + tn else math.nan,
            }
        )
    return pd.DataFrame(rows)


def detection_delays(df: pd.DataFrame, severity_threshold: float = 0.5) -> pd.DataFrame:
    """Seconds from a fault becoming clear (severity >= threshold) to the alarm.

    ``df`` must be ONE episode. delay is NaN when the fault was never detected.
    """
    rows = []
    for sensor, g in df.groupby("sensor"):
        g = g.sort_values("time")
        onset = g[g["truth_severity"] >= severity_threshold]
        if onset.empty:
            continue
        t0 = float(onset["time"].iloc[0])
        alarms = g[(g["time"] >= t0) & (g["status"] != "healthy")]
        delay = float(alarms["time"].iloc[0]) - t0 if not alarms.empty else math.nan
        rows.append({"sensor": sensor, "onset_s": t0, "delay_s": delay})
    return pd.DataFrame(rows)


def collect_dataset(
    scenarios: Iterable[str], profiles: Iterable[str], seeds: Iterable[int]
) -> pd.DataFrame:
    """Run every (scenario, profile, seed) with the health monitor switched on."""
    frames = []
    seeds = list(seeds)
    profiles = list(profiles)
    for scenario in scenarios:
        for profile in profiles:
            for seed in seeds:
                _, result = run_episode(
                    scenario, seed, fault_profile=profile, with_health=True
                )
                df = health_dataframe(result)
                df.insert(0, "seed", seed)
                df.insert(0, "profile", profile)
                df.insert(0, "scenario", scenario)
                frames.append(df)
    return pd.concat(frames, ignore_index=True)
