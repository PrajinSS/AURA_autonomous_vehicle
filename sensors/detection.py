"""Data produced by sensors.

IMPORTANT: every field whose name starts with ``truth_`` exists only so we can
*evaluate* the system (miss rate, error, fault labels for ML training).
Perception, fusion, risk and planning code must NEVER read ``truth_*`` fields.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One object reported by one sensor at one instant."""

    sensor: str
    timestamp: float  # when the measurement was taken (s)
    x: float  # estimated world position (m)
    y: float
    range_m: float  # measured distance from the vehicle
    bearing_rad: float  # measured angle relative to the vehicle heading
    radial_velocity: float | None = None  # m/s, + = moving away (radar only)
    label: str | None = None  # object class (camera only)
    label_confidence: float | None = None
    # ---- evaluation-only ground truth ----------------------------------
    truth_id: str | None = None  # None for ghosts (false alarms)
    truth_x: float | None = None
    truth_y: float | None = None
    truth_fault: str = ""  # "", "outlier" or "ghost"


@dataclass(frozen=True)
class SensorFrame:
    """Everything one sensor delivered in one measurement cycle."""

    sensor: str
    timestamp: float  # measurement time
    arrival_time: float  # delivery time (>= timestamp when delayed)
    detections: tuple[Detection, ...]
    # ---- evaluation-only ground truth ----------------------------------
    truth_expected_ids: tuple[str, ...] = ()  # visible under nominal conditions
    truth_frozen: bool = False  # stale data was repeated
    truth_active_faults: tuple[str, ...] = ()
    truth_severity: float = 0.0

    @property
    def latency(self) -> float:
        return self.arrival_time - self.timestamp
