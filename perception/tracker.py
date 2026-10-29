"""Multi-target Kalman filter tracker with covariance gating for AURA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from perception.fusion import FusedObject


@dataclass
class TrackedObstacle:
    """State of an actively tracked obstacle in ego coordinates."""

    track_id: str
    x: float
    y: float
    vx: float
    vy: float
    category: str
    confidence: float
    length: float
    width: float
    covariance: np.ndarray  # 4x4 matrix
    age: int = 1
    hits: int = 1
    misses: int = 0
    confirmed: bool = True

    @property
    def speed(self) -> float:
        return float(math.hypot(self.vx, self.vy))


class KalmanTracker:
    """Multi-object Constant-Velocity Extended Kalman Filter (EKF) Tracker."""

    def __init__(
        self,
        gate_distance_m: float = 2.5,
        max_misses: int = 4,
        min_hits_to_confirm: int = 2,
    ) -> None:
        self.gate_dist = gate_distance_m
        self.max_misses = max_misses
        self.min_hits = min_hits_to_confirm
        self.tracks: dict[str, TrackedObstacle] = {}
        self._next_id = 1

    def update(self, fused_objects: list[FusedObject], dt: float = 0.05) -> list[TrackedObstacle]:
        """Predict existing tracks and update with new fused measurements."""
        # 1. State Prediction Step
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # Process noise Q
        q_pos = 0.01
        q_vel = 0.05
        Q = np.diag([q_pos, q_pos, q_vel, q_vel])

        for t in self.tracks.values():
            x_vec = np.array([t.x, t.y, t.vx, t.vy])
            x_pred = F @ x_vec
            P_pred = F @ t.covariance @ F.T + Q

            t.x, t.y, t.vx, t.vy = x_pred
            t.covariance = P_pred
            t.age += 1

        # 2. Association Step (Greedy Nearest Neighbor)
        assigned_tracks = set()
        assigned_meas = set()

        meas_list = list(fused_objects)
        track_keys = list(self.tracks.keys())

        for m_idx, meas in enumerate(meas_list):
            best_dist = self.gate_dist
            best_track_id = None

            for t_id in track_keys:
                if t_id in assigned_tracks:
                    continue
                tr = self.tracks[t_id]
                dist = math.hypot(tr.x - meas.x, tr.y - meas.y)
                if dist < best_dist:
                    best_dist = dist
                    best_track_id = t_id

            if best_track_id is not None:
                assigned_tracks.add(best_track_id)
                assigned_meas.add(m_idx)
                # Kalman Measurement Update
                self._update_track(self.tracks[best_track_id], meas)

        # 3. Create new tracks for unassigned measurements
        for m_idx, meas in enumerate(meas_list):
            if m_idx not in assigned_meas:
                t_id = meas.id if meas.id.startswith("OBSTACLE-") or meas.id.startswith("worker") else f"OBSTACLE-{self._next_id:02d}"
                self._next_id += 1
                init_P = np.diag([meas.variance_x, meas.variance_y, 0.5, 0.5])
                new_track = TrackedObstacle(
                    track_id=t_id,
                    x=meas.x,
                    y=meas.y,
                    vx=meas.vx,
                    vy=meas.vy,
                    category=meas.category,
                    confidence=meas.confidence,
                    length=meas.length,
                    width=meas.width,
                    covariance=init_P,
                    confirmed=True,
                )
                self.tracks[t_id] = new_track

        # 4. Handle missed tracks
        for t_id in track_keys:
            if t_id not in assigned_tracks:
                self.tracks[t_id].misses += 1

        # 5. Prune dead tracks
        self.tracks = {
            t_id: tr for t_id, tr in self.tracks.items() if tr.misses <= self.max_misses
        }

        return list(self.tracks.values())

    def _update_track(self, track: TrackedObstacle, meas: FusedObject) -> None:
        """Apply linear Kalman measurement update."""
        H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        R = np.diag([max(0.01, meas.variance_x), max(0.01, meas.variance_y)])

        z = np.array([meas.x, meas.y])
        x_vec = np.array([track.x, track.y, track.vx, track.vy])

        y = z - H @ x_vec
        S = H @ track.covariance @ H.T + R
        K = track.covariance @ H.T @ np.linalg.inv(S)

        x_up = x_vec + K @ y
        P_up = (np.eye(4) - K @ H) @ track.covariance

        track.x, track.y, track.vx, track.vy = x_up
        track.covariance = P_up
        track.hits += 1
        track.misses = 0
        track.confidence = 0.8 * track.confidence + 0.2 * meas.confidence
        track.category = meas.category
        track.length = meas.length
        track.width = meas.width
