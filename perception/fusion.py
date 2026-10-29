"""Uncertainty-aware multimodal sensor fusion for AURA.

Dynamically fuses Camera, LiDAR, and Radar detections weighted by real-time
sensor reliability scores and covariance matrices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from datasets.schema import CameraDetection, ObjectObservation3D, RadarDetection, UnifiedFrame


@dataclass
class FusedObject:
    """Consolidated representation of a detected object after multimodal fusion."""

    id: str
    x: float  # forward distance in ego frame (m)
    y: float  # lateral offset in ego frame (m)
    vx: float  # longitudinal velocity (m/s)
    vy: float  # lateral velocity (m/s)
    length: float
    width: float
    category: str
    confidence: float
    sources: list[str]  # e.g. ["camera", "lidar", "radar"]
    variance_x: float = 0.05
    variance_y: float = 0.05


class UncertaintyAwareFusion:
    """Covariance-weighted multimodal fusion engine."""

    def __init__(self, association_threshold_m: float = 2.2) -> None:
        self.assoc_thresh = association_threshold_m
        self._counter = 0

    def fuse(
        self,
        frame: UnifiedFrame,
        sensor_weights: dict[str, float] | None = None,
    ) -> list[FusedObject]:
        """Merge detections from all available sensors weighted by reliability."""
        weights = sensor_weights or frame.sensor_confidence
        w_cam = max(0.01, weights.get("camera", 0.9))
        w_lidar = max(0.01, weights.get("lidar", 0.95))
        w_radar = max(0.01, weights.get("radar", 0.85))

        candidates: list[dict] = []

        # 1. LiDAR observations (highest spatial accuracy)
        for obs in frame.lidar_observations:
            ox, oy, _ = obs.center
            candidates.append({
                "sensor": "lidar",
                "x": ox,
                "y": oy,
                "vx": 0.0,
                "vy": 0.0,
                "length": obs.size[0],
                "width": obs.size[1],
                "category": "obstacle",
                "confidence": obs.confidence * w_lidar,
                "weight": w_lidar,
                "var_pos": 0.04 / w_lidar,
                "track_id": obs.track_id,
            })

        # 2. Camera detections (high semantic classification accuracy)
        for cam in frame.camera_detections:
            if cam.center_ego is not None:
                cx, cy = cam.center_ego
                candidates.append({
                    "sensor": "camera",
                    "x": cx,
                    "y": cy,
                    "vx": 0.0,
                    "vy": 0.0,
                    "length": 0.8,
                    "width": 0.8,
                    "category": cam.category,
                    "confidence": cam.confidence * w_cam,
                    "weight": w_cam,
                    "var_pos": 0.25 / w_cam,
                    "track_id": cam.track_id,
                })

        # 3. Radar detections (high Doppler velocity accuracy)
        for rad in frame.radar_detections:
            # Estimate longitudinal velocity from radial velocity
            theta = rad.azimuth_rad
            vx_est = rad.radial_velocity_mps * math.cos(theta)
            vy_est = rad.radial_velocity_mps * math.sin(theta)
            candidates.append({
                "sensor": "radar",
                "x": rad.x_ego,
                "y": rad.y_ego,
                "vx": vx_est,
                "vy": vy_est,
                "length": 1.0,
                "width": 1.0,
                "category": "dynamic_target",
                "confidence": rad.confidence * w_radar,
                "weight": w_radar,
                "var_pos": 0.50 / w_radar,
                "track_id": None,
            })

        # Cluster and fuse candidates
        fused_objects: list[FusedObject] = []
        visited = [False] * len(candidates)

        for i in range(len(candidates)):
            if visited[i]:
                continue
            group = [candidates[i]]
            visited[i] = True

            c1 = candidates[i]
            for j in range(i + 1, len(candidates)):
                if visited[j]:
                    continue
                c2 = candidates[j]
                dist = math.hypot(c1["x"] - c2["x"], c1["y"] - c2["y"])
                if dist <= self.assoc_thresh:
                    group.append(c2)
                    visited[j] = True

            # Perform inverse-variance / reliability-weighted state fusion
            sum_w = 0.0
            sum_x = 0.0
            sum_y = 0.0
            sum_vx = 0.0
            sum_vy = 0.0
            sources = []
            category = "obstacle"
            length = 1.0
            width = 1.0
            max_conf = 0.0

            for member in group:
                src = member["sensor"]
                sources.append(src)
                w = member["weight"] / member["var_pos"]
                sum_w += w
                sum_x += member["x"] * w
                sum_y += member["y"] * w

                # Velocity from radar takes highest weight
                v_weight = 4.0 if src == "radar" else 1.0
                sum_vx += member["vx"] * v_weight
                sum_vy += member["vy"] * v_weight

                # Camera takes priority for semantic class
                if src == "camera" and member["category"] != "obstacle":
                    category = member["category"]
                if src == "lidar":
                    length = member["length"]
                    width = member["width"]

                max_conf = max(max_conf, member["confidence"])

            fused_x = sum_x / max(sum_w, 1e-4)
            fused_y = sum_y / max(sum_w, 1e-4)
            fused_vx = sum_vx / max(len(group), 1)
            fused_vy = sum_vy / max(len(group), 1)

            # Assign track ID
            self._counter += 1
            t_id = group[0].get("track_id") or f"OBSTACLE-{self._counter:02d}"

            fused_objects.append(
                FusedObject(
                    id=t_id,
                    x=round(fused_x, 2),
                    y=round(fused_y, 2),
                    vx=round(fused_vx, 2),
                    vy=round(fused_vy, 2),
                    length=round(length, 2),
                    width=round(width, 2),
                    category=category,
                    confidence=round(min(0.99, max_conf * (1.0 + 0.05 * (len(group) - 1))), 2),
                    sources=list(set(sources)),
                    variance_x=round(1.0 / max(sum_w, 1e-3), 3),
                    variance_y=round(1.0 / max(sum_w, 1e-3), 3),
                )
            )

        return fused_objects
