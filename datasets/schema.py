"""Unified internal multimodal perception schema for AURA.

Standardizes nuScenes, KITTI, and CARLA live streams into a single
interoperable representation consumed by fusion, tracking, prediction, and planning.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class EgoPose3D:
    """Vehicle state in map/global coordinates."""

    x: float
    y: float
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))


@dataclass
class CameraDetection:
    """2D/3D camera detection in sensor or ego frame."""

    box2d: tuple[float, float, float, float]  # [ymin, xmin, ymax, xmax] normalized or pixels
    category: str
    confidence: float
    estimated_depth_m: float | None = None
    center_ego: tuple[float, float] | None = None  # (x, y) in vehicle coordinates
    track_id: str | None = None


@dataclass
class ObjectObservation3D:
    """3D bounding-box / cluster observation derived from LiDAR."""

    center: tuple[float, float, float]  # (x, y, z) in ego coordinates
    size: tuple[float, float, float]  # (length, width, height)
    yaw: float = 0.0
    confidence: float = 1.0
    num_points: int = 0
    track_id: str | None = None


@dataclass
class RadarDetection:
    """Radar point detection with Doppler radial velocity."""

    range_m: float
    azimuth_rad: float
    radial_velocity_mps: float
    rcs_db: float = 10.0
    confidence: float = 0.85
    x_ego: float = 0.0
    y_ego: float = 0.0

    def __post_init__(self) -> None:
        if self.x_ego == 0.0 and self.y_ego == 0.0:
            self.x_ego = self.range_m * np.cos(self.azimuth_rad)
            self.y_ego = self.range_m * np.sin(self.azimuth_rad)


@dataclass
class GroundTruthObject3D:
    """Ground-truth 3D bounding box for validation and metrics."""

    id: str
    category: str
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    heading: float
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class SensorCalibration:
    """Sensor extrinsic transform (relative to ego) and intrinsic parameters."""

    sensor_name: str
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # [w, x, y, z]
    intrinsics: list[list[float]] | None = None  # 3x3 camera matrix


@dataclass
class CalibrationData:
    """Extrinsics & intrinsics collection for all vehicle sensors."""

    sensors: dict[str, SensorCalibration] = field(default_factory=dict)


@dataclass
class UnifiedFrame:
    """Unified internal multi-sensor perception frame.

    Standardizes raw dataset samples and live simulation states.
    """

    timestamp: float
    frame_id: str
    ego_pose: EgoPose3D
    camera_detections: list[CameraDetection] = field(default_factory=list)
    lidar_points: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    lidar_observations: list[ObjectObservation3D] = field(default_factory=list)
    radar_detections: list[RadarDetection] = field(default_factory=list)
    sensor_confidence: dict[str, float] = field(
        default_factory=lambda: {"camera": 1.0, "lidar": 1.0, "radar": 1.0}
    )
    ground_truth_objects: list[GroundTruthObject3D] = field(default_factory=list)
    calibration: CalibrationData = field(default_factory=CalibrationData)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert frame into JSON-serializable dictionary."""
        d = {
            "timestamp": self.timestamp,
            "frame_id": self.frame_id,
            "ego_pose": asdict(self.ego_pose),
            "camera_detections": [asdict(c) for c in self.camera_detections],
            "lidar_point_count": int(len(self.lidar_points)),
            "lidar_observations": [asdict(o) for o in self.lidar_observations],
            "radar_detections": [asdict(r) for r in self.radar_detections],
            "sensor_confidence": self.sensor_confidence,
            "ground_truth_objects": [asdict(g) for g in self.ground_truth_objects],
            "metadata": self.metadata,
        }
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
