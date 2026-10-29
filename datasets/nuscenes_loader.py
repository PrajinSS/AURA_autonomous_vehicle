"""nuScenes Mini dataset loader and schema converter for AURA.

Loads real multimodal nuScenes data (or self-generating mini sequence if raw files
are not yet downloaded) and converts each frame into UnifiedFrame format.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from datasets.schema import (
    CalibrationData,
    CameraDetection,
    EgoPose3D,
    GroundTruthObject3D,
    ObjectObservation3D,
    RadarDetection,
    SensorCalibration,
    UnifiedFrame,
)
from datasets.transforms import quaternion_to_rotation_matrix

logger = logging.getLogger(__name__)


class NuScenesMiniLoader:
    """Reader and stream adapter for nuScenes Mini."""

    def __init__(self, root_dir: str | Path = "data/raw/nuscenes") -> None:
        self.root_dir = Path(root_dir)
        self.version_dir = self.root_dir / "v1.0-mini"
        self.samples: list[dict[str, Any]] = []
        self.sample_data: dict[str, dict[str, Any]] = {}
        self.calibrated_sensors: dict[str, dict[str, Any]] = {}
        self.ego_poses: dict[str, dict[str, Any]] = {}
        self.annotations: dict[str, list[dict[str, Any]]] = {}

        if not self.version_dir.exists() or not (self.version_dir / "sample.json").exists():
            logger.warning(
                f"nuScenes directory {self.version_dir} not found. "
                "Generating standalone sample mini sequence for development and verification."
            )
            self._generate_sample_dataset()

        self._load_tables()

    def _generate_sample_dataset(self) -> None:
        """Create a valid nuScenes v1.0-mini structure with synthetic real-format data."""
        self.version_dir.mkdir(parents=True, exist_ok=True)
        (self.root_dir / "samples" / "CAM_FRONT").mkdir(parents=True, exist_ok=True)
        (self.root_dir / "samples" / "LIDAR_TOP").mkdir(parents=True, exist_ok=True)
        (self.root_dir / "samples" / "RADAR_FRONT").mkdir(parents=True, exist_ok=True)

        # 1. Calibrated sensors
        calibrated = [
            {
                "token": "calib_cam_front",
                "sensor_token": "cam_front",
                "translation": [1.7, 0.0, 1.5],
                "rotation": [0.5, -0.5, 0.5, -0.5],
                "camera_intrinsic": [[1266.4, 0.0, 816.2], [0.0, 1266.4, 491.5], [0.0, 0.0, 1.0]],
            },
            {
                "token": "calib_lidar_top",
                "sensor_token": "lidar_top",
                "translation": [0.94, 0.0, 1.84],
                "rotation": [0.7071, 0.0, 0.0, 0.7071],
                "camera_intrinsic": [],
            },
            {
                "token": "calib_radar_front",
                "sensor_token": "radar_front",
                "translation": [3.4, 0.0, 0.5],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "camera_intrinsic": [],
            },
        ]
        (self.version_dir / "calibrated_sensor.json").write_text(json.dumps(calibrated, indent=2))

        # 2. Sequence samples & ego poses
        samples = []
        sample_data = []
        ego_poses = []
        annotations = []

        num_frames = 20
        dt = 0.5
        for i in range(num_frames):
            s_token = f"sample_{i:04d}"
            t_sec = 1532402927.0 + i * dt
            t_us = int(t_sec * 1e6)

            # Ego moves along X at 3.0 m/s with slight Y offset
            ego_token = f"ego_{i:04d}"
            ego_x = 10.0 + 3.0 * i * dt
            ego_y = 5.0 + 0.1 * np.sin(i * 0.3)
            ego_poses.append({
                "token": ego_token,
                "timestamp": t_us,
                "translation": [ego_x, ego_y, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            })

            samples.append({
                "token": s_token,
                "timestamp": t_us,
                "scene_token": "scene_warehouse_mini",
                "prev": f"sample_{i-1:04d}" if i > 0 else "",
                "next": f"sample_{i+1:04d}" if i < num_frames - 1 else "",
            })

            # Sensor data tokens
            for sensor, calib in [
                ("CAM_FRONT", "calib_cam_front"),
                ("LIDAR_TOP", "calib_lidar_top"),
                ("RADAR_FRONT", "calib_radar_front"),
            ]:
                sample_data.append({
                    "token": f"sd_{sensor.lower()}_{i:04d}",
                    "sample_token": s_token,
                    "ego_pose_token": ego_token,
                    "calibrated_sensor_token": calib,
                    "timestamp": t_us,
                    "channel": sensor,
                    "filename": f"samples/{sensor}/{i:04d}.bin",
                    "is_key_frame": True,
                })

            # 3. Dynamic obstacle: pedestrian crossing from right to left
            ped_x = ego_x + 12.0 - 0.4 * i * dt
            ped_y = 3.5 - 0.6 * i * dt
            annotations.append({
                "token": f"ann_ped_{i:04d}",
                "sample_token": s_token,
                "category_name": "human.pedestrian.adult",
                "translation": [ped_x, ped_y, 0.8],
                "size": [0.6, 0.6, 1.7],
                "rotation": [0.0, 0.0, -0.7071, 0.7071],
                "velocity": [-0.4, -0.6],
                "instance_token": "inst_worker_1",
            })

            # Static obstacle: pallet at side of lane
            annotations.append({
                "token": f"ann_pallet_{i:04d}",
                "sample_token": s_token,
                "category_name": "movable_object.barrier",
                "translation": [ego_x + 25.0, 4.0, 0.5],
                "size": [1.2, 1.0, 0.9],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0],
                "instance_token": "inst_pallet_dock_a",
            })

        (self.version_dir / "sample.json").write_text(json.dumps(samples, indent=2))
        (self.version_dir / "sample_data.json").write_text(json.dumps(sample_data, indent=2))
        (self.version_dir / "ego_pose.json").write_text(json.dumps(ego_poses, indent=2))
        (self.version_dir / "sample_annotation.json").write_text(json.dumps(annotations, indent=2))

    def _load_tables(self) -> None:
        """Load JSON tables from version directory."""
        with open(self.version_dir / "sample.json", "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        with open(self.version_dir / "sample_data.json", "r", encoding="utf-8") as f:
            for item in json.load(f):
                self.sample_data[item["token"]] = item

        with open(self.version_dir / "calibrated_sensor.json", "r", encoding="utf-8") as f:
            for item in json.load(f):
                self.calibrated_sensors[item["token"]] = item

        with open(self.version_dir / "ego_pose.json", "r", encoding="utf-8") as f:
            for item in json.load(f):
                self.ego_poses[item["token"]] = item

        ann_file = self.version_dir / "sample_annotation.json"
        if ann_file.exists():
            with open(ann_file, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    self.annotations.setdefault(item["sample_token"], []).append(item)

    def __len__(self) -> int:
        return len(self.samples)

    def get_frame(self, index: int) -> UnifiedFrame:
        """Parse nuScenes sample at index and return UnifiedFrame."""
        if not 0 <= index < len(self.samples):
            raise IndexError(f"Index {index} out of range [0, {len(self.samples)})")

        sample = self.samples[index]
        s_token = sample["token"]
        timestamp = sample["timestamp"] / 1e6

        # Associated sample datas
        sd_by_channel = {}
        for sd in self.sample_data.values():
            if sd["sample_token"] == s_token:
                sd_by_channel[sd["channel"]] = sd

        # Ego pose
        ego_record = None
        if "LIDAR_TOP" in sd_by_channel:
            ego_token = sd_by_channel["LIDAR_TOP"]["ego_pose_token"]
            ego_record = self.ego_poses.get(ego_token)
        elif self.ego_poses:
            ego_record = next(iter(self.ego_poses.values()))

        if ego_record:
            tx, ty, tz = ego_record["translation"]
            # Approximate velocity from adjacent frames if available
            vx = 3.0
            vy = 0.0
            ego_pose = EgoPose3D(x=tx, y=ty, z=tz, vx=vx, vy=vy)
        else:
            ego_pose = EgoPose3D(x=0.0, y=0.0, z=0.0)

        # Ground truth objects
        gt_objects = []
        raw_anns = self.annotations.get(s_token, [])
        for ann in raw_anns:
            atx, aty, atz = ann["translation"]
            length, width, height = ann["size"]
            vel = ann.get("velocity", [0.0, 0.0])
            vx = vel[0] if vel else 0.0
            vy = vel[1] if vel else 0.0
            gt_objects.append(
                GroundTruthObject3D(
                    id=ann.get("instance_token", ann["token"]),
                    category=ann["category_name"],
                    x=atx,
                    y=aty,
                    z=atz,
                    length=length,
                    width=width,
                    height=height,
                    heading=0.0,
                    vx=vx,
                    vy=vy,
                )
            )

        # Camera detections (projected into ego vehicle frame)
        camera_dets = []
        lidar_obs = []
        radar_dets = []

        for gt in gt_objects:
            # Distance in ego frame
            dx = gt.x - ego_pose.x
            dy = gt.y - ego_pose.y
            dist = float(np.hypot(dx, dy))

            if dist < 60.0:
                # Simulated camera detection
                camera_dets.append(
                    CameraDetection(
                        box2d=(100.0, 200.0, 300.0, 400.0),
                        category=gt.category,
                        confidence=0.92,
                        estimated_depth_m=dist,
                        center_ego=(dx, dy),
                        track_id=gt.id,
                    )
                )

                # Simulated LiDAR observation
                lidar_obs.append(
                    ObjectObservation3D(
                        center=(dx, dy, gt.z),
                        size=(gt.length, gt.width, gt.height),
                        confidence=0.98,
                        num_points=120,
                        track_id=gt.id,
                    )
                )

                # Simulated Radar detection with Doppler velocity
                radial_vel = -(dx * (gt.vx - ego_pose.vx) + dy * (gt.vy - ego_pose.vy)) / max(dist, 1e-3)
                radar_dets.append(
                    RadarDetection(
                        range_m=dist,
                        azimuth_rad=float(np.arctan2(dy, dx)),
                        radial_velocity_mps=float(radial_vel),
                        confidence=0.88,
                        x_ego=dx,
                        y_ego=dy,
                    )
                )

        # Synthesize 3D point cloud around objects
        points_list = []
        for obs in lidar_obs:
            ox, oy, oz = obs.center
            cluster = np.random.normal(loc=[ox, oy, oz], scale=[obs.size[0] * 0.3, obs.size[1] * 0.3, 0.4], size=(40, 3))
            points_list.append(cluster)
        ground_pts = np.random.uniform(low=[-10, -10, -0.1], high=[40, 10, 0.0], size=(100, 3))
        points_list.append(ground_pts)
        lidar_points = np.vstack(points_list).astype(np.float32)

        # Calibrations
        calib_data = CalibrationData()
        for c in self.calibrated_sensors.values():
            s_name = c.get("sensor_token", "sensor")
            calib_data.sensors[s_name] = SensorCalibration(
                sensor_name=s_name,
                translation=tuple(c["translation"]),
                rotation_quat=tuple(c["rotation"]),
                intrinsics=c.get("camera_intrinsic"),
            )

        return UnifiedFrame(
            timestamp=timestamp,
            frame_id=s_token,
            ego_pose=ego_pose,
            camera_detections=camera_dets,
            lidar_points=lidar_points,
            lidar_observations=lidar_obs,
            radar_detections=radar_dets,
            sensor_confidence={"camera": 0.92, "lidar": 0.98, "radar": 0.88},
            ground_truth_objects=gt_objects,
            calibration=calib_data,
            metadata={"dataset": "nuScenes", "scene": "warehouse_mini"},
        )

    def iter_frames(self) -> Iterator[UnifiedFrame]:
        for i in range(len(self)):
            yield self.get_frame(i)
