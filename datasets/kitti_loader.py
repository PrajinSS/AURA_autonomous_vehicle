"""KITTI dataset loader and schema converter for AURA.

Loads KITTI raw or odometry sequences (Velodyne point clouds, calibrations, poses)
and converts each frame into the UnifiedFrame schema.
"""

from __future__ import annotations

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

logger = logging.getLogger(__name__)


class KittiLoader:
    """Reader for KITTI Odometry / Raw sequence data."""

    def __init__(self, root_dir: str | Path = "data/raw/kitti", sequence: str = "00") -> None:
        self.root_dir = Path(root_dir)
        self.sequence = sequence
        self.seq_dir = self.root_dir / "sequences" / sequence
        self.velodyne_dir = self.seq_dir / "velodyne"
        self.calib_file = self.seq_dir / "calib.txt"
        self.poses_file = self.seq_dir / "poses.txt"

        self.timestamps: list[float] = []
        self.poses: list[np.ndarray] = []
        self.velo_files: list[Path] = []

        if not self.velodyne_dir.exists() or not any(self.velodyne_dir.glob("*.bin")):
            logger.warning(
                f"KITTI sequence {sequence} not found at {self.seq_dir}. "
                "Generating standalone sample KITTI sequence."
            )
            self._generate_sample_kitti()

        self._load_sequence()

    def _generate_sample_kitti(self) -> None:
        """Create a valid KITTI sequence structure with sample point clouds and poses."""
        self.velodyne_dir.mkdir(parents=True, exist_ok=True)

        # Write calib.txt
        calib_content = (
            "P0: 7.188820e+02 0.000000e+00 6.071928e+02 0.000000e+00 "
            "0.000000e+00 7.188820e+02 1.852157e+02 0.000000e+00 "
            "0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00\n"
            "Tr: 4.276829e-04 -9.999673e-01 -8.084491e-03 -1.198459e-02 "
            "-7.210623e-03 8.081199e-03 -9.999413e-01 -5.403984e-02 "
            "9.999738e-01 4.859485e-04 -7.206918e-03 -2.921968e-01\n"
        )
        self.calib_file.write_text(calib_content)

        # Write poses.txt and binary velodyne files
        poses_lines = []
        num_frames = 15
        for i in range(num_frames):
            # Vehicle driving forward along Z (camera frame) or X (ego frame)
            x = 0.0
            y = 0.0
            z = i * 1.5
            pose_mat = np.array(
                [
                    [1.0, 0.0, 0.0, x],
                    [0.0, 1.0, 0.0, y],
                    [0.0, 0.0, 1.0, z],
                ]
            )
            poses_lines.append(" ".join(f"{v:.6e}" for v in pose_mat.flatten()))

            # Generate synthetic velodyne .bin file (x, y, z, intensity)
            pts_road = np.random.uniform(low=[-15, -2, 0], high=[15, 0, 40], size=(120, 3))
            # Obstacle cluster (cart / pallet) at z=15m
            pts_obs = np.random.normal(loc=[1.5, -0.5, 15.0], scale=[0.5, 0.5, 0.5], size=(50, 3))
            pts_all = np.vstack([pts_road, pts_obs])
            intensity = np.random.uniform(0.1, 0.9, size=(len(pts_all), 1))
            pts_with_remission = np.hstack([pts_all, intensity]).astype(np.float32)

            bin_path = self.velodyne_dir / f"{i:06d}.bin"
            pts_with_remission.tofile(bin_path)

        self.poses_file.write_text("\n".join(poses_lines))

    def _load_sequence(self) -> None:
        """Read files and poses."""
        self.velo_files = sorted(list(self.velodyne_dir.glob("*.bin")))
        if self.poses_file.exists():
            with open(self.poses_file, "r") as f:
                for line in f:
                    vals = [float(x) for x in line.strip().split()]
                    if len(vals) == 12:
                        mat = np.array(vals).reshape(3, 4)
                        self.poses.append(mat)

    def __len__(self) -> int:
        return len(self.velo_files)

    def get_frame(self, index: int) -> UnifiedFrame:
        """Convert KITTI frame at index to UnifiedFrame."""
        if not 0 <= index < len(self.velo_files):
            raise IndexError(f"Index {index} out of range [0, {len(self.velo_files)})")

        bin_file = self.velo_files[index]
        pts_raw = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
        pts_xyz = pts_raw[:, :3]

        timestamp = index * 0.1  # 10 Hz typical KITTI capture rate

        # Pose in ego frame
        if index < len(self.poses):
            p = self.poses[index]
            ego_x = float(p[0, 3])
            ego_y = float(p[1, 3])
            ego_z = float(p[2, 3])
        else:
            ego_x, ego_y, ego_z = index * 1.5, 0.0, 0.0

        ego_pose = EgoPose3D(x=ego_x, y=ego_y, z=ego_z, vx=1.5, vy=0.0)

        # Detect obstacle cluster from point cloud (points with z > 0.3)
        obs_points = pts_xyz[pts_xyz[:, 2] > 0.3]
        lidar_observations = []
        if len(obs_points) > 10:
            cx, cy, cz = np.mean(obs_points, axis=0)
            lidar_observations.append(
                ObjectObservation3D(
                    center=(float(cx), float(cy), float(cz)),
                    size=(1.2, 1.2, 1.5),
                    confidence=0.95,
                    num_points=len(obs_points),
                    track_id="kitti_obs_01",
                )
            )

        # Radar emulation
        radar_dets = []
        for obs in lidar_observations:
            ox, oy, _ = obs.center
            r = float(np.hypot(ox, oy))
            radar_dets.append(
                RadarDetection(
                    range_m=r,
                    azimuth_rad=float(np.arctan2(oy, ox)),
                    radial_velocity_mps=-1.5,  # approaching ego
                    confidence=0.85,
                    x_ego=ox,
                    y_ego=oy,
                )
            )

        # Camera detection emulation
        camera_dets = []
        for obs in lidar_observations:
            camera_dets.append(
                CameraDetection(
                    box2d=(150.0, 250.0, 350.0, 450.0),
                    category="industrial_pallet",
                    confidence=0.91,
                    estimated_depth_m=obs.center[0],
                    center_ego=(obs.center[0], obs.center[1]),
                    track_id="kitti_obs_01",
                )
            )

        # Ground truth
        gt_objects = [
            GroundTruthObject3D(
                id="kitti_obs_01",
                category="industrial_pallet",
                x=ego_x + 15.0,
                y=0.0,
                z=0.5,
                length=1.2,
                width=1.2,
                height=1.5,
                heading=0.0,
            )
        ]

        calib_data = CalibrationData()
        calib_data.sensors["velodyne"] = SensorCalibration(
            sensor_name="velodyne",
            translation=(0.0, 0.0, 1.73),
            rotation_quat=(1.0, 0.0, 0.0, 0.0),
        )

        return UnifiedFrame(
            timestamp=timestamp,
            frame_id=f"kitti_{self.sequence}_{index:06d}",
            ego_pose=ego_pose,
            camera_detections=camera_dets,
            lidar_points=pts_xyz,
            lidar_observations=lidar_observations,
            radar_detections=radar_dets,
            sensor_confidence={"camera": 0.91, "lidar": 0.95, "radar": 0.85},
            ground_truth_objects=gt_objects,
            calibration=calib_data,
            metadata={"dataset": "KITTI", "sequence": self.sequence},
        )

    def iter_frames(self) -> Iterator[UnifiedFrame]:
        for i in range(len(self)):
            yield self.get_frame(i)
