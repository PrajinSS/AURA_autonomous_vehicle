"""Unit tests for Phase 5: Datasets, schema converters, and visualizers."""

from pathlib import Path

import numpy as np
import pytest

from datasets.kitti_loader import KittiLoader
from datasets.nuscenes_loader import NuScenesMiniLoader
from datasets.schema import EgoPose3D, UnifiedFrame
from datasets.transforms import (
    ego_to_world_2d,
    euler_to_rotation_matrix,
    quaternion_to_rotation_matrix,
    sensor_to_ego_2d,
    transform_points,
    world_to_ego_2d,
)
from datasets.visualizer import plot_unified_frame


def test_transforms() -> None:
    # Test Euler to rotation matrix
    r = euler_to_rotation_matrix(0, 0, 0)
    assert np.allclose(r, np.eye(3))

    # Test 90 deg yaw
    r_yaw90 = euler_to_rotation_matrix(0, 0, np.pi / 2)
    assert np.allclose(r_yaw90 @ np.array([1, 0, 0]), np.array([0, 1, 0]), atol=1e-6)

    # Test quaternion identity
    rq = quaternion_to_rotation_matrix([1.0, 0.0, 0.0, 0.0])
    assert np.allclose(rq, np.eye(3))

    # Test point transformation
    pts = np.array([[1.0, 2.0, 3.0]])
    t_pts = transform_points(pts, translation=[10.0, 0.0, 0.0], rotation_matrix=np.eye(3))
    assert np.allclose(t_pts, [[11.0, 2.0, 3.0]])

    # Test 2D ego to world and back
    ego_x, ego_y = 10.0, 20.0
    heading = np.pi / 4
    wx, wy = ego_to_world_2d(5.0, 0.0, ego_x, ego_y, heading)
    ex, ey = world_to_ego_2d(wx, wy, ego_x, ego_y, heading)
    assert np.allclose([ex, ey], [5.0, 0.0], atol=1e-6)


def test_nuscenes_loader(tmp_path: Path) -> None:
    loader = NuScenesMiniLoader(root_dir=tmp_path / "nuscenes")
    assert len(loader) > 0

    frame = loader.get_frame(0)
    assert isinstance(frame, UnifiedFrame)
    assert frame.ego_pose.vx > 0
    assert len(frame.camera_detections) > 0
    assert len(frame.lidar_observations) > 0
    assert len(frame.radar_detections) > 0
    assert len(frame.ground_truth_objects) > 0
    assert frame.lidar_points.shape[1] == 3

    # Check serialization
    d = frame.to_dict()
    assert d["frame_id"] == frame.frame_id
    assert "ego_pose" in d


def test_kitti_loader(tmp_path: Path) -> None:
    loader = KittiLoader(root_dir=tmp_path / "kitti")
    assert len(loader) > 0

    frame = loader.get_frame(0)
    assert isinstance(frame, UnifiedFrame)
    assert frame.lidar_points.shape[1] == 3
    assert len(frame.ground_truth_objects) > 0


def test_dataset_visualizer(tmp_path: Path) -> None:
    loader = NuScenesMiniLoader(root_dir=tmp_path / "nuscenes")
    frame = loader.get_frame(0)
    out_img = tmp_path / "test_preview.png"
    plot_unified_frame(frame, save_path=out_img)
    assert out_img.exists()
    assert out_img.stat().st_size > 1000
