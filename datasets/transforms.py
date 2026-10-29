"""Coordinate-frame transformation helpers for 2D/3D autonomous perception.

Handles conversions between:
- Sensor coordinate systems (Camera, LiDAR, Radar)
- Vehicle / Ego coordinate frame (ISO 8855: X forward, Y left, Z up)
- World / Map global frame
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Generate 3x3 rotation matrix from Z-Y-X Tait-Bryan Euler angles (radians)."""
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    r_matrix = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )
    return r_matrix


def quaternion_to_rotation_matrix(q: Sequence[float]) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    sqw = w * w
    sqx = x * x
    sqy = y * y
    sqz = z * z

    invs = 1.0 / (sqx + sqy + sqz + sqw)
    m00 = (sqx - sqy - sqz + sqw) * invs
    m11 = (-sqx + sqy - sqz + sqw) * invs
    m22 = (-sqx - sqy + sqz + sqw) * invs

    tmp1 = x * y
    tmp2 = z * w
    m10 = 2.0 * (tmp1 + tmp2) * invs
    m01 = 2.0 * (tmp1 - tmp2) * invs

    tmp1 = x * z
    tmp2 = y * w
    m20 = 2.0 * (tmp1 - tmp2) * invs
    m02 = 2.0 * (tmp1 + tmp2) * invs

    tmp1 = y * z
    tmp2 = x * w
    m21 = 2.0 * (tmp1 + tmp2) * invs
    m12 = 2.0 * (tmp1 - tmp2) * invs

    return np.array(
        [[m00, m01, m02], [m10, m11, m12], [m20, m21, m22]], dtype=float
    )


def transform_points(
    points: np.ndarray,
    translation: Sequence[float],
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """Transform an (N, 3) point cloud using R * P + T."""
    if len(points) == 0:
        return np.empty((0, 3), dtype=float)
    p3 = points[:, :3]
    transformed = (rotation_matrix @ p3.T).T + np.array(translation, dtype=float)
    if points.shape[1] > 3:
        # Preserve extra channels (e.g. intensity, ring, timestamp)
        return np.hstack([transformed, points[:, 3:]])
    return transformed


def sensor_to_ego_2d(
    x_s: float, y_s: float, sensor_x: float, sensor_y: float, sensor_yaw: float
) -> tuple[float, float]:
    """Transform 2D sensor detection to vehicle ego frame."""
    cos_y = math.cos(sensor_yaw)
    sin_y = math.sin(sensor_yaw)
    x_ego = sensor_x + (cos_y * x_s - sin_y * y_s)
    y_ego = sensor_y + (sin_y * x_s + cos_y * y_s)
    return x_ego, y_ego


def ego_to_world_2d(
    x_ego: float, y_ego: float, ego_x: float, ego_y: float, ego_heading: float
) -> tuple[float, float]:
    """Transform vehicle ego frame coordinate to global world frame."""
    cos_h = math.cos(ego_heading)
    sin_h = math.sin(ego_heading)
    x_w = ego_x + (cos_h * x_ego - sin_h * y_ego)
    y_w = ego_y + (sin_h * x_ego + cos_h * y_ego)
    return x_w, y_w


def world_to_ego_2d(
    x_w: float, y_w: float, ego_x: float, ego_y: float, ego_heading: float
) -> tuple[float, float]:
    """Transform global world coordinate to vehicle ego frame."""
    dx = x_w - ego_x
    dy = y_w - ego_y
    cos_h = math.cos(-ego_heading)
    sin_h = math.sin(-ego_heading)
    x_ego = cos_h * dx - sin_h * dy
    y_ego = sin_h * dx + cos_h * dy
    return x_ego, y_ego
