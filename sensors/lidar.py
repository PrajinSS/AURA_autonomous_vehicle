"""Virtual 2D LiDAR: very accurate position, full 360 degree view, but it
reports no velocity or class, and it loses returns in dust or rain."""

from __future__ import annotations

from sensors.base import BaseSensor, SensorConfig

LIDAR_CONFIG = SensorConfig(
    name="lidar",
    rate_hz=10.0,
    max_range=60.0,
    fov_deg=360.0,
    range_sigma0=0.05,
    range_sigma_per_m=0.002,
    bearing_sigma_deg=0.2,
    base_dropout=0.02,
    far_dropout=0.25,
    occludable=True,
)


class LidarSensor(BaseSensor):
    DEFAULT_CONFIG = LIDAR_CONFIG
