"""High-Fidelity 3D Factory Digital Twin fallback for AURA.

Runs continuous real-time logistics simulation with exact AMR forklift dynamics,
factory environment (Docks A-F), moving workers, and synthetic Camera, LiDAR,
and Radar sensor streams producing UnifiedFrame telemetry.
"""

from __future__ import annotations

import time
from typing import Any

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
from simulation.amr_vehicle import AMRParams, AMRState, AMRVehicle
from simulation.factory_world import FactoryWorld


class FallbackDigitalTwin:
    """Seamless real-time digital twin when CARLA server is offline or on incompatible Python."""

    def __init__(
        self,
        factory_world: FactoryWorld | None = None,
        amr_params: AMRParams | None = None,
        initial_pose: tuple[float, float, float] = (-15.0, 0.0, 0.0),
        dt: float = 0.05,
    ) -> None:
        self.factory = factory_world or FactoryWorld()
        init_state = AMRState(x=initial_pose[0], y=initial_pose[1], heading=initial_pose[2])
        self.vehicle = AMRVehicle(init_state, params=amr_params)
        self.dt = dt
        self.sim_time = 0.0
        self.frame_count = 0

        # Sensor faults config
        self.faults: dict[str, Any] = {
            "camera_degraded": False,
            "lidar_degraded": False,
            "radar_degraded": False,
        }

    def set_sensor_fault(self, sensor: str, degraded: bool) -> None:
        """Inject or clear a sensor fault dynamically."""
        if f"{sensor}_degraded" in self.faults:
            self.faults[f"{sensor}_degraded"] = degraded

    def step(self, accel_cmd: float = 0.0, steer_cmd: float = 0.0) -> tuple[AMRState, UnifiedFrame]:
        """Advance digital twin by one timestep dt."""
        state = self.vehicle.step(accel_cmd, steer_cmd, self.dt)
        self.sim_time += self.dt
        self.frame_count += 1

        env = self.factory.build_environment(self.sim_time)

        # Build ground-truth objects relative to ego
        gt_objects: list[GroundTruthObject3D] = []
        camera_dets: list[CameraDetection] = []
        lidar_obs: list[ObjectObservation3D] = []
        radar_dets: list[RadarDetection] = []
        point_cloud_list: list[np.ndarray] = []

        # Sensor confidences based on injected faults
        cam_conf = 0.35 if self.faults["camera_degraded"] else 0.92
        lidar_conf = 0.40 if self.faults["lidar_degraded"] else 0.98
        rad_conf = 0.30 if self.faults["radar_degraded"] else 0.85

        for obs in env.obstacles:
            ox, oy = obs.x, obs.y
            dx = ox - state.x
            dy = oy - state.y
            dist = float(np.hypot(dx, dy))

            # Transform to ego frame: X forward, Y left
            cos_h = np.cos(state.heading)
            sin_h = np.sin(state.heading)
            x_ego = cos_h * dx + sin_h * dy
            y_ego = -sin_h * dx + cos_h * dy

            if dist > 35.0:
                continue

            extent = obs.shape.extent
            gt = GroundTruthObject3D(
                id=obs.obstacle_id,
                category="worker" if "worker" in obs.obstacle_id else "pallet",
                x=ox,
                y=oy,
                z=0.5,
                length=extent[0],
                width=extent[1],
                height=1.0 if "pallet" in obs.obstacle_id else 1.7,
                heading=0.0,
            )
            gt_objects.append(gt)

            # Camera: visible if in front (x_ego > 0.5) and within 60 deg FOV
            azimuth = np.arctan2(y_ego, x_ego)
            if x_ego > 0.5 and abs(azimuth) < np.radians(45.0) and not self.faults["camera_degraded"]:
                camera_dets.append(
                    CameraDetection(
                        box2d=(120.0, 200.0, 320.0, 420.0),
                        category=gt.category,
                        confidence=cam_conf,
                        estimated_depth_m=dist,
                        center_ego=(float(x_ego), float(y_ego)),
                        track_id=obs.obstacle_id,
                    )
                )

            # LiDAR: 360 degree coverage
            if not self.faults["lidar_degraded"]:
                lidar_obs.append(
                    ObjectObservation3D(
                        center=(float(x_ego), float(y_ego), 0.5),
                        size=(gt.length, gt.width, gt.height),
                        confidence=lidar_conf,
                        num_points=int(max(10, 80 - dist * 2)),
                        track_id=obs.obstacle_id,
                    )
                )
                pts = np.random.normal(
                    loc=[x_ego, y_ego, 0.5],
                    scale=[gt.length * 0.25, gt.width * 0.25, 0.3],
                    size=(25, 3),
                )
                point_cloud_list.append(pts)

            # Radar: Doppler velocity
            if dist < 30.0 and not self.faults["radar_degraded"]:
                rad_vel = -state.speed * np.cos(azimuth)
                radar_dets.append(
                    RadarDetection(
                        range_m=dist,
                        azimuth_rad=float(azimuth),
                        radial_velocity_mps=float(rad_vel),
                        confidence=rad_conf,
                        x_ego=float(x_ego),
                        y_ego=float(y_ego),
                    )
                )

        if point_cloud_list:
            lidar_pts = np.vstack(point_cloud_list).astype(np.float32)
        else:
            lidar_pts = np.empty((0, 3), dtype=np.float32)

        ego_pose = EgoPose3D(
            x=state.x,
            y=state.y,
            z=0.0,
            yaw=state.heading,
            vx=state.speed * np.cos(state.heading),
            vy=state.speed * np.sin(state.heading),
            yaw_rate=(state.speed / 1.8) * np.tan(state.steer),
        )

        calib = CalibrationData()
        calib.sensors["camera"] = SensorCalibration("camera", (1.2, 0.0, 1.4))
        calib.sensors["lidar"] = SensorCalibration("lidar", (0.0, 0.0, 1.8))
        calib.sensors["radar"] = SensorCalibration("radar", (1.5, 0.0, 0.4))

        frame = UnifiedFrame(
            timestamp=self.sim_time,
            frame_id=f"frame_{self.frame_count:06d}",
            ego_pose=ego_pose,
            camera_detections=camera_dets,
            lidar_points=lidar_pts,
            lidar_observations=lidar_obs,
            radar_detections=radar_dets,
            sensor_confidence={
                "camera": cam_conf,
                "lidar": lidar_conf,
                "radar": rad_conf,
            },
            ground_truth_objects=gt_objects,
            calibration=calib,
            metadata={"source": "AURA_FactoryDigitalTwin", "operational": state.is_operational},
        )

        return state, frame
