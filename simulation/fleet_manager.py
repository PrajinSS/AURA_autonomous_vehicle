"""Autonomous fleet manager and multi-vehicle traffic coordinator for AURA.

Features:
- Fleet of 3 independently controlled AMR vehicles: AMR-01, AMR-02, AMR-03
- Independent mission lifecycle, state machine, and path planning per vehicle
- Multi-vehicle spatial deconfliction & intersection right-of-way
- Dynamic obstacle generation (moving workers, static racks, other vehicles)
- Real 2D spatial traffic density heatmap computation
- Battery charging at charging bays
- Swept-footprint collision prevention (zero wall tunneling)
- Authoritative simulation clock with pause/resume and speed scaling
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from config import CampusNavParams
from mission.parser import CommandParser, MissionTask, TaskType
from mission.state_machine import MissionProgress, MissionStateMachine, MissionStatus
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
from perception.fusion import UncertaintyAwareFusion
from perception.tracker import KalmanTracker, TrackedObstacle
from prediction.trajectory_predictor import TrajectoryPredictor
from risk.risk_assessor import AssessmentReport, RiskAssessor
from simulation.amr_vehicle import AMRParams, AMRState, AMRVehicle
from simulation.environment import Environment
from simulation.factory_world import ChargingBay, DockLocation, FactoryWorld
from simulation.geometry import Bounds
from simulation.motion import LinearMotion, StaticMotion
from simulation.navigator import AutonomousNavigator, NavigationCommand
from simulation.obstacles import Obstacle
from simulation.road_network import RoadNetwork
from simulation.shapes import Circle, Rectangle


@dataclass
class FleetVehicleUnit:
    """Encapsulates a single autonomous vehicle in the logistics fleet."""

    vehicle_id: str
    home_charger: tuple[float, float]
    vehicle: AMRVehicle
    state_machine: MissionStateMachine
    navigator: AutonomousNavigator
    fusion: UncertaintyAwareFusion
    tracker: KalmanTracker
    predictor: TrajectoryPredictor
    risk_assessor: RiskAssessor

    # Sensor fault simulation
    fault_camera: bool = False
    fault_lidar: bool = False
    fault_radar: bool = False

    # Cached last outputs
    last_nav_cmd: NavigationCommand | None = None
    last_risk_report: AssessmentReport | None = None
    last_tracks: list[TrackedObstacle] = field(default_factory=list)
    last_sensor_confidence: dict[str, float] = field(default_factory=dict)


class FleetManager:
    """Orchestrates 3 AMR vehicles, traffic rules, and authoritative campus state."""

    def __init__(self) -> None:
        self.world = FactoryWorld()
        self.parser = CommandParser(locations=self.world.DOCKS)
        self.nav_params = CampusNavParams()

        # Simulation clock
        self.sim_time = 0.0
        self.dt = self.nav_params.fixed_dt_s  # 20 Hz tick
        self.frame_count = 0
        self.is_paused = False
        self.sim_speed = 1.0

        # Spatial Heatmap Grid: expanded site X in [-65, 75], Y in [-40, 48]
        self.grid_x_min = -65.0
        self.grid_x_max = 75.0
        self.grid_y_min = -40.0
        self.grid_y_max = 48.0
        self.grid_cols = 35
        self.grid_rows = 22
        self.heatmap_matrix = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        # Initialize 3 Fleet Vehicles
        self.vehicles: dict[str, FleetVehicleUnit] = {}
        self._init_fleet()

        # Currently selected vehicle for detailed UI telemetry
        self.selected_vehicle_id = "AMR-01"

        # Timestamped Event Log
        self.event_log: list[dict[str, Any]] = []
        self._max_events = 60
        self.log_event("system", "AURA Logistics Campus Digital Twin Initialized")

    def _init_fleet(self) -> None:
        """Create AMR-01, AMR-02, and AMR-03 with distinct chargers and starting poses."""
        configs = [
            ("AMR-01", (-25.0, -28.0), math.pi / 2, 88.0),
            ("AMR-02", (-15.0, -28.0), math.pi / 2, 76.0),
            ("AMR-03", (-5.0, -28.0), math.pi / 2, 92.0),
        ]

        for v_id, (hx, hy), head, soc in configs:
            amr = AMRVehicle(
                initial_state=AMRState(
                    x=hx,
                    y=hy,
                    heading=head,
                    speed=0.0,
                    battery_soc_pct=soc,
                    status="OPERATIONAL",
                ),
                params=AMRParams(
                    radius=self.nav_params.vehicle_radius_m,
                    max_speed=self.nav_params.cruise_speed_mps,
                    max_accel=self.nav_params.max_accel_mps2,
                    max_brake=self.nav_params.max_brake_mps2,
                ),
            )
            sm = MissionStateMachine(
                waypoint_tolerance=2.2,
                target_dwell=5.0,  # 5-second loading/unloading
                require_worker_approval=True,
                home_charger=(hx, hy),
            )
            nav = AutonomousNavigator()
            fusion = UncertaintyAwareFusion()
            tracker = KalmanTracker()
            predictor = TrajectoryPredictor()
            risk = RiskAssessor()

            self.vehicles[v_id] = FleetVehicleUnit(
                vehicle_id=v_id,
                home_charger=(hx, hy),
                vehicle=amr,
                state_machine=sm,
                navigator=nav,
                fusion=fusion,
                tracker=tracker,
                predictor=predictor,
                risk_assessor=risk,
            )

    def log_event(self, event_type: str, message: str, vehicle_id: str | None = None) -> None:
        """Add timestamped entry to fleet audit log."""
        prefix = f"[{vehicle_id}] " if vehicle_id else ""
        self.event_log.append({
            "time": round(self.sim_time, 1),
            "type": event_type,
            "vehicle_id": vehicle_id or "FLEET",
            "message": f"{prefix}{message}",
        })
        if len(self.event_log) > self._max_events:
            self.event_log = self.event_log[-self._max_events:]

    def dispatch_mission(self, task: MissionTask) -> str:
        """Dispatch a mission task to an explicit or auto-selected vehicle."""
        v_id = task.target_vehicle_id

        # Auto-assignment if no vehicle explicitly addressed
        if not v_id:
            candidates = [
                u for u in self.vehicles.values()
                if u.state_machine.status in (MissionStatus.IDLE, MissionStatus.IDLE_AT_CHARGER, MissionStatus.CHARGING, MissionStatus.COMPLETED)
                and u.vehicle.state.battery_soc_pct >= 15.0
            ]
            if not candidates:
                raise ValueError("No AMR vehicle is currently available. All units are busy or low battery.")
            candidates.sort(key=lambda u: u.vehicle.state.battery_soc_pct, reverse=True)
            v_id = candidates[0].vehicle_id

        if v_id not in self.vehicles:
            raise ValueError(f"Unknown vehicle ID: '{v_id}'. Available fleet: {list(self.vehicles.keys())}")

        unit = self.vehicles[v_id]

        if unit.vehicle.state.battery_soc_pct < 15.0 and task.task_type == TaskType.PICKUP_AND_DELIVER:
            raise ValueError(f"Vehicle {v_id} cannot accept mission: Low battery ({unit.vehicle.state.battery_soc_pct}%). Return to charger required.")

        curr_pose = (unit.vehicle.state.x, unit.vehicle.state.y)
        unit.state_machine.set_task(task, curr_pose)

        # Clear navigator cache
        unit.navigator.planner.current_normal_path = []
        unit.navigator.planner.current_alt_path = None
        unit.navigator._last_goal = None

        self.log_event("dispatch", f"Dispatched: {task.display_text}", vehicle_id=v_id)
        return v_id

    def cancel_mission(self, vehicle_id: str | None = None) -> None:
        """Cancel mission for specified vehicle or all vehicles."""
        targets = [self.vehicles[vehicle_id]] if vehicle_id and vehicle_id in self.vehicles else self.vehicles.values()
        for u in targets:
            u.state_machine.cancel()
            u.navigator.planner.current_normal_path = []
            u.navigator.planner.current_alt_path = None
            self.log_event("cancel", f"Mission cancelled for {u.vehicle_id}", vehicle_id=u.vehicle_id)

    def emergency_stop(self, vehicle_id: str | None = None) -> None:
        """Engage immediate emergency stop for specified vehicle or all."""
        targets = [self.vehicles[vehicle_id]] if vehicle_id and vehicle_id in self.vehicles else self.vehicles.values()
        for u in targets:
            u.state_machine.emergency_stop()
            u.vehicle.state.speed = 0.0
            u.vehicle.state.status = "EMERGENCY_STOP"
            self.log_event("estop", f"EMERGENCY STOP activated for {u.vehicle_id}", vehicle_id=u.vehicle_id)

    def approve_worker_action(self, vehicle_id: str, action: str, dock_id: str | None = None) -> bool:
        """Operator / worker approval gate."""
        if vehicle_id not in self.vehicles:
            return False

        unit = self.vehicles[vehicle_id]
        success = False
        if action in ("approve_load", "approve"):
            success = unit.state_machine.approve_load(dock_id)
            if success:
                # Rotate AMR in place to face South toward exit doorway (-pi/2)
                unit.vehicle.state.heading = -math.pi / 2
                unit.navigator.planner.current_normal_path = []
                unit.navigator.planner.current_alt_path = None
                self.log_event("approval", f"Worker approved loading departure at {dock_id or 'Dock'}", vehicle_id=vehicle_id)
        elif action in ("approve_unload", "release"):
            success = unit.state_machine.approve_unload(dock_id)
            if success:
                # Rotate AMR in place to face South toward exit doorway (-pi/2)
                unit.vehicle.state.heading = -math.pi / 2
                unit.navigator.planner.current_normal_path = []
                unit.navigator.planner.current_alt_path = None
                self.log_event("approval", f"Worker approved delivery and released to charger at {dock_id or 'Dock'}", vehicle_id=vehicle_id)

        return success

    def set_sensor_fault(self, vehicle_id: str, camera: bool, lidar: bool, radar: bool) -> None:
        """Inject sensor degradation for specified vehicle."""
        if vehicle_id in self.vehicles:
            u = self.vehicles[vehicle_id]
            u.fault_camera = camera
            u.fault_lidar = lidar
            u.fault_radar = radar
            self.log_event(
                "fault",
                f"Sensors updated (Cam: {'FAULT' if camera else 'OK'}, LiDAR: {'FAULT' if lidar else 'OK'}, Radar: {'FAULT' if radar else 'OK'})",
                vehicle_id=vehicle_id,
            )

    def _generate_simulated_sensor_frame(self, unit: FleetVehicleUnit, env: Environment) -> UnifiedFrame:
        """Generate simulated sensor frame for this vehicle with noise and degradation."""
        cam_conf = 0.15 if unit.fault_camera else 0.96
        lidar_conf = 0.20 if unit.fault_lidar else 0.98
        radar_conf = 0.25 if unit.fault_radar else 0.92

        cam_conf = max(0.05, min(1.0, cam_conf + float(np.random.normal(0, 0.01))))
        lidar_conf = max(0.05, min(1.0, lidar_conf + float(np.random.normal(0, 0.01))))
        radar_conf = max(0.05, min(1.0, radar_conf + float(np.random.normal(0, 0.01))))

        confidences = {
            "camera": round(float(cam_conf), 3),
            "lidar": round(float(lidar_conf), 3),
            "radar": round(float(radar_conf), 3),
        }
        unit.last_sensor_confidence = confidences

        camera_dets: list[CameraDetection] = []
        lidar_obs: list[ObjectObservation3D] = []
        radar_dets: list[RadarDetection] = []

        veh_state = unit.vehicle.state
        v_x, v_y, v_h = veh_state.x, veh_state.y, veh_state.heading
        cos_h = math.cos(v_h)
        sin_h = math.sin(v_h)

        for obs in env.obstacles:
            if getattr(obs, "category", "") == "wall":
                continue
            dx = obs.x - v_x
            dy = obs.y - v_y
            dist = float(np.hypot(dx, dy))
            if dist > 35.0:
                continue

            # Ego relative: x forward, y left
            x_ego = cos_h * dx + sin_h * dy
            y_ego = -sin_h * dx + cos_h * dy

            extent = obs.shape.extent if hasattr(obs.shape, "extent") else (1.0, 1.0)
            cat = getattr(obs, "category", "obstacle")

            lidar_obs.append(
                ObjectObservation3D(
                    center=(float(x_ego), float(y_ego), 0.5),
                    size=(float(extent[0]), float(extent[1]), 1.2),
                    yaw=0.0,
                    confidence=lidar_conf,
                    num_points=120 if dist < 12 else 40,
                )
            )

            azimuth = math.atan2(y_ego, x_ego)
            if x_ego > 0.5 and abs(azimuth) < math.radians(55.0) and not unit.fault_camera:
                camera_dets.append(
                    CameraDetection(
                        box2d=(100.0, 100.0, 300.0, 300.0),
                        category=cat,
                        confidence=cam_conf,
                        estimated_depth_m=dist,
                        center_ego=(float(x_ego), float(y_ego)),
                        track_id=obs.obstacle_id,
                    )
                )

            if x_ego > 0.8 and abs(azimuth) < math.radians(65.0) and dist < 25.0 and not unit.fault_radar:
                radar_dets.append(
                    RadarDetection(
                        range_m=dist,
                        azimuth_rad=azimuth,
                        radial_velocity_mps=0.0,
                        rcs_db=15.0 if cat == "worker" else 22.0,
                        confidence=radar_conf,
                    )
                )

        ego_pose = EgoPose3D(
            x=v_x,
            y=v_y,
            z=0.0,
            yaw=v_h,
            vx=veh_state.speed * cos_h,
            vy=veh_state.speed * sin_h,
            yaw_rate=(veh_state.speed / 1.8) * math.tan(veh_state.steer),
        )

        calib = CalibrationData()
        calib.sensors["camera"] = SensorCalibration("camera", (1.2, 0.0, 1.4))
        calib.sensors["lidar"] = SensorCalibration("lidar", (0.0, 0.0, 1.8))
        calib.sensors["radar"] = SensorCalibration("radar", (1.5, 0.0, 0.4))

        return UnifiedFrame(
            timestamp=self.sim_time,
            frame_id=f"frame_{self.frame_count:06d}",
            ego_pose=ego_pose,
            camera_detections=camera_dets,
            lidar_points=np.empty((0, 3), dtype=np.float32),
            lidar_observations=lidar_obs,
            radar_detections=radar_dets,
            sensor_confidence=confidences,
            calibration=calib,
        )

    def step(self) -> dict[str, Any]:
        """Advance authoritative simulation by one time step."""
        if self.is_paused:
            return self.get_telemetry()

        dt = self.dt * self.sim_speed
        self.sim_time += dt
        self.frame_count += 1

        # 1. Update Spatial Heatmap
        self.heatmap_matrix *= 0.998
        for u in self.vehicles.values():
            gx = int((u.vehicle.state.x - self.grid_x_min) / (self.grid_x_max - self.grid_x_min) * (self.grid_cols - 1))
            gy = int((u.vehicle.state.y - self.grid_y_min) / (self.grid_y_max - self.grid_y_min) * (self.grid_rows - 1))
            if 0 <= gx < self.grid_cols and 0 <= gy < self.grid_rows:
                self.heatmap_matrix[gy, gx] = min(1.0, self.heatmap_matrix[gy, gx] + 0.04)

        for w in self.world.workers:
            wx, wy = w.position_at(self.sim_time)
            gx = int((wx - self.grid_x_min) / (self.grid_x_max - self.grid_x_min) * (self.grid_cols - 1))
            gy = int((wy - self.grid_y_min) / (self.grid_y_max - self.grid_y_min) * (self.grid_rows - 1))
            if 0 <= gx < self.grid_cols and 0 <= gy < self.grid_rows:
                self.heatmap_matrix[gy, gx] = min(1.0, self.heatmap_matrix[gy, gx] + 0.03)

        # 2. Step each vehicle independently
        for v_id, unit in self.vehicles.items():
            state = unit.vehicle.state
            current_pose = (state.x, state.y)

            progress = unit.state_machine.update(current_pose, speed=state.speed, dt=dt)

            # Build environment with solid walls + other vehicles as obstacles
            env = self.world.build_environment(self.sim_time, goal=progress.current_goal)
            for other_id, other_unit in self.vehicles.items():
                if other_id != v_id:
                    o_state = other_unit.vehicle.state
                    env.obstacles.append(
                        Obstacle(
                            obstacle_id=f"veh_{other_id}",
                            shape=Circle(radius=1.3),
                            motion=StaticMotion(o_state.x, o_state.y),
                            category="vehicle",
                        )
                    )

            frame = self._generate_simulated_sensor_frame(unit, env)
            fused_objects = unit.fusion.fuse(frame)
            tracks = unit.tracker.update(fused_objects, dt=dt)
            predictions = unit.predictor.predict(tracks)
            report = unit.risk_assessor.assess(
                vehicle_state=state,
                tracks=tracks,
                predictions=predictions,
                sensor_confidence=frame.sensor_confidence,
            )

            unit.last_tracks = tracks
            unit.last_risk_report = report

            nav_cmd = unit.navigator.compute(
                vehicle_state=state,
                mission_progress=progress,
                env=env,
                tracks=tracks,
                risk_report=report,
            )

            accel = nav_cmd.accel_cmd
            steer = nav_cmd.steer_cmd

            # Multi-Vehicle Traffic Coordination / In-Lane Headway Check (4.5m ahead, 1.3m lateral)
            cos_h = math.cos(state.heading)
            sin_h = math.sin(state.heading)
            for other_id, other_unit in self.vehicles.items():
                if other_id != v_id:
                    ox = other_unit.vehicle.state.x
                    oy = other_unit.vehicle.state.y
                    dx = ox - state.x
                    dy = oy - state.y
                    ox_ego = cos_h * dx + sin_h * dy
                    oy_ego = -sin_h * dx + cos_h * dy

                    # In-lane headway: only brake if other vehicle is directly ahead within our lane
                    if 0.5 < ox_ego < 4.5 and abs(oy_ego) < 1.3:
                        accel = min(accel, -3.5)

            unit.last_nav_cmd = NavigationCommand(
                accel_cmd=accel,
                steer_cmd=steer,
                normal_path=nav_cmd.normal_path,
                alt_path=nav_cmd.alt_path,
                is_rerouting=nav_cmd.is_rerouting,
                is_waiting=nav_cmd.is_waiting,
            )

            # SWEPT COLLISION SAFETY CHECK (PREVENT WALL PENETRATION / TUNNELING)
            # Predict tentative position
            pred_speed = max(0.0, min(unit.vehicle.params.max_speed, state.speed + accel * dt))
            v_mid = 0.5 * (state.speed + pred_speed)
            pred_heading = state.heading + (v_mid / unit.vehicle.params.wheelbase) * math.tan(steer) * dt
            pred_x = state.x + v_mid * math.cos(pred_heading) * dt
            pred_y = state.y + v_mid * math.sin(pred_heading) * dt

            # Test swept line segment against environment colliders (walls, racks, obstacles)
            is_blocked = env.check_swept_collision(state.x, state.y, pred_x, pred_y, unit.vehicle.params.radius, static_only=True)
            if is_blocked and v_mid > 0.05:
                # Halt vehicle immediately at contact boundary
                unit.vehicle.state.speed = 0.0
                unit.vehicle.state.status = "BLOCKED"
            else:
                new_state = unit.vehicle.step(accel, steer, dt)
                if progress.cargo_loaded and not new_state.cargo_loaded:
                    new_state.cargo_loaded = True
                elif not progress.cargo_loaded and new_state.cargo_loaded:
                    new_state.cargo_loaded = False

            # Charging at charger logic
            dist_to_home = float(np.hypot(state.x - unit.home_charger[0], state.y - unit.home_charger[1]))
            if dist_to_home < 2.5 and state.speed < 0.1:
                if unit.state_machine.status in (MissionStatus.CHARGING, MissionStatus.IDLE_AT_CHARGER, MissionStatus.IDLE, MissionStatus.COMPLETED):
                    new_soc = min(100.0, unit.vehicle.state.battery_soc_pct + 0.3 * dt)
                    unit.vehicle.state.battery_soc_pct = round(new_soc, 2)
                    unit.vehicle.state.status = "CHARGING"

        return self.get_telemetry()

    def get_telemetry(self) -> dict[str, Any]:
        """Construct unified telemetry payload for frontend dashboard."""
        sel_unit = self.vehicles.get(self.selected_vehicle_id, next(iter(self.vehicles.values())))
        sel_state = sel_unit.vehicle.state
        sel_progress = sel_unit.state_machine.update((sel_state.x, sel_state.y), speed=sel_state.speed, dt=0.0)
        sel_nav = sel_unit.last_nav_cmd
        sel_report = sel_unit.last_risk_report

        fleet_data = []
        for v_id, u in self.vehicles.items():
            st = u.vehicle.state
            prog = u.state_machine._build_progress((st.x, st.y), st.speed)
            fleet_data.append({
                "vehicle_id": v_id,
                "x": round(st.x, 2),
                "y": round(st.y, 2),
                "heading": round(st.heading, 3),
                "speed": round(st.speed, 2),
                "battery_soc_pct": round(st.battery_soc_pct, 1),
                "status": st.status,
                "mission_status": prog.status.value,
                "subtext": prog.subtext,
                "cargo_loaded": st.cargo_loaded,
                "awaiting_approval": prog.awaiting_approval,
                "approval_type": prog.approval_type,
                "dock_name": prog.current_dock_name,
            })

        workers_data = []
        for w in self.world.workers:
            wx, wy = w.position_at(self.sim_time)
            workers_data.append({
                "id": w.id,
                "name": w.name,
                "x": round(wx, 2),
                "y": round(wy, 2),
            })

        normal_path_data = []
        alt_path_data = []
        if sel_nav:
            for wp in sel_nav.normal_path:
                normal_path_data.append({"name": wp.name, "x": wp.x, "y": wp.y, "eta": wp.eta_str})
            if sel_nav.alt_path:
                for wp in sel_nav.alt_path:
                    alt_path_data.append({"name": wp.name, "x": wp.x, "y": wp.y, "eta": wp.eta_str})

        env = self.world.build_environment(self.sim_time)
        obstacles_world = []
        wall_colliders = []
        for obs in env.obstacles:
            cat = getattr(obs, "category", "static")
            w_ext = round(obs.shape.extent[0], 2) if hasattr(obs.shape, "extent") else 0.8
            h_ext = round(obs.shape.extent[1], 2) if hasattr(obs.shape, "extent") else 0.8
            obs_dict = {
                "id": obs.obstacle_id,
                "x": round(obs.x, 2),
                "y": round(obs.y, 2),
                "type": cat,
                "w": w_ext,
                "h": h_ext,
            }
            obstacles_world.append(obs_dict)
            if cat == "wall":
                wall_colliders.append(obs_dict)

        for other_id, other_unit in self.vehicles.items():
            if other_id != self.selected_vehicle_id:
                obstacles_world.append({
                    "id": f"veh_{other_id}",
                    "x": round(other_unit.vehicle.state.x, 2),
                    "y": round(other_unit.vehicle.state.y, 2),
                    "type": "vehicle",
                    "w": 1.6,
                    "h": 2.4,
                })

        # Road segments geometry for debug and rendering
        road_segments_data = []
        for seg in self.world.road_network.segments:
            n1 = self.world.road_network.nodes[seg.u]
            n2 = self.world.road_network.nodes[seg.v]
            road_segments_data.append({
                "u": seg.u,
                "v": seg.v,
                "x1": n1.x,
                "y1": n1.y,
                "x2": n2.x,
                "y2": n2.y,
                "width": seg.width,
            })

        campus_meta = {
            "warehouses": [
                {
                    "name": b.name,
                    "id": b.building_id,
                    "x_min": b.x_min,
                    "x_max": b.x_max,
                    "y_min": b.y_min,
                    "y_max": b.y_max,
                    "color": b.color_hex,
                    "docks": b.docks,
                }
                for b in self.world.BUILDINGS.values()
            ],
            "docks": [
                {
                    "name": d.name,
                    "id": d.dock_id,
                    "x": d.x,
                    "y": d.y,
                    "heading": d.heading,
                    "warehouse": d.warehouse_name,
                    "is_pickup": d.is_pickup,
                }
                for d in self.world.DOCKS.values()
                if "Dock " in d.name
            ],
            "chargers": [
                {
                    "name": c.name,
                    "id": c.bay_id,
                    "x": c.x,
                    "y": c.y,
                }
                for c in self.world.CHARGING_BAYS.values()
            ],
            "roads": road_segments_data,
            "wall_colliders": wall_colliders,
        }

        flat_heatmap = [round(float(v), 2) for v in self.heatmap_matrix.flatten().tolist()]

        return {
            "timestamp": round(self.sim_time, 2),
            "sim_state": {
                "paused": self.is_paused,
                "speed": self.sim_speed,
                "frame": self.frame_count,
            },
            "selected_vehicle_id": self.selected_vehicle_id,
            "fleet": fleet_data,
            "vehicle": {
                "id": self.selected_vehicle_id,
                "x": round(sel_state.x, 2),
                "y": round(sel_state.y, 2),
                "heading": round(sel_state.heading, 3),
                "speed": round(sel_state.speed, 2),
                "battery_soc_pct": round(sel_state.battery_soc_pct, 1),
                "runtime_remaining_hours": round(sel_state.runtime_remaining_hours, 1),
                "status": sel_state.status,
                "motor_health": sel_state.motor_health,
                "brake_health": sel_state.brake_health,
                "temperature_c": sel_state.temperature_c,
                "cargo_loaded": sel_state.cargo_loaded,
            },
            "workers": workers_data,
            "sensors": sel_unit.last_sensor_confidence or {"camera": 0.95, "lidar": 0.98, "radar": 0.92},
            "mission": {
                "status": sel_progress.status.value,
                "percent_complete": sel_progress.percent_complete,
                "estimated_remaining_time_str": sel_progress.estimated_remaining_time_str,
                "subtext": sel_progress.subtext,
                "active_task": {
                    "display_text": sel_progress.active_task.display_text if sel_progress.active_task else "",
                    "task_type": sel_progress.active_task.task_type.value if sel_progress.active_task else "",
                    "quantity": sel_progress.active_task.quantity if sel_progress.active_task else 1,
                }
                if sel_progress.active_task
                else None,
                "cargo_loaded": sel_progress.cargo_loaded,
                "dwell_progress_pct": sel_progress.dwell_progress_pct,
                "awaiting_approval": sel_progress.awaiting_approval,
                "approval_type": sel_progress.approval_type,
                "dock_name": sel_progress.current_dock_name,
                "current_goal": {
                    "x": round(sel_progress.current_goal[0], 2),
                    "y": round(sel_progress.current_goal[1], 2),
                },
            },
            "navigation": {
                "is_rerouting": sel_nav.is_rerouting if sel_nav else False,
                "is_waiting": sel_nav.is_waiting if sel_nav else False,
                "accel_cmd": round(sel_nav.accel_cmd, 2) if sel_nav else 0.0,
                "steer_cmd": round(sel_nav.steer_cmd, 3) if sel_nav else 0.0,
                "normal_path": normal_path_data,
                "alt_path": alt_path_data,
            },
            "risk": {
                "composite_risk_score": sel_report.composite_risk_score if sel_report else 0.1,
                "overall_risk_level": sel_report.overall_risk_level.value if sel_report else "LOW",
                "min_ttc_sec": sel_report.min_ttc_sec if sel_report else 99.9,
                "recommended_action": sel_report.recommended_action if sel_report else "CONTINUE",
                "decision_reason": sel_report.decision_reason if sel_report else "Corridor clear",
                "explanation": sel_report.explanation if sel_report else "Nominal operation",
                "confidence_pct": round(sel_report.confidence_pct) if sel_report else 95,

                "obstacle_risks": [
                    {
                        "object_id": o.object_id,
                        "distance_m": round(o.distance_m, 1),
                        "ttc_sec": round(o.ttc_sec, 1),
                        "risk_level": o.risk_level.value,
                    }
                    for o in (sel_report.obstacle_risks if sel_report else [])
                ],
            },
            "campus": campus_meta,
            "obstacles": obstacles_world,
            "heatmap": {
                "rows": self.grid_rows,
                "cols": self.grid_cols,
                "x_min": self.grid_x_min,
                "x_max": self.grid_x_max,
                "y_min": self.grid_y_min,
                "y_max": self.grid_y_max,
                "data": flat_heatmap,
            },
            "events": self.event_log[-20:],
        }
