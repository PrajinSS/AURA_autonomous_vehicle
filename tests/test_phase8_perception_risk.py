"""Unit tests for Phase 8: Sensor fusion, Kalman tracker, TTC, and Risk Assessor."""

import numpy as np
import pytest

from datasets.schema import CameraDetection, EgoPose3D, ObjectObservation3D, RadarDetection, UnifiedFrame
from perception.fusion import FusedObject, UncertaintyAwareFusion
from perception.tracker import KalmanTracker
from prediction.trajectory_predictor import TrajectoryPredictor
from risk.risk_assessor import RiskAssessor, RiskLevel
from risk.ttc import TTCCalculator
from simulation.amr_vehicle import AMRState


def test_multimodal_fusion() -> None:
    fusion = UncertaintyAwareFusion()

    # Frame with camera, lidar, radar detecting same object at ~ (10.0, 2.0)
    frame = UnifiedFrame(
        timestamp=1.0,
        frame_id="f1",
        ego_pose=EgoPose3D(x=0, y=0),
        camera_detections=[
            CameraDetection(box2d=(0, 0, 0, 0), category="worker", confidence=0.9, center_ego=(10.2, 2.1))
        ],
        lidar_observations=[
            ObjectObservation3D(center=(9.9, 1.9, 0.5), size=(0.6, 0.6, 1.7), confidence=0.95)
        ],
        radar_detections=[
            RadarDetection(range_m=10.1, azimuth_rad=np.arctan2(2.0, 10.0), radial_velocity_mps=-1.2, x_ego=10.0, y_ego=2.0)
        ],
        sensor_confidence={"camera": 0.9, "lidar": 0.98, "radar": 0.85},
    )

    fused = fusion.fuse(frame)
    assert len(fused) == 1
    obj = fused[0]
    assert 9.8 <= obj.x <= 10.3
    assert 1.8 <= obj.y <= 2.2
    assert obj.category == "worker"
    assert "lidar" in obj.sources and "camera" in obj.sources


def test_kalman_tracker() -> None:
    tracker = KalmanTracker()
    meas = [
        FusedObject(
            id="OBSTACLE-01",
            x=5.0,
            y=1.0,
            vx=-0.5,
            vy=0.0,
            length=0.6,
            width=0.6,
            category="worker",
            confidence=0.9,
            sources=["lidar"],
        )
    ]
    tracks = tracker.update(meas, dt=0.1)
    assert len(tracks) == 1
    t1 = tracks[0]
    assert t1.track_id == "OBSTACLE-01"


def test_ttc_and_risk_assessor() -> None:
    tracker = KalmanTracker()
    # Obstacle directly ahead at x=4.0 moving towards vehicle
    meas = [
        FusedObject(
            id="OBSTACLE-01",
            x=4.0,
            y=0.0,
            vx=-1.0,
            vy=0.0,
            length=0.6,
            width=0.6,
            category="worker",
            confidence=0.95,
            sources=["lidar", "radar"],
        )
    ]
    tracks = tracker.update(meas, dt=0.05)

    predictor = TrajectoryPredictor()
    predictions = predictor.predict(tracks)

    assessor = RiskAssessor()
    state = AMRState(x=0.0, y=0.0, heading=0.0, speed=1.5, battery_soc_pct=72.0)
    report = assessor.assess(state, tracks, predictions, {"camera": 0.9, "lidar": 0.95, "radar": 0.85})

    assert report.min_ttc_sec < 3.0
    assert report.overall_risk_level in (RiskLevel.HIGH, RiskLevel.MEDIUM)
    assert "SLOW DOWN" in report.recommended_action or "STOP" in report.recommended_action
    assert len(report.obstacle_risks) == 1
    assert report.explanation is not None
