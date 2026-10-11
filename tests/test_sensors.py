import math

import numpy as np
import pytest

from sensors import (
    CameraSensor,
    FaultSchedule,
    FaultSpec,
    FaultType,
    LidarSensor,
    RadarSensor,
)
from simulation.world_state import ObstacleTruth, VehicleTruth, WorldState


def _world(obstacles, t=0.0, speed=0.0, heading=0.0):
    veh = VehicleTruth(0.0, 0.0, heading, speed, 0.0)
    return WorldState(t, veh, tuple(obstacles))


def _obj(oid, x, y, vx=0.0, vy=0.0, cat="worker", r=0.5):
    return ObstacleTruth(oid, cat, "circle", x, y, vx, vy, 2 * r, 2 * r)


def _faults(*specs):
    return FaultSchedule(list(specs))


def _mean_error(frames, x, y):
    errs = [
        math.hypot(f.detections[0].x - x, f.detections[0].y - y)
        for f in frames
        if f.detections and f.detections[0].truth_id
    ]
    return float(np.mean(errs))


def _collect(sensor_cls, world_fn, n=300, faults=None, seed=1):
    sensor = sensor_cls(faults, np.random.default_rng(seed))
    frames = []
    for k in range(n):
        frame = sensor.sample(world_fn(k * 0.2))  # 5 Hz < every sensor rate
        if frame is not None:
            frames.append(frame)
    return frames


def test_sensor_respects_its_rate():
    sensor = LidarSensor(rng=np.random.default_rng(0))  # 10 Hz
    got = [sensor.sample(_world([], t=k * 0.05)) for k in range(21)]  # 20 Hz, 1 s
    assert sum(f is not None for f in got) == 11


def test_lidar_is_accurate_and_unbiased():
    frames = _collect(LidarSensor, lambda t: _world([_obj("a", 20, 5)], t))
    errors = np.array(
        [
            math.hypot(f.detections[0].x - 20, f.detections[0].y - 5)
            for f in frames
            if f.detections
        ]
    )
    assert errors.mean() < 0.3
    assert len(errors) > 0.9 * len(frames)


def test_radar_is_less_accurate_than_lidar():
    def world(t):
        return _world([_obj("a", 30, 0)], t)

    lidar_error = _mean_error(_collect(LidarSensor, world), 30, 0)
    radar_error = _mean_error(_collect(RadarSensor, world), 30, 0)
    assert radar_error > 3 * lidar_error


def test_radar_radial_velocity_sign_and_value():
    # Target 30 m ahead moving away at 2 m/s while the vehicle stands still.
    frames = _collect(RadarSensor, lambda t: _world([_obj("a", 30, 0, vx=2.0)], t))
    vals = [d.radial_velocity for f in frames for d in f.detections if d.truth_id]
    assert np.mean(vals) == pytest.approx(2.0, abs=0.1)
    # Vehicle driving at the target at 5 m/s: closing speed -> negative.
    frames = _collect(RadarSensor, lambda t: _world([_obj("a", 30, 0)], t, speed=5.0))
    vals = [d.radial_velocity for f in frames for d in f.detections if d.truth_id]
    assert np.mean(vals) == pytest.approx(-5.0, abs=0.1)


def test_camera_reports_class_and_confidence():
    frames = _collect(
        CameraSensor, lambda t: _world([_obj("w", 15, 0, cat="worker")], t)
    )
    dets = [d for f in frames for d in f.detections]
    assert dets and all(0.0 < d.label_confidence <= 0.99 for d in dets)
    share = np.mean([d.label == "person" for d in dets])
    assert share > 0.9


def test_field_of_view_and_range_limits():
    behind = _obj("behind", -10, 0)
    far = _obj("far", 70, 0)
    side = _obj("side", 0, 10)
    cam = _collect(CameraSensor, lambda t: _world([behind, far, side], t), n=20)
    assert all(not f.detections and not f.truth_expected_ids for f in cam)
    lid = _collect(LidarSensor, lambda t: _world([behind, far, side], t), n=20)
    seen = {d.truth_id for f in lid for d in f.detections}
    assert seen == {"behind", "side"}  # 360 degrees, but 70 m is beyond 60 m


def test_occlusion_hides_target_behind_wall_for_lidar_not_radar():
    wall = ObstacleTruth("wall", "static", "rectangle", 10, 0, 0, 0, 3, 10)
    target = _obj("t", 25, 0)
    world = lambda t: _world([wall, target], t)  # noqa: E731
    lid = _collect(LidarSensor, world, n=30)
    assert all("t" not in f.truth_expected_ids for f in lid)
    rad = _collect(RadarSensor, world, n=30)
    assert any(d.truth_id == "t" for f in rad for d in f.detections)


def test_dropout_fault_reduces_detections():
    world = lambda t: _world([_obj("a", 20, 0)], t)  # noqa: E731
    healthy = _collect(LidarSensor, world)
    faulty = _collect(
        LidarSensor, world, faults=_faults(FaultSpec("lidar", FaultType.DROPOUT, 0.8))
    )
    rate = lambda fs: np.mean([bool(f.detections) for f in fs])  # noqa: E731
    assert rate(healthy) > 0.9 and rate(faulty) < 0.35
    assert all("dropout" in f.truth_active_faults for f in faulty)


def test_noise_fault_increases_error():
    def world(t):
        return _world([_obj("a", 20, 0)], t)

    noise = _faults(FaultSpec("camera", FaultType.NOISE, 5.0))
    base = _mean_error(_collect(CameraSensor, world), 20, 0)
    noisy = _mean_error(_collect(CameraSensor, world, faults=noise), 20, 0)
    assert noisy > 3 * base


def test_range_and_fov_faults_shrink_visibility_but_keep_expected():
    world = lambda t: _world([_obj("a", 30, 0)], t)  # noqa: E731
    frames = _collect(
        CameraSensor,
        world,
        n=20,
        faults=_faults(FaultSpec("camera", FaultType.RANGE, 0.5)),
    )
    assert all(not f.detections for f in frames)  # 40 m * 0.5 = 20 m < 30 m
    assert all("a" in f.truth_expected_ids for f in frames)  # counts as a MISS


def test_outliers_are_flagged_and_far_off():
    frames = _collect(
        LidarSensor,
        lambda t: _world([_obj("a", 20, 0)], t),
        faults=_faults(FaultSpec("lidar", FaultType.OUTLIER, 1.0)),
        n=60,
    )
    dets = [d for f in frames for d in f.detections]
    assert dets and all(d.truth_fault == "outlier" for d in dets)
    assert np.mean([math.hypot(d.x - 20, d.y) for d in dets]) > 3.0


def test_ghosts_have_no_truth_id():
    frames = _collect(
        RadarSensor,
        lambda t: _world([], t),
        faults=_faults(FaultSpec("radar", FaultType.GHOST, 5.0)),
        n=60,
    )
    dets = [d for f in frames for d in f.detections]
    assert len(dets) > 20
    assert all(d.truth_id is None and d.truth_fault == "ghost" for d in dets)


def test_delay_fault_sets_arrival_time():
    frames = _collect(
        LidarSensor,
        lambda t: _world([_obj("a", 20, 0)], t),
        n=10,
        faults=_faults(FaultSpec("lidar", FaultType.DELAY, 0.4)),
    )
    assert all(f.latency == pytest.approx(0.4) for f in frames)


def test_freeze_repeats_stale_detections():
    moving = lambda t: _world([_obj("a", 10 + 2 * t, 0)], t)  # noqa: E731
    frames = _collect(
        LidarSensor,
        moving,
        n=40,
        faults=_faults(FaultSpec("lidar", FaultType.FREEZE, 1.0, start=2.0)),
    )
    frozen = [f for f in frames if f.truth_frozen]
    assert frozen
    assert len({round(f.detections[0].x, 6) for f in frozen if f.detections}) == 1


def test_same_seed_is_reproducible():
    world = lambda t: _world([_obj("a", 20, 3)], t)  # noqa: E731
    a = _collect(CameraSensor, world, n=40, seed=9)
    b = _collect(CameraSensor, world, n=40, seed=9)
    assert [f.detections for f in a] == [f.detections for f in b]
