import dataclasses
from types import SimpleNamespace

import pytest

from health import HealthConfig, HealthStatus, SensorHealthMonitor
from sensors import FaultSchedule, FaultSpec, FaultType, SensorSuite
from simulation.world_state import ObstacleTruth, VehicleTruth, WorldState

EGO = SimpleNamespace(x=0.0, y=0.0, heading=0.0)
OBJECTS = tuple(
    ObstacleTruth(f"o{i}", "static", "circle", x, y, 0.0, 0.0, 1.5, 1.5)
    for i, (x, y) in enumerate([(15, 3), (25, -4), (30, 6), (12, -5), (22, 9)])
)


def _run(faults=(), duration=12.0, seed=1, strip_truth=False, monitor=None):
    suite = SensorSuite.default(FaultSchedule(list(faults)), seed)
    monitor = monitor or SensorHealthMonitor()
    for k in range(int(duration / 0.05) + 1):
        t = k * 0.05
        world = WorldState(t, VehicleTruth(0, 0, 0, 0, 0), OBJECTS)
        frames = suite.step(world)
        if strip_truth:
            frames = [_strip(f) for f in frames]
        monitor.update(frames, EGO, t)
    return monitor


def _strip(frame):
    dets = tuple(
        dataclasses.replace(
            d, truth_id=None, truth_x=None, truth_y=None, truth_fault=""
        )
        for d in frame.detections
    )
    return dataclasses.replace(
        frame,
        detections=dets,
        truth_expected_ids=(),
        truth_frozen=False,
        truth_active_faults=(),
        truth_severity=0.0,
    )


def _status(monitor):
    return {n: h.status for n, h in monitor.latest.sensors.items()}


def test_healthy_sensors_stay_healthy_with_high_reliability():
    mon = _run()
    assert set(_status(mon).values()) == {HealthStatus.HEALTHY}
    assert min(mon.latest.reliabilities().values()) > 0.9


def test_reports_are_produced_at_the_configured_rate():
    mon = _run(duration=10.0)
    assert len(mon.history) == pytest.approx(10.0 * HealthConfig().eval_rate_hz, abs=3)


def test_dropout_fault_flags_only_the_faulty_sensor():
    mon = _run([FaultSpec("camera", FaultType.DROPOUT, 0.9, start=2.0)])
    status = _status(mon)
    assert status["camera"] is not HealthStatus.HEALTHY
    assert status["lidar"] is status["radar"] is HealthStatus.HEALTHY


def test_ghosts_are_caught_without_any_witness_help():
    mon = _run([FaultSpec("radar", FaultType.GHOST, 6.0, start=2.0)])
    radar = mon.latest.sensors["radar"]
    assert radar.status is not HealthStatus.HEALTHY
    assert any("persist" in r or "confirmed" in r for r in radar.reasons)
    assert _status(mon)["lidar"] is HealthStatus.HEALTHY


def test_frozen_sensor_is_flagged_with_a_clear_reason():
    mon = _run([FaultSpec("lidar", FaultType.FREEZE, 1.0, start=3.0)])
    lidar = mon.latest.sensors["lidar"]
    assert lidar.status is not HealthStatus.HEALTHY
    assert any("identical" in r for r in lidar.reasons)


def test_delayed_sensor_is_flagged():
    mon = _run([FaultSpec("lidar", FaultType.DELAY, 0.4, start=2.0)])
    lidar = mon.latest.sensors["lidar"]
    assert lidar.status is not HealthStatus.HEALTHY
    assert any("late" in r for r in lidar.reasons)


def test_calibration_drift_blames_the_odd_one_out_only():
    mon = _run([FaultSpec("camera", FaultType.BIAS, 8.0, start=2.0)], duration=14.0)
    status = _status(mon)
    assert status["camera"] is not HealthStatus.HEALTHY
    assert status["lidar"] is HealthStatus.HEALTHY  # not blamed for the camera's bias


def test_silent_sensor_is_detected():
    mon = SensorHealthMonitor()
    suite = SensorSuite.default(seed=1)
    for k in range(241):
        t = k * 0.05
        frames = suite.step(WorldState(t, VehicleTruth(0, 0, 0, 0, 0), OBJECTS))
        if t > 3.0:
            frames = [f for f in frames if f.sensor != "radar"]  # radar goes dead
        mon.update(frames, EGO, t)
    radar = mon.latest.sensors["radar"]
    assert radar.status is not HealthStatus.HEALTHY
    assert any("no new data" in r for r in radar.reasons)


def test_recovery_is_slower_than_failure():
    fault = [FaultSpec("camera", FaultType.DROPOUT, 0.95, start=2.0, end=8.0)]
    mon = _run(fault, duration=20.0)
    cam = [r.sensors["camera"].reliability for r in mon.history]
    times = [r.time for r in mon.history]
    fall = next(t for t, q in zip(times, cam, strict=True) if t > 2 and q < 0.65) - 2.0
    rise_start = next(t for t, q in zip(times, cam, strict=True) if t > 8 and q > 0.65)
    assert _status(mon)["camera"] is HealthStatus.HEALTHY  # it does recover
    assert rise_start - 8.0 > fall  # ... but more gradually than it fell


def test_the_engine_never_reads_truth_fields():
    faults = [FaultSpec("camera", FaultType.NOISE, 4.0, start=2.0)]
    with_truth = _run(faults, seed=3)
    without = _run(faults, seed=3, strip_truth=True)
    a = [r.sensors["camera"].reliability for r in with_truth.history]
    b = [r.sensors["camera"].reliability for r in without.history]
    assert a == b


def test_confidence_is_low_without_evidence_and_reliability_is_held():
    mon = SensorHealthMonitor()
    mon.update([], EGO, 0.0)
    h = mon.latest.sensors["camera"]
    assert h.confidence < 0.3 and h.reliability == 1.0


def test_status_hysteresis_prevents_chattering():
    mon = SensorHealthMonitor()
    cfg = mon.config
    q = cfg.degraded_below + 0.02  # just above the line
    assert mon._next_status(HealthStatus.HEALTHY, q) is HealthStatus.HEALTHY
    assert mon._next_status(HealthStatus.DEGRADED, q) is HealthStatus.DEGRADED
    assert (
        mon._next_status(
            HealthStatus.DEGRADED, cfg.degraded_below + cfg.hysteresis + 0.01
        )
        is HealthStatus.HEALTHY
    )
    assert (
        mon._next_status(HealthStatus.HEALTHY, cfg.failed_below - 0.01)
        is HealthStatus.FAILED
    )
    assert (
        mon._next_status(HealthStatus.FAILED, cfg.failed_below + 0.02)
        is HealthStatus.FAILED
    )
