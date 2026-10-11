import math

import pytest

from experiments.runner import run_episode
from sensors import FaultSchedule, SensorSuite
from sensors.metrics import detections_dataframe, frames_dataframe, sensor_metrics
from sensors.profiles import (
    FaultProfileError,
    available_fault_profiles,
    build_fault_schedule,
    parse_fault_profile,
)
from simulation.world_state import ObstacleTruth, VehicleTruth, WorldState


def _world(t):
    obj = ObstacleTruth("a", "worker", "circle", 20.0, 0.0, 0.0, 0.0, 1.0, 1.0)
    return WorldState(t, VehicleTruth(0.0, 0.0, 0.0, 0.0, 0.0), (obj,))


def test_every_shipped_profile_loads():
    names = available_fault_profiles()
    assert {"nominal", "camera_darkness", "lidar_dropout", "radar_interference"} <= set(
        names
    )
    for name in names:
        assert isinstance(build_fault_schedule(name), FaultSchedule)


@pytest.mark.parametrize(
    "data,fragment",
    [
        ({}, "missing required field 'name'"),
        ({"name": "x", "faults": [{"sensor": "camera", "type": "noise"}]}, "'level'"),
        (
            {
                "name": "x",
                "faults": [{"sensor": "camera", "type": "smoke", "level": 1}],
            },
            "unknown fault",
        ),
        (
            {"name": "x", "faults": [{"sensor": "gps", "type": "noise", "level": 1}]},
            "sensor must be",
        ),
        (
            {
                "name": "x",
                "faults": [{"sensor": "camera", "type": "dropout", "level": 3}],
            },
            "level for",
        ),
    ],
)
def test_bad_profiles_have_clear_errors(data, fragment):
    with pytest.raises(FaultProfileError, match=fragment):
        parse_fault_profile(data)


def test_unknown_profile_name():
    with pytest.raises(ValueError, match="Unknown fault profile"):
        build_fault_schedule("nope")


def test_suite_delivers_delayed_frames_late_and_in_order():
    sched = parse_fault_profile(
        {"name": "d", "faults": [{"sensor": "lidar", "type": "delay", "level": 0.3}]}
    )
    suite = SensorSuite.default(sched, seed=1)
    delivered = []
    for k in range(21):
        delivered += [(k * 0.05, f) for f in suite.step(_world(k * 0.05))]
    lidar = [(now, f) for now, f in delivered if f.sensor == "lidar"]
    assert lidar and all(now >= f.timestamp + 0.3 - 1e-9 for now, f in lidar)
    arrivals = [f.arrival_time for _, f in delivered]
    assert arrivals == sorted(arrivals)


def test_adding_a_fault_to_one_sensor_does_not_change_another_sensors_noise():
    def lidar_xs(profile):
        suite = SensorSuite.default(build_fault_schedule(profile), seed=3)
        return [
            d.x
            for k in range(40)
            for f in suite.step(_world(k * 0.05))
            if f.sensor == "lidar"
            for d in f.detections
        ]

    assert lidar_xs("nominal") == lidar_xs("camera_darkness")


def test_metrics_detect_degradation():
    def metrics(profile):
        frames = []
        for seed in range(3):
            frames += run_episode("cluttered", seed, fault_profile=profile)[
                1
            ].sensor_frames
        return sensor_metrics(frames).set_index("sensor")

    nominal, dusty = metrics("nominal"), metrics("camera_dust_progressive")
    assert dusty.loc["camera", "miss_rate"] > nominal.loc["camera", "miss_rate"] + 0.1
    assert dusty.loc["camera", "pos_rmse_m"] > 1.5 * nominal.loc["camera", "pos_rmse_m"]
    assert dusty.loc["lidar", "pos_rmse_m"] == pytest.approx(
        nominal.loc["lidar", "pos_rmse_m"]
    )
    assert nominal.loc["lidar", "pos_rmse_m"] < 0.3


def test_dataframes_have_expected_columns():
    _, result = run_episode("crossing", 1, fault_profile="nominal")
    assert {"sensor", "latency_s", "n_detections", "active_faults"} <= set(
        frames_dataframe(result.sensor_frames).columns
    )
    det = detections_dataframe(result.sensor_frames)
    assert {"x", "y", "error_m", "truth_fault"} <= set(det.columns)
    assert det["error_m"].dropna().ge(0).all()


def test_no_sensors_means_no_frames():
    _, result = run_episode("static", 1)
    assert result.sensor_frames == []


def test_sensor_runs_do_not_change_vehicle_behaviour():
    _, plain = run_episode("cluttered", 5)
    _, sensed = run_episode("cluttered", 5, fault_profile="multi_sensor_degradation")
    assert plain.summary() == sensed.summary()
    assert math.isclose(plain.path_length, sensed.path_length)
