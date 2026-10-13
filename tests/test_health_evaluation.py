import math

import pandas as pd
import pytest

from experiments.runner import run_episode
from health.evaluation import (
    classification_metrics,
    detection_delays,
    health_dataframe,
    truth_class,
)
from simulation.simulator import Simulator


def test_truth_class_rules():
    assert truth_class(0.0, 0.5) == "healthy"
    assert truth_class(0.2, 0.5) == "ambiguous"
    assert truth_class(0.8, 0.5) == "faulty"
    assert truth_class(0.0, 0.5, since_fault_s=1.0) == "ambiguous"  # still recovering


def _df(rows):
    base = dict(
        sensor="camera", status="healthy", truth_severity=0.0, since_fault_s=math.inf
    )
    return pd.DataFrame([{**base, **r} for r in rows])


def test_classification_metrics_counts():
    df = _df(
        [
            dict(time=3, status="healthy"),  # true negative
            dict(time=4, status="degraded"),  # false alarm
            dict(time=5, status="degraded", truth_severity=1.0),  # true positive
            dict(time=6, status="healthy", truth_severity=1.0),  # miss
            dict(time=7, status="healthy", truth_severity=0.2),  # ambiguous: ignored
            dict(time=1, status="degraded"),  # warm-up: ignored
        ]
    )
    m = classification_metrics(df).iloc[0]
    assert (m["healthy_rows"], m["faulty_rows"]) == (2, 2)
    assert m["recall"] == 0.5 and m["precision"] == 0.5 and m["false_alarm_rate"] == 0.5


def test_detection_delay_measured_from_clear_fault_to_first_alarm():
    df = _df(
        [
            dict(time=3, status="healthy", truth_severity=0.6),
            dict(time=4, status="healthy", truth_severity=0.8),
            dict(time=5, status="degraded", truth_severity=1.0),
        ]
    )
    d = detection_delays(df).iloc[0]
    assert d["onset_s"] == 3 and d["delay_s"] == 2


def test_undetected_fault_has_nan_delay():
    df = _df([dict(time=3, status="healthy", truth_severity=1.0)])
    assert math.isnan(detection_delays(df).iloc[0]["delay_s"])


def test_health_needs_sensors():
    with pytest.raises(ValueError, match="needs sensors"):
        Simulator(None, None, None, health=object())


def test_health_dataframe_columns_and_truth_alignment():
    _, result = run_episode(
        "cluttered", 2, fault_profile="camera_darkness", with_health=True
    )
    df = health_dataframe(result)
    assert {
        "time",
        "sensor",
        "reliability",
        "status",
        "truth_severity",
        "since_fault_s",
    } <= set(df.columns)
    assert df["reliability"].between(0, 1).all()
    cam = df[df["sensor"] == "camera"]
    assert cam.loc[cam["time"] < 2.5, "truth_severity"].max() == 0.0
    assert cam["truth_severity"].max() == 1.0


def test_nominal_run_raises_no_alarms():
    for scenario in ("cluttered", "warehouse", "static"):
        _, result = run_episode(scenario, 5, fault_profile="nominal", with_health=True)
        df = health_dataframe(result)
        assert (df[df["time"] >= 2.0]["status"] == "healthy").all(), scenario


@pytest.mark.parametrize(
    "profile,sensor",
    [
        ("camera_darkness", "camera"),
        ("camera_freeze", "camera"),
        ("lidar_delay", "lidar"),
        ("radar_interference", "radar"),
    ],
)
def test_each_fault_type_is_detected_on_its_sensor(profile, sensor):
    _, result = run_episode("cluttered", 6, fault_profile=profile, with_health=True)
    df = health_dataframe(result)
    delays = detection_delays(df)
    row = delays[delays["sensor"] == sensor].iloc[0]
    assert not math.isnan(row["delay_s"]) and row["delay_s"] < 5.0
    others = df[
        (df["sensor"] != sensor) & (df["time"] >= 2.0) & (df["since_fault_s"] >= 3.0)
    ]
    assert (others["status"] == "healthy").mean() > 0.97


def test_health_does_not_change_vehicle_behaviour():
    _, plain = run_episode("cluttered", 4)
    _, monitored = run_episode(
        "cluttered", 4, fault_profile="multi_sensor_degradation", with_health=True
    )
    assert plain.summary() == monitored.summary()
