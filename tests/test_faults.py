import math

import pytest

from sensors.faults import FaultSchedule, FaultSpec, FaultType


def test_severity_ramps_and_stops():
    spec = FaultSpec("camera", FaultType.NOISE, 2.0, start=5.0, end=15.0, ramp=4.0)
    assert spec.severity(4.9) == 0.0
    assert spec.severity(7.0) == pytest.approx(0.5)
    assert spec.severity(10.0) == 1.0
    assert spec.severity(15.0) == 0.0


def test_zero_ramp_is_sudden():
    assert FaultSpec("lidar", FaultType.DROPOUT, 0.5, start=1.0).severity(1.0) == 1.0


def test_effects_only_for_matching_sensor():
    sched = FaultSchedule([FaultSpec("camera", FaultType.NOISE, 3.0)])
    assert sched.effects_at("camera", 1.0).noise_multiplier == pytest.approx(4.0)
    assert sched.effects_at("lidar", 1.0).noise_multiplier == 1.0


def test_probabilities_combine_as_independent_events():
    sched = FaultSchedule(
        [
            FaultSpec("lidar", FaultType.DROPOUT, 0.5),
            FaultSpec("lidar", FaultType.DROPOUT, 0.5),
        ]
    )
    assert sched.effects_at("lidar", 0.0).dropout_prob == pytest.approx(0.75)


def test_bias_is_converted_to_radians_and_scaled_by_severity():
    sched = FaultSchedule([FaultSpec("camera", FaultType.BIAS, 6.0, ramp=10.0)])
    assert sched.effects_at("camera", 5.0).bearing_bias_rad == pytest.approx(
        math.radians(3.0)
    )


def test_active_fault_names_and_severity_reported():
    sched = FaultSchedule([FaultSpec("radar", FaultType.GHOST, 3.0, start=2.0)])
    assert sched.effects_at("radar", 1.0).active == ()
    fx = sched.effects_at("radar", 3.0)
    assert fx.active == ("ghost",) and fx.severity == 1.0 and fx.ghost_rate == 3.0


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(sensor="gps", fault=FaultType.NOISE, level=1.0),
        dict(sensor="camera", fault=FaultType.DROPOUT, level=1.5),
        dict(sensor="camera", fault=FaultType.NOISE, level=1.0, ramp=-1.0),
        dict(sensor="camera", fault=FaultType.NOISE, level=1.0, start=5.0, end=2.0),
    ],
)
def test_invalid_specs_rejected(kwargs):
    with pytest.raises(ValueError):
        FaultSpec(**kwargs)
