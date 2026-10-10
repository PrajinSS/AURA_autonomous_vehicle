import math

import pytest

from config import VehicleParams
from simulation.vehicle import Vehicle, VehicleCommand, VehicleState


def test_straight_line_motion():
    v = Vehicle(VehicleState(0.0, 0.0, 0.0, 2.0))
    for _ in range(20):
        v.step(VehicleCommand(target_speed=2.0, steering=0.0), 0.05)
    assert math.isclose(v.state.x, 2.0, rel_tol=1e-6)
    assert math.isclose(v.state.y, 0.0, abs_tol=1e-9)


def test_acceleration_limit_respected():
    v = Vehicle(VehicleState(0.0, 0.0))
    v.step(VehicleCommand(target_speed=4.0, steering=0.0), 0.1)
    assert math.isclose(v.state.speed, VehicleParams().max_accel * 0.1)


def test_speed_never_exceeds_max_or_goes_negative():
    v = Vehicle(VehicleState(0.0, 0.0))
    for _ in range(400):
        v.step(VehicleCommand(target_speed=99.0, steering=0.0), 0.05)
    assert v.state.speed <= v.params.max_speed + 1e-9
    for _ in range(400):
        v.step(VehicleCommand(target_speed=-5.0, steering=0.0), 0.05)
    assert v.state.speed == 0.0


def test_steering_turns_heading():
    v = Vehicle(VehicleState(0.0, 0.0, 0.0, 2.0))
    v.step(VehicleCommand(target_speed=2.0, steering=0.3), 0.1)
    assert v.state.heading > 0.0


def test_invalid_dt():
    with pytest.raises(ValueError):
        Vehicle(VehicleState(0, 0)).step(VehicleCommand(1.0, 0.0), 0.0)
