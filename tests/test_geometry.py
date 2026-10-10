import math

import pytest

from simulation.geometry import Bounds, clamp, wrap_angle


def test_wrap_angle_range():
    for raw in (-10.0, -math.pi, 0.0, 3.5, 7.0, 100.0):
        wrapped = wrap_angle(raw)
        assert -math.pi <= wrapped < math.pi
        assert math.isclose(math.cos(wrapped), math.cos(raw), abs_tol=1e-9)


def test_clamp():
    assert clamp(5.0, 0.0, 3.0) == 3.0
    assert clamp(-1.0, 0.0, 3.0) == 0.0
    assert clamp(2.0, 0.0, 3.0) == 2.0


def test_bounds_contains_circle():
    b = Bounds(0, 10, 0, 10)
    assert b.contains_circle(5, 5, 1)
    assert not b.contains_circle(0.5, 5, 1)


def test_bounds_invalid():
    with pytest.raises(ValueError):
        Bounds(1, 1, 0, 5)
