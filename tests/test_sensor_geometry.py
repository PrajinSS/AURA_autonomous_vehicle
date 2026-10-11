import math

import pytest

from sensors.geometry import (
    line_of_sight_blocked,
    obstacle_distance,
    polar,
    polar_to_world,
)
from simulation.world_state import ObstacleTruth


def _obs(oid, x, y, w, h, shape="circle"):
    return ObstacleTruth(oid, "static", shape, x, y, 0.0, 0.0, w, h)


def test_polar_round_trip():
    r, b = polar(1.0, 2.0, math.pi / 2, 1.0, 12.0)
    assert r == pytest.approx(10.0) and b == pytest.approx(0.0, abs=1e-9)
    x, y = polar_to_world(1.0, 2.0, math.pi / 2, r, b)
    assert (x, y) == pytest.approx((1.0, 12.0))


def test_obstacle_distance_circle_and_rectangle():
    assert float(obstacle_distance(_obs("c", 0, 0, 2, 2), 5.0, 0.0)) == pytest.approx(
        4.0
    )
    box = _obs("r", 0, 0, 4, 2, "rectangle")
    assert float(obstacle_distance(box, 5.0, 0.0)) == pytest.approx(3.0)
    assert float(obstacle_distance(box, 0.5, 0.5)) == 0.0


def test_line_of_sight_blocked_by_obstacle_in_between():
    target = _obs("t", 20, 0, 1, 1)
    wall = _obs("w", 10, 0, 4, 4)
    assert line_of_sight_blocked(0.0, 0.0, target, [wall])
    side = _obs("s", 10, 8, 4, 4)
    assert not line_of_sight_blocked(0.0, 0.0, target, [side])


def test_no_blockers_never_blocked():
    assert not line_of_sight_blocked(0.0, 0.0, _obs("t", 20, 0, 1, 1), [])
