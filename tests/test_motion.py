import math

import pytest

from simulation.motion import LinearMotion, StaticMotion, WaypointMotion


def test_static_never_moves():
    s = StaticMotion(3.0, 4.0).state_at(99.0)
    assert (s.x, s.y, s.vx, s.vy) == (3.0, 4.0, 0.0, 0.0)


def test_linear_is_exact_at_any_time():
    s = LinearMotion(1.0, 2.0, 0.5, -1.0).state_at(10.0)
    assert math.isclose(s.x, 6.0) and math.isclose(s.y, -8.0)


def test_waypoint_pingpong_goes_and_returns():
    m = WaypointMotion([(0, 0), (10, 0)], speed=2.0, mode="pingpong")
    assert math.isclose(m.state_at(2.5).x, 5.0)
    assert math.isclose(m.state_at(5.0).x, 10.0)
    back = m.state_at(7.5)
    assert math.isclose(back.x, 5.0) and back.vx < 0
    assert math.isclose(m.state_at(10.0).x, 0.0, abs_tol=1e-9)


def test_waypoint_loop_closes_the_polygon():
    m = WaypointMotion([(0, 0), (10, 0), (10, 10)], speed=1.0, mode="loop")
    total = 20.0 + math.hypot(10, 10)
    p = m.state_at(total)  # exactly one full lap
    assert math.isclose(p.x, 0.0, abs_tol=1e-9) and math.isclose(p.y, 0.0, abs_tol=1e-9)


def test_waypoint_once_stops_at_end():
    m = WaypointMotion([(0, 0), (4, 0)], speed=1.0, mode="once")
    end = m.state_at(100.0)
    assert (end.x, end.vx) == (4.0, 0.0)


def test_waypoint_velocity_magnitude_equals_speed():
    m = WaypointMotion([(0, 0), (3, 4)], speed=2.5)
    s = m.state_at(0.3)
    assert math.isclose(math.hypot(s.vx, s.vy), 2.5)


def test_shifted_translates_everything():
    m = WaypointMotion([(0, 0), (10, 0)], speed=1.0).shifted(1.0, 2.0)
    assert m.state_at(0.0).x == 1.0 and m.state_at(0.0).y == 2.0


@pytest.mark.parametrize(
    "points,speed,mode",
    [
        ([(0, 0)], 1.0, "loop"),
        ([(0, 0), (1, 0)], 0.0, "loop"),
        ([(0, 0), (1, 0)], 1.0, "bad"),
        ([(0, 0), (0, 0)], 1.0, "once"),
    ],
)
def test_invalid_waypoint_configs(points, speed, mode):
    with pytest.raises(ValueError):
        WaypointMotion(points, speed, mode)
