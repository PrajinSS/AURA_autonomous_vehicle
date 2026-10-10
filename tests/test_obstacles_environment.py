import math

import numpy as np
import pytest

from simulation.environment import Environment
from simulation.geometry import Bounds
from simulation.motion import WaypointMotion
from simulation.obstacles import Obstacle
from simulation.shapes import Rectangle

BOUNDS = Bounds(0, 20, 0, 20)


def test_static_obstacle_does_not_move():
    o = Obstacle.static("a", 5, 5, 1)
    o.update(10.0)
    assert (o.x, o.y) == (5, 5) and o.is_static


def test_moving_obstacle_follows_motion_model():
    o = Obstacle.moving("w", 2, 2, 0.5, vx=1.0, vy=0.0)
    o.update(4.0)
    assert math.isclose(o.x, 6.0) and not o.is_static


def test_gap_circle_and_rectangle():
    circle = Obstacle.static("c", 10, 10, 1)
    assert math.isclose(circle.gap(5, 10, 1.0), 3.0)
    box = Obstacle("r", Rectangle(4, 2), WaypointMotion([(10, 10), (12, 10)], 1.0))
    assert math.isclose(box.gap(5, 10, 1.0), 2.0)  # 3 m from the box edge, minus 1


def test_future_positions_are_exact():
    o = Obstacle.moving("w", 0, 0, 0.5, vx=2.0, vy=1.0)
    future = o.future_positions(t0=1.0, horizon=2.0, step=0.5)
    assert future.shape == (4, 2)
    assert np.allclose(future[-1], [6.0, 3.0])


def test_frame_reports_size():
    frame = Obstacle(
        "r", Rectangle(4, 2), WaypointMotion([(1, 1), (3, 1)], 1.0)
    ).frame()
    assert (frame.kind, frame.width, frame.height) == ("rectangle", 4, 2)


def test_environment_collision_and_update():
    env = Environment(BOUNDS, (18, 10), [Obstacle.moving("w", 2, 10, 1, 2.0, 0.0)])
    assert not env.in_collision(8, 10, 1.0)
    env.update(3.0)  # obstacle now at x = 8
    assert env.in_collision(8, 10, 1.0)
    assert env.time == 3.0


def test_static_dynamic_split():
    env = Environment(
        BOUNDS,
        (18, 10),
        [Obstacle.static("s", 5, 5, 1), Obstacle.moving("m", 8, 8, 1, 1, 0)],
    )
    assert [o.obstacle_id for o in env.static_obstacles] == ["s"]
    assert [o.obstacle_id for o in env.dynamic_obstacles] == ["m"]


def test_goal_reached():
    env = Environment(BOUNDS, (10, 10), [], goal_tolerance=1.0)
    assert env.goal_reached(10.5, 10) and not env.goal_reached(12, 10)


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        Environment(
            BOUNDS,
            (1, 1),
            [Obstacle.static("a", 5, 5, 1), Obstacle.static("a", 8, 8, 1)],
        )


def test_no_obstacles_gap_is_infinite():
    assert math.isinf(Environment(BOUNDS, (1, 1), []).min_gap(0, 0, 1))
