import math

import pytest

from simulation.environment import Environment
from simulation.geometry import Bounds
from simulation.obstacles import Obstacle

BOUNDS = Bounds(0, 20, 0, 20)


def test_static_obstacle_does_not_move():
    o = Obstacle.static("a", 5, 5, 1)
    o.step(1.0, BOUNDS)
    assert (o.x, o.y) == (5, 5)
    assert o.is_static


def test_moving_obstacle_bounces_at_wall():
    o = Obstacle.moving("w", 18.5, 10, 1, vx=2.0, vy=0.0)
    o.step(1.0, BOUNDS)
    assert o.vx < 0
    assert o.x + o.radius <= BOUNDS.x_max


def test_gap_and_collision():
    env = Environment(BOUNDS, (18, 10), [Obstacle.static("a", 10, 10, 1)])
    assert math.isclose(env.min_gap(5, 10, 1.0), 3.0)
    assert not env.in_collision(5, 10, 1.0)
    assert env.in_collision(9, 10, 1.0)


def test_goal_reached():
    env = Environment(BOUNDS, (10, 10), [], goal_tolerance=1.0)
    assert env.goal_reached(10.5, 10)
    assert not env.goal_reached(12, 10)


def test_duplicate_obstacle_ids_rejected():
    with pytest.raises(ValueError):
        Environment(
            BOUNDS,
            (1, 1),
            [Obstacle.static("a", 5, 5, 1), Obstacle.static("a", 8, 8, 1)],
        )


def test_no_obstacles_gap_is_infinite():
    env = Environment(BOUNDS, (1, 1), [])
    assert math.isinf(env.min_gap(0, 0, 1))
