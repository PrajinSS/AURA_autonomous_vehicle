import math

import pytest

from health.config import HealthConfig
from health.indicators import (
    FrameStore,
    can_observe,
    gate,
    in_coverage,
    los_clear,
    nearest_within,
    sigma_pos,
)
from health.monitor import DEFAULT_SPECS
from sensors.detection import Detection, SensorFrame
from simulation import Environment, Obstacle, OccupancyGrid
from simulation.geometry import Bounds


def _frame(t, sensor="lidar", arrival=None, dets=()):
    return SensorFrame(sensor, t, t if arrival is None else arrival, tuple(dets))


def _det(x, y):
    return Detection("lidar", 0.0, x, y, math.hypot(x, y), math.atan2(y, x))


def test_frame_store_keeps_time_order_even_for_late_frames():
    store = FrameStore()
    store.add(_frame(1.0, arrival=1.5))
    store.add(_frame(0.5, arrival=1.6))  # measured earlier but arrived later
    assert store.times == [0.5, 1.0]
    assert store.last_arrival == 1.6


def test_frame_store_nearest_and_prune():
    store = FrameStore()
    for t in (0.0, 0.1, 0.2, 0.3):
        store.add(_frame(t))
    assert store.nearest(0.14, 0.05).timestamp == pytest.approx(0.1)
    assert store.nearest(0.5, 0.05) is None
    store.prune(0.15)
    assert store.times == [0.2, 0.3]


def test_sigma_grows_with_distance_and_radar_is_coarser():
    cam, lid, rad = (DEFAULT_SPECS[n] for n in ("camera", "lidar", "radar"))
    assert sigma_pos(lid, 40) > sigma_pos(lid, 10)
    assert sigma_pos(rad, 30) > sigma_pos(cam, 30) > sigma_pos(lid, 30)


def test_coverage_margin_and_field_of_view():
    cam = DEFAULT_SPECS["camera"]  # 40 m, 90 degrees
    ego = (0.0, 0.0, 0.0)
    assert in_coverage(cam, ego, 20.0, 0.0, 0.9)
    assert not in_coverage(cam, ego, 38.0, 0.0, 0.9)  # beyond 90 % of range
    assert not in_coverage(cam, ego, 10.0, 10.0, 0.9)  # 45 deg, beyond 90 % of 45
    assert in_coverage(DEFAULT_SPECS["lidar"], ego, -30.0, 0.0, 0.9)  # 360 degrees


def test_gate_is_never_smaller_than_minimum():
    cfg = HealthConfig()
    assert (
        gate(DEFAULT_SPECS["lidar"], DEFAULT_SPECS["lidar"], 1.0, cfg) == cfg.min_gate_m
    )


def test_nearest_within_respects_limit():
    dets = (_det(10, 0), _det(20, 0))
    assert nearest_within(10.5, 0, dets, 1.0) is dets[0]
    assert nearest_within(15, 0, dets, 1.0) is None


def _grid_with_wall():
    wall = Obstacle.static("wall", 20, 0, 2.0)
    env = Environment(Bounds(0, 60, -20, 20), (55, 0), [wall])
    return OccupancyGrid.from_environment(env, 0.5)


def test_line_of_sight_blocked_by_known_static_object():
    grid, ego = _grid_with_wall(), (0.0, 0.0, 0.0)
    assert not los_clear(grid, ego, 35.0, 0.0)  # directly behind the wall
    assert los_clear(grid, ego, 35.0, 12.0)  # well to the side
    assert los_clear(grid, ego, 20.0, 0.0)  # the wall itself is visible
    assert los_clear(None, ego, 35.0, 0.0)  # no map: assume clear


def test_radar_can_see_through_but_lidar_cannot():
    grid, ego, cfg = _grid_with_wall(), (0.0, 0.0, 0.0), HealthConfig()
    assert not can_observe(DEFAULT_SPECS["lidar"], ego, 35.0, 0.0, cfg, grid)
    assert can_observe(DEFAULT_SPECS["radar"], ego, 35.0, 0.0, cfg, grid)


def test_los_handles_points_outside_the_map_without_crashing():
    assert los_clear(_grid_with_wall(), (0.0, 0.0, 0.0), 500.0, 0.0) in (True, False)
