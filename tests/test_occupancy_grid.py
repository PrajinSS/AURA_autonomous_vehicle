import math

from simulation.environment import Environment
from simulation.geometry import Bounds
from simulation.obstacles import Obstacle
from simulation.occupancy_grid import OccupancyGrid


def _env():
    return Environment(
        Bounds(0, 20, 0, 10),
        (18, 5),
        [Obstacle.static("a", 10, 5, 2.0), Obstacle.moving("m", 3, 3, 0.5, 1, 0)],
    )


def test_shape_and_blocked_cells():
    grid = OccupancyGrid.from_environment(_env(), resolution=0.5)
    assert grid.shape == (20, 40)
    assert grid.is_occupied(10, 5)
    assert not grid.is_occupied(2, 2)


def test_moving_obstacles_are_not_rasterised():
    grid = OccupancyGrid.from_environment(_env(), resolution=0.5)
    assert not grid.is_occupied(3, 3)


def test_inflation_grows_blocked_area():
    plain = OccupancyGrid.from_environment(_env(), 0.5, inflation=0.0)
    inflated = OccupancyGrid.from_environment(_env(), 0.5, inflation=1.2)
    assert inflated.occupied_fraction > plain.occupied_fraction
    assert inflated.is_occupied(12.9, 5) and not plain.is_occupied(12.9, 5)


def test_outside_map_counts_as_occupied():
    grid = OccupancyGrid.from_environment(_env(), 0.5)
    assert grid.world_to_cell(-1, 5) is None and grid.is_occupied(-1, 5)


def test_cell_round_trip():
    grid = OccupancyGrid.from_environment(_env(), 0.5)
    row, col = grid.world_to_cell(4.3, 2.2)
    x, y = grid.cell_to_world(row, col)
    assert math.isclose(x, 4.25) and math.isclose(y, 2.25)
