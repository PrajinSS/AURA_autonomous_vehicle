"""Repeatable scenario builders. Same seed -> identical world."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from simulation.environment import Environment
from simulation.geometry import Bounds
from simulation.obstacles import Obstacle
from simulation.vehicle import VehicleState

FACTORY_BOUNDS = Bounds(0.0, 80.0, 0.0, 40.0)
START = (5.0, 20.0)
GOAL = (75.0, 20.0)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    description: str
    environment: Environment
    initial_state: VehicleState


def _spec(name: str, description: str, obstacles: list[Obstacle]) -> ScenarioSpec:
    env = Environment(FACTORY_BOUNDS, GOAL, obstacles)
    return ScenarioSpec(name, description, env, VehicleState(*START, heading=0.0))


def static_obstacles(seed: int = 42) -> ScenarioSpec:
    """Three pallets/machines between the start and the goal."""
    rng = np.random.default_rng(seed)
    j = rng.uniform(-0.5, 0.5, size=3)
    obstacles = [
        Obstacle.static("machine_1", 28.0, 20.0 + j[0], 2.5),
        Obstacle.static("pallet_2", 48.0, 17.0 + j[1], 3.0),
        Obstacle.static("machine_3", 62.0, 24.0 + j[2], 2.5),
    ]
    return _spec("static", "Reach the goal around three static obstacles.", obstacles)


def crossing_worker(seed: int = 42) -> ScenarioSpec:
    """A worker walks across the vehicle's path near the middle of the map."""
    rng = np.random.default_rng(seed)
    start_y = 2.0 + rng.uniform(-0.5, 0.5)
    obstacles = [
        Obstacle.static("machine_1", 25.0, 23.0, 2.5),
        Obstacle.moving("worker_1", 45.0, start_y, 0.6, vx=0.0, vy=1.5),
    ]
    return _spec("crossing", "A worker crosses the vehicle's route.", obstacles)


def cluttered(seed: int = 42) -> ScenarioSpec:
    """Static clutter plus two moving workers."""
    rng = np.random.default_rng(seed)
    j = rng.uniform(-0.6, 0.6, size=(5, 2))
    layout = [(22, 14), (30, 27), (42, 18), (55, 26), (63, 14)]
    obstacles = [
        Obstacle.static(f"rack_{i + 1}", x + j[i, 0], y + j[i, 1], 2.2)
        for i, (x, y) in enumerate(layout)
    ]
    obstacles.append(Obstacle.moving("worker_1", 36.0, 5.0, 0.6, 0.0, 1.3))
    obstacles.append(Obstacle.moving("worker_2", 58.0, 35.0, 0.6, 0.0, -1.2))
    return _spec("cluttered", "Static racks and two moving workers.", obstacles)


SCENARIOS: dict[str, Callable[[int], ScenarioSpec]] = {
    "static": static_obstacles,
    "crossing": crossing_worker,
    "cluttered": cluttered,
}


def build_scenario(name: str, seed: int = 42) -> ScenarioSpec:
    try:
        return SCENARIOS[name](seed)
    except KeyError:
        valid = ", ".join(sorted(SCENARIOS))
        raise ValueError(f"Unknown scenario '{name}'. Choose from: {valid}") from None
