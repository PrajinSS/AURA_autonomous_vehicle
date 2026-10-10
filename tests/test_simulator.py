import math

import pytest

from config import SimulationParams
from experiments.runner import run_episode
from experiments.scenarios import available_scenarios
from simulation.environment import Environment
from simulation.geometry import Bounds
from simulation.obstacles import Obstacle
from simulation.simulator import Simulator, TerminationReason
from simulation.vehicle import Vehicle, VehicleCommand, VehicleState


@pytest.mark.parametrize("name", available_scenarios())
def test_every_scenario_reaches_goal_without_collision(name):
    _, result = run_episode(name, seed=42)
    assert result.reason is TerminationReason.GOAL_REACHED
    assert result.summary()["min_clearance_m"] > 0.0


def test_runs_are_reproducible_for_same_seed():
    _, a = run_episode("cluttered", 7)
    _, b = run_episode("cluttered", 7)
    assert a.summary() == b.summary()


def test_collision_is_detected():
    class DriveStraight:
        def compute(self, state, goal, obstacles):
            return VehicleCommand(target_speed=4.0, steering=0.0)

    env = Environment(
        Bounds(0, 50, 0, 20), (45, 10), [Obstacle.static("wall", 20, 10, 2.0)]
    )
    result = Simulator(env, Vehicle(VehicleState(5.0, 10.0)), DriveStraight()).run()
    assert result.reason is TerminationReason.COLLISION


def test_timeout_when_vehicle_cannot_move():
    class Stay:
        def compute(self, state, goal, obstacles):
            return VehicleCommand(0.0, 0.0)

    env = Environment(Bounds(0, 50, 0, 20), (45, 10), [])
    result = Simulator(
        env, Vehicle(VehicleState(5.0, 10.0)), Stay(), SimulationParams(max_time=2.0)
    ).run()
    assert result.reason is TerminationReason.TIMEOUT


def test_world_state_exposes_ground_truth():
    env = Environment(
        Bounds(0, 50, 0, 20), (45, 10), [Obstacle.moving("w", 10, 10, 0.5, 1.0, 0.0)]
    )
    sim = Simulator(env, Vehicle(VehicleState(5.0, 10.0, speed=2.0)), None)
    world = sim.world_state()
    assert world.vehicle.speed == 2.0
    assert world.obstacles[0].obstacle_id == "w" and world.obstacles[0].vx == 1.0
    assert world.to_dict()["time"] == 0.0


def test_running_twice_gives_same_clock_start():
    spec, first = run_episode("static", 1)
    assert first.records[0].t == 0.0
    assert math.isclose(first.records[1].t, SimulationParams().dt)


def test_dataframe_has_expected_columns():
    _, result = run_episode("static", 42)
    cols = set(result.to_dataframe().columns)
    assert {"t", "x", "y", "heading", "speed", "acceleration", "min_gap"} <= cols
