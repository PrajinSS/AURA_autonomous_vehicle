import pytest

from config import SimulationParams
from experiments.scenarios import SCENARIOS, build_scenario
from simulation.baseline_controller import ReactiveGoalController
from simulation.environment import Environment
from simulation.geometry import Bounds
from simulation.obstacles import Obstacle
from simulation.simulator import Simulator, TerminationReason
from simulation.vehicle import Vehicle, VehicleCommand, VehicleState


def _run(name: str, seed: int = 42):
    spec = build_scenario(name, seed)
    vehicle = Vehicle(spec.initial_state)
    sim = Simulator(
        spec.environment,
        vehicle,
        ReactiveGoalController(vehicle.params),
        SimulationParams(seed=seed),
        name=name,
    )
    return sim.run()


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_every_scenario_reaches_goal_without_collision(name):
    result = _run(name)
    assert result.reason is TerminationReason.GOAL_REACHED
    assert result.summary()["min_clearance_m"] > 0.0


def test_runs_are_reproducible_for_same_seed():
    a, b = _run("cluttered", 7), _run("cluttered", 7)
    assert a.summary() == b.summary()


def test_collision_is_detected():
    class DriveStraight:
        def compute(self, state, goal, obstacles):
            return VehicleCommand(target_speed=4.0, steering=0.0)

    env = Environment(
        Bounds(0, 50, 0, 20), (45, 10), [Obstacle.static("wall", 20, 10, 2.0)]
    )
    vehicle = Vehicle(VehicleState(5.0, 10.0))
    result = Simulator(env, vehicle, DriveStraight()).run()
    assert result.reason is TerminationReason.COLLISION


def test_timeout_when_vehicle_cannot_move():
    class Stay:
        def compute(self, state, goal, obstacles):
            return VehicleCommand(0.0, 0.0)

    env = Environment(Bounds(0, 50, 0, 20), (45, 10), [])
    result = Simulator(
        env,
        Vehicle(VehicleState(5.0, 10.0)),
        Stay(),
        SimulationParams(max_time=2.0),
    ).run()
    assert result.reason is TerminationReason.TIMEOUT


def test_dataframe_has_expected_columns():
    df = _run("static").to_dataframe()
    assert {"t", "x", "y", "speed", "min_gap", "goal_distance"} <= set(df.columns)


def test_unknown_scenario_message():
    with pytest.raises(ValueError, match="Unknown scenario"):
        build_scenario("nope")
