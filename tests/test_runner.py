from experiments.runner import aggregate, run_batch, run_episode
from simulation.simulator import TerminationReason
from simulation.vehicle import VehicleCommand


def test_run_episode_returns_spec_and_result():
    spec, result = run_episode("static", seed=3)
    assert spec.name == "static" and result.seed == 3


def test_batch_has_one_row_per_episode():
    df = run_batch(["static", "crossing"], seeds=range(3))
    assert len(df) == 6
    assert {"controller", "scenario", "seed", "success", "collision"} <= set(df.columns)


def test_aggregate_rates_between_zero_and_one():
    df = run_batch(["static"], seeds=range(4))
    table = aggregate(df)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["runs"] == 4 and 0.0 <= row["success_rate"] <= 1.0


def test_custom_controller_can_be_evaluated():
    class Reckless:
        def compute(self, state, goal, obstacles):
            return VehicleCommand(target_speed=4.0, steering=0.0)

    _, result = run_episode("static", 42, controller_factory=lambda vp: Reckless())
    assert result.reason is TerminationReason.COLLISION
