"""Unit tests for Phase 11: Benchmark suite runner and metrics collector."""

from pathlib import Path

from experiments.benchmark_suite import BENCHMARK_SCENARIOS, CONTROLLERS, run_scenario
from experiments.metrics_collector import ScenarioMetrics, compute_displacement_errors, compute_rmse
import numpy as np


def test_metrics_collector() -> None:
    p_traj = np.array([[0, 0], [1, 1], [2, 2]])
    gt_traj = np.array([[0, 0], [1, 1], [2, 2]])
    ade, fde = compute_displacement_errors(p_traj, gt_traj)
    assert ade == 0.0
    assert fde == 0.0

    rmse = compute_rmse(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert rmse == 0.0


def test_benchmark_scenario_runner() -> None:
    # Test single scenario execution
    metrics = run_scenario("normal_operation", "full_aura_system", max_steps=10)
    assert isinstance(metrics, ScenarioMetrics)
    assert metrics.mission_success
    assert metrics.battery_consumed_pct >= 0.0
    assert metrics.simulation_fps > 0.0
