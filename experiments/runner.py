"""Episode and batch runners: the evaluation harness used in every later phase.

Any controller can be evaluated by passing a ``controller_factory``; the
Phase 9 adaptive planner will be compared with the baseline using exactly
the same scenarios, seeds and metrics.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

from config import SimulationParams, VehicleParams
from experiments.scenario_loader import ScenarioSpec
from experiments.scenarios import build_scenario
from health.monitor import SensorHealthMonitor
from sensors.profiles import build_fault_schedule
from sensors.suite import SensorSuite
from simulation.baseline_controller import ReactiveGoalController
from simulation.occupancy_grid import OccupancyGrid
from simulation.simulator import Controller, SimulationResult, Simulator
from simulation.vehicle import Vehicle

ControllerFactory = Callable[[VehicleParams], Controller]


def reactive_baseline(vehicle_params: VehicleParams) -> Controller:
    return ReactiveGoalController(vehicle_params)


def run_episode(
    scenario: str,
    seed: int = 42,
    controller_factory: ControllerFactory = reactive_baseline,
    sim_params: SimulationParams | None = None,
    fault_profile: str | None = None,
    with_health: bool = False,
) -> tuple[ScenarioSpec, SimulationResult]:
    """Run one scenario once. Returns the spec (for plotting) and the result.

    ``fault_profile`` fits camera + LiDAR + radar and injects that profile's
    faults ("nominal" = healthy sensors). ``None`` runs without sensors.
    """
    spec = build_scenario(scenario, seed)
    params = sim_params or SimulationParams(seed=seed)
    vehicle = Vehicle(spec.initial_state)
    controller = controller_factory(vehicle.params)
    sensors = None
    if fault_profile is not None:
        sensors = SensorSuite.default(build_fault_schedule(fault_profile), seed)
    health = None
    if with_health:
        static_map = OccupancyGrid.from_environment(spec.environment, resolution=0.5)
        health = SensorHealthMonitor(static_map=static_map)
    simulator = Simulator(
        spec.environment,
        vehicle,
        controller,
        params,
        spec.name,
        sensors=sensors,
        health=health,
    )
    return spec, simulator.run()


def run_batch(
    scenarios: Iterable[str],
    seeds: Iterable[int],
    controller_factory: ControllerFactory = reactive_baseline,
    label: str = "reactive_baseline",
) -> pd.DataFrame:
    """One row per (scenario, seed) episode."""
    rows = []
    for scenario in scenarios:
        for seed in seeds:
            _, result = run_episode(scenario, seed, controller_factory)
            rows.append({"controller": label, **result.summary()})
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Per-scenario success / collision rates and mean metrics."""
    grouped = df.groupby(["controller", "scenario"])
    table = grouped.agg(
        runs=("seed", "count"),
        success_rate=("success", "mean"),
        collision_rate=("collision", "mean"),
        mean_duration_s=("duration_s", "mean"),
        mean_path_m=("path_length_m", "mean"),
        mean_min_clearance_m=("min_clearance_m", "mean"),
        mean_near_misses=("near_miss_events", "mean"),
    )
    return table.round(3).reset_index()
