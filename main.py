"""Command-line entry point: run one scenario and save log, plots and metadata.

Examples:
    python main.py --list
    python main.py --scenario warehouse --seed 7 --animate --save-grid
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from config import ControllerParams, SimulationParams, VehicleParams
from experiments.runner import run_episode
from experiments.scenarios import available_scenarios
from sensors.metrics import detections_dataframe, frames_dataframe, sensor_metrics
from sensors.profiles import available_fault_profiles
from sensors.visualization import save_sensor_overview
from simulation.occupancy_grid import OccupancyGrid
from simulation.simulator import TerminationReason
from simulation.visualization import (
    save_animation,
    save_occupancy_grid_plot,
    save_trajectory_plot,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Industrial-vehicle simulator")
    parser.add_argument("--scenario", default="static")
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parser.add_argument(
        "--faults",
        default=None,
        help="fit sensors and inject this fault profile ('nominal' = healthy)",
    )
    parser.add_argument(
        "--list-faults", action="store_true", help="list fault profiles"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--animate", action="store_true", help="also save a GIF")
    parser.add_argument("--save-grid", action="store_true", help="save occupancy grid")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.faults is not None and args.faults not in available_fault_profiles():
        parser.error(
            f"unknown fault profile '{args.faults}'. "
            f"Choose from: {', '.join(available_fault_profiles())}"
        )
    if (
        not (args.list or args.list_faults)
        and args.scenario not in available_scenarios()
    ):
        parser.error(
            f"unknown scenario '{args.scenario}'. "
            f"Choose from: {', '.join(available_scenarios())}"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        print("\n".join(available_scenarios()))
        return 0
    if args.list_faults:
        print("\n".join(available_fault_profiles()))
        return 0
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    spec, result = run_episode(args.scenario, args.seed, fault_profile=args.faults)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.name}_seed{args.seed}"
    result.to_dataframe().to_csv(out_dir / f"{stem}.csv", index=False)
    save_trajectory_plot(result, spec.environment, out_dir / f"{stem}.png")
    if args.animate:
        save_animation(result, spec.environment, out_dir / f"{stem}.gif")
    if args.save_grid:
        grid = OccupancyGrid.from_environment(
            spec.environment, 0.5, VehicleParams().radius
        )
        save_occupancy_grid_plot(grid, spec.environment, out_dir / f"{stem}_grid.png")
    sensor_table = None
    if args.faults is not None:
        frames = result.sensor_frames
        frames_dataframe(frames).to_csv(
            out_dir / f"{stem}_sensor_frames.csv", index=False
        )
        detections_dataframe(frames).to_csv(
            out_dir / f"{stem}_detections.csv", index=False
        )
        save_sensor_overview(
            result,
            spec.environment,
            out_dir / f"{stem}_sensors.png",
            title=f"{spec.name} with faults: {args.faults}",
        )
        sensor_table = sensor_metrics(frames)
    metadata = {
        "fault_profile": args.faults,
        "summary": result.summary(),
        "vehicle_params": asdict(VehicleParams()),
        "controller_params": asdict(ControllerParams()),
        "simulation_params": asdict(SimulationParams(seed=args.seed)),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2))
    print(f"Scenario: {spec.name} - {spec.description}")
    for key, value in result.summary().items():
        print(f" {key:>18}: {value}")
    if sensor_table is not None:
        print(f"\nSensor quality (fault profile: {args.faults})")
        print(sensor_table.to_string(index=False))
    print(f"Files saved in: {out_dir.resolve()}")
    return 0 if result.reason is TerminationReason.GOAL_REACHED else 1


if __name__ == "__main__":
    sys.exit(main())
