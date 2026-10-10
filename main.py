"""Command-line entry point for the Phase 1 simulator.

Example:
    python main.py --scenario static --seed 42 --animate
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from config import SimulationParams
from experiments.scenarios import SCENARIOS, build_scenario
from simulation.baseline_controller import ReactiveGoalController
from simulation.simulator import Simulator, TerminationReason
from simulation.vehicle import Vehicle
from simulation.visualization import save_animation, save_trajectory_plot


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 industrial-vehicle simulator")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="static")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--animate", action="store_true", help="also save a GIF")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    spec = build_scenario(args.scenario, args.seed)
    sim_params = SimulationParams(seed=args.seed)
    vehicle = Vehicle(spec.initial_state)
    controller = ReactiveGoalController(vehicle.params)
    simulator = Simulator(
        spec.environment, vehicle, controller, sim_params, name=spec.name
    )
    result = simulator.run()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.name}_seed{args.seed}"
    result.to_dataframe().to_csv(out_dir / f"{stem}.csv", index=False)
    save_trajectory_plot(result, spec.environment, out_dir / f"{stem}.png")
    if args.animate:
        save_animation(result, spec.environment, out_dir / f"{stem}.gif")

    print(f"Scenario: {spec.name} - {spec.description}")
    for key, value in result.summary().items():
        print(f"  {key:>18}: {value}")
    print(f"Files saved in: {out_dir.resolve()}")
    return 0 if result.reason is TerminationReason.GOAL_REACHED else 1


if __name__ == "__main__":
    sys.exit(main())
