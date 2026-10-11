"""Measure how each fault profile changes sensor quality.

Example:
    python evaluate_sensors.py --scenario cluttered --seeds 5
    python evaluate_sensors.py --profiles nominal camera_darkness lidar_dropout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from experiments.runner import run_episode
from experiments.scenarios import available_scenarios
from sensors.metrics import sensor_metrics
from sensors.profiles import available_fault_profiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sensor quality under faults")
    parser.add_argument("--scenario", default="cluttered")
    parser.add_argument("--profiles", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    if args.scenario not in available_scenarios():
        parser.error(f"unknown scenario '{args.scenario}'")
    profiles = args.profiles or available_fault_profiles()
    unknown = set(profiles) - set(available_fault_profiles())
    if unknown:
        parser.error(f"unknown profile(s): {', '.join(sorted(unknown))}")
    tables = []
    for profile in profiles:
        frames = []
        for seed in range(args.seeds):
            _, result = run_episode(args.scenario, seed, fault_profile=profile)
            frames.extend(result.sensor_frames)
        table = sensor_metrics(frames)
        table.insert(0, "profile", profile)
        tables.append(table)
    report = pd.concat(tables, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output_dir / "sensor_evaluation.csv", index=False)
    cols = [
        "profile",
        "sensor",
        "miss_rate",
        "pos_rmse_m",
        "outlier_rate",
        "ghosts_per_frame",
        "frozen_frames",
        "mean_latency_s",
    ]
    print(report[cols].to_string(index=False))
    print(f"\nSaved to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
