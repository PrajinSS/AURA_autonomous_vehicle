"""Grade the sensor-health engine against injected faults.

Example:
    python evaluate_health.py --scenarios cluttered --seeds 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from experiments.scenarios import available_scenarios
from health.evaluation import classification_metrics, collect_dataset, detection_delays
from sensors.profiles import available_fault_profiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sensor-health engine evaluation")
    parser.add_argument("--scenarios", nargs="+", default=["cluttered"])
    parser.add_argument("--profiles", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument(
        "--first-seed",
        type=int,
        default=0,
        help="use seeds first-seed ... first-seed+seeds-1 (calibration used 0-5)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="'clearly faulty' severity"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)

    profiles = args.profiles or available_fault_profiles()
    for name in args.scenarios:
        if name not in available_scenarios():
            parser.error(f"unknown scenario '{name}'")
    unknown = set(profiles) - set(available_fault_profiles())
    if unknown:
        parser.error(f"unknown profile(s): {', '.join(sorted(unknown))}")

    data = collect_dataset(
        args.scenarios, profiles, range(args.first_seed, args.first_seed + args.seeds)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output_dir / "health_dataset.csv", index=False)

    print("\nOVERALL (all profiles pooled)")
    print(classification_metrics(data, args.threshold).to_string(index=False))

    delays = []
    for (_scenario, profile, _seed), episode in data.groupby(
        ["scenario", "profile", "seed"]
    ):
        d = detection_delays(episode, args.threshold)
        d.insert(0, "profile", profile)
        delays.append(d)
    delay_table = pd.concat(delays, ignore_index=True)
    summary = (
        delay_table.groupby(["profile", "sensor"])
        .agg(
            episodes=("delay_s", "size"),
            detected=("delay_s", "count"),
            mean_delay_s=("delay_s", "mean"),
        )
        .round(2)
        .reset_index()
    )
    print("\nDETECTION DELAY (seconds from 'clearly faulty' to first alarm)")
    print(summary.to_string(index=False))

    print(
        "\nFALSE ALARMS ON HEALTHY SENSORS, PER PROFILE (rows where truth is healthy)"
    )
    work = data[
        (data["time"] >= 2.0)
        & (data["truth_severity"] <= 0.0)
        & (data["since_fault_s"] >= 3.0)
    ]
    fa = (
        work.assign(alarm=work["status"] != "healthy")
        .groupby(["profile", "sensor"])["alarm"]
        .mean()
    )
    print(fa.round(3).unstack().to_string())
    summary.to_csv(args.output_dir / "health_detection_delay.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
