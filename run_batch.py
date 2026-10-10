"""Run many scenarios x seeds and save a results table plus a summary.
Example:
 python run_batch.py --seeds 20
 python run_batch.py --scenarios crossing intersection --seeds 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.runner import aggregate, run_batch
from experiments.scenarios import available_scenarios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch evaluation of the baseline")
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    names = args.scenarios or available_scenarios()
    unknown = set(names) - set(available_scenarios())
    if unknown:
        parser.error(f"unknown scenario(s): {', '.join(sorted(unknown))}")
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    results = run_batch(names, range(args.seeds))
    summary = aggregate(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "batch_results.csv", index=False)
    summary.to_csv(args.output_dir / "batch_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
