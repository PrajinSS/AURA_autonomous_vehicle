"""Measure indicator values on HEALTHY runs to (re)calibrate the health engine.

Run this whenever a sensor model changes. Compare the percentiles with the
``nominal`` and ``floor`` values in ``health/config.py``: 'nominal' should sit
just inside the healthy range (around the 5th percentile for 'higher is better'
indicators, the 95th for 'lower is better' ones).

Example:
    python calibrate_health.py --seeds 6
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from experiments.scenarios import available_scenarios
from health.config import INDICATORS, HealthConfig
from health.evaluation import collect_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate health indicators on healthy data"
    )
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=6)
    args = parser.parse_args(argv)

    scenarios = args.scenarios or available_scenarios()
    data = collect_dataset(scenarios, ["nominal"], range(args.seeds))
    data = data[data["time"] >= 2.0]
    cfg = HealthConfig()
    rows = []
    for key in INDICATORS:
        cal = cfg.calibrations[key]
        for sensor, g in data.groupby("sensor"):
            v = g.loc[g[f"{key}_evidence"] >= cfg.min_evidence, f"{key}_value"].dropna()
            if v.empty:
                continue
            rows.append(
                {
                    "indicator": key,
                    "sensor": sensor,
                    "n": len(v),
                    "p1": v.quantile(0.01),
                    "p5": v.quantile(0.05),
                    "median": v.median(),
                    "p95": v.quantile(0.95),
                    "p99": v.quantile(0.99),
                    "config_nominal": cal.nominal,
                    "config_floor": cal.floor,
                }
            )
    table = pd.DataFrame(rows).round(3)
    print(table.to_string(index=False))
    alarms = (
        data.assign(alarm=data["status"] != "healthy").groupby("sensor")["alarm"].mean()
    )
    print("\nFalse-alarm rate on healthy data:", alarms.round(4).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
