"""Rule-based health engine versus a supervised Random Forest.

Two questions:
    1. SAME faults in training and test (different random seeds): who is better?
    2. UNSEEN fault type (leave one fault profile out of training): who generalises?

Needs scikit-learn (free): pip install scikit-learn

Example:
    python compare_health_models.py --train-seeds 5 --test-seeds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from experiments.scenarios import available_scenarios
from health.evaluation import classification_metrics, collect_dataset
from health.ml_baseline import RandomForestHealthModel, safe_round
from sensors.profiles import available_fault_profiles

TEST_SEED_START = 100  # calibration and training only ever use seeds below this


def _pooled(
    df: pd.DataFrame, pred: pd.Series | None, threshold: float
) -> dict[str, float]:
    m = classification_metrics(df, threshold, predicted_faulty=pred)
    return {
        "accuracy": safe_round(m["accuracy"].mean()),
        "precision": safe_round(m["precision"].mean()),
        "recall": safe_round(m["recall"].mean()),
        "false_alarm_rate": safe_round(m["false_alarm_rate"].mean()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rule-based vs Random Forest")
    parser.add_argument(
        "--scenarios", nargs="+", default=["cluttered", "warehouse", "static"]
    )
    parser.add_argument("--train-seeds", type=int, default=5, help="seeds 0..N-1")
    parser.add_argument(
        "--test-seeds",
        type=int,
        default=5,
        help=f"seeds {TEST_SEED_START}..: never used for tuning or training",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--dataset", type=Path, default=None, help="reuse a saved dataset CSV"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)

    for name in args.scenarios:
        if name not in available_scenarios():
            parser.error(f"unknown scenario '{name}'")
    if args.train_seeds < 1 or args.test_seeds < 1:
        parser.error("--train-seeds and --test-seeds must be >= 1")

    if args.dataset and args.dataset.exists():
        data = pd.read_csv(args.dataset)
    else:
        data = collect_dataset(
            args.scenarios,
            available_fault_profiles(),
            [
                *range(args.train_seeds),
                *range(TEST_SEED_START, TEST_SEED_START + args.test_seeds),
            ],
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.output_dir / "health_dataset.csv", index=False)

    train = data[data["seed"] < TEST_SEED_START]
    test = data[data["seed"] >= TEST_SEED_START]

    model = RandomForestHealthModel().fit(train, args.threshold)
    rows = [
        {
            "model": "rule-based (calibrated on healthy data only)",
            **_pooled(test, None, args.threshold),
        },
        {
            "model": "random forest (trained with fault labels)",
            **_pooled(test, model.predict(test), args.threshold),
        },
    ]
    print("\n1) SAME FAULT TYPES, NEW RANDOM SEEDS (held-out test seeds)")
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n2) UNSEEN FAULT TYPE: random forest trained WITHOUT the profile shown")
    rows = []
    for profile in available_fault_profiles():
        if profile == "nominal":
            continue
        held_out = test[test["profile"] == profile]
        loo = RandomForestHealthModel().fit(
            train[train["profile"] != profile], args.threshold
        )
        rule = _pooled(held_out, None, args.threshold)
        rf = _pooled(held_out, loo.predict(held_out), args.threshold)
        rows.append(
            {
                "held_out_profile": profile,
                "rule_recall": rule["recall"],
                "rf_recall": rf["recall"],
                "rule_false_alarm": rule["false_alarm_rate"],
                "rf_false_alarm": rf["false_alarm_rate"],
            }
        )
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    print(
        "\nmean recall on unseen fault types: "
        f"rule-based {table['rule_recall'].mean():.3f}, "
        f"random forest {table['rf_recall'].mean():.3f}"
    )

    print("\nMost important features (random forest):")
    print(model.feature_importance().head(6).round(3).to_string())

    table.to_csv(args.output_dir / "health_model_comparison.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
