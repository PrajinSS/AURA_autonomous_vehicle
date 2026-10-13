"""Optional supervised baseline: a Random Forest on the same indicators.

The rule-based engine is calibrated on HEALTHY data only. This model is trained
with fault labels, so it shows what supervised learning adds, and where it
breaks (faults it never saw during training).
"""

from __future__ import annotations

import math

import pandas as pd

from health.config import INDICATORS
from health.evaluation import truth_class

SENSORS = ("camera", "lidar", "radar")
FEATURES = (
    [f"{k}_value" for k in INDICATORS]
    + [f"{k}_evidence" for k in INDICATORS]
    + [f"is_{s}" for s in SENSORS]
)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Indicator values and evidence counts. Missing values become -1."""
    out = pd.DataFrame(index=df.index)
    for key in INDICATORS:
        out[f"{key}_value"] = df[f"{key}_value"].fillna(-1.0)
        out[f"{key}_evidence"] = df[f"{key}_evidence"].astype(float)
    for sensor in SENSORS:
        out[f"is_{sensor}"] = (df["sensor"] == sensor).astype(float)
    return out[FEATURES]


class RandomForestHealthModel:
    def __init__(self, n_estimators: int = 200, seed: int = 0) -> None:
        self.n_estimators = n_estimators
        self.seed = seed
        self._model = None

    def fit(
        self,
        df: pd.DataFrame,
        severity_threshold: float = 0.5,
        warmup_s: float = 2.0,
    ) -> RandomForestHealthModel:
        from sklearn.ensemble import RandomForestClassifier  # optional dependency

        work = df[df["time"] >= warmup_s]
        classes = [
            truth_class(sev, severity_threshold, since)
            for sev, since in zip(
                work["truth_severity"], work["since_fault_s"], strict=True
            )
        ]
        work = work.assign(cls=classes)
        work = work[work["cls"] != "ambiguous"]
        if work["cls"].nunique() < 2:
            raise ValueError("training data needs both healthy and faulty rows")
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=self.seed,
            n_jobs=-1,
        ).fit(make_features(work), work["cls"] == "faulty")
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """True where the model says the sensor is faulty (index matches ``df``)."""
        if self._model is None:
            raise RuntimeError("call fit() first")
        return pd.Series(self._model.predict(make_features(df)), index=df.index).astype(
            bool
        )

    def feature_importance(self) -> pd.Series:
        if self._model is None:
            raise RuntimeError("call fit() first")
        imp = pd.Series(self._model.feature_importances_, index=FEATURES)
        return imp.sort_values(ascending=False)


def safe_round(x: float, n: int = 3) -> float:
    return round(x, n) if not math.isnan(x) else x
