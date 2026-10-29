import numpy as np
import pandas as pd
import pytest

try:
    import sklearn
except (ImportError, Exception) as e:
    pytest.skip(f"sklearn unavailable: {e}", allow_module_level=True)

from health.config import INDICATORS  # noqa: E402
from health.ml_baseline import FEATURES, RandomForestHealthModel, make_features  # noqa: E402


def _synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    faulty = rng.random(n) < 0.5
    data = {
        "time": np.full(n, 5.0),
        "sensor": rng.choice(["camera", "lidar", "radar"], n),
        "truth_severity": np.where(faulty, 1.0, 0.0),
        "since_fault_s": np.full(n, np.inf),
    }
    for key in INDICATORS:
        data[f"{key}_value"] = rng.normal(0.9, 0.02, n)
        data[f"{key}_evidence"] = rng.integers(5, 50, n)
    data["completeness_value"] = np.where(
        faulty, rng.normal(0.3, 0.05, n), data["completeness_value"]
    )
    return pd.DataFrame(data)


def test_features_fill_missing_and_one_hot_encode():
    df = _synthetic(10)
    df.loc[0, "noise_value"] = np.nan
    feats = make_features(df)
    assert list(feats.columns) == FEATURES
    assert feats.loc[0, "noise_value"] == -1.0
    assert feats[["is_camera", "is_lidar", "is_radar"]].sum(axis=1).eq(1).all()


def test_model_learns_a_separable_problem():
    train, test = _synthetic(500, 1), _synthetic(200, 2)
    model = RandomForestHealthModel(n_estimators=50).fit(train)
    accuracy = (model.predict(test) == (test["truth_severity"] >= 0.5)).mean()
    assert accuracy > 0.95
    assert model.feature_importance().index[0] == "completeness_value"


def test_predict_before_fit_and_single_class_are_errors():
    with pytest.raises(RuntimeError):
        RandomForestHealthModel().predict(_synthetic(5))
    one_class = _synthetic(50).assign(truth_severity=0.0)
    with pytest.raises(ValueError):
        RandomForestHealthModel().fit(one_class)
