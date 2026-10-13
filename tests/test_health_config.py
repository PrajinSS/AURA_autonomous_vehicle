import pytest

from health.config import INDICATORS, Calibration, HealthConfig


def test_calibration_maps_nominal_to_one_and_floor_to_zero():
    higher_is_better = Calibration(nominal=0.9, floor=0.3)
    assert higher_is_better.score(0.9) == 1.0
    assert higher_is_better.score(0.3) == 0.0
    assert higher_is_better.score(0.6) == pytest.approx(0.5)
    assert higher_is_better.score(5.0) == 1.0  # clipped


def test_calibration_works_for_lower_is_better():
    lower_is_better = Calibration(nominal=0.0, floor=0.4)
    assert lower_is_better.score(0.0) == 1.0
    assert lower_is_better.score(0.2) == pytest.approx(0.5)
    assert lower_is_better.score(0.9) == 0.0


def test_shrinkage_pulls_towards_nominal_when_evidence_is_thin():
    cal = Calibration(nominal=0.9, floor=0.3, prior_strength=10)
    assert cal.shrink(0.1, evidence=0) == pytest.approx(0.9)
    assert cal.shrink(0.1, evidence=1000) == pytest.approx(0.1, abs=0.01)
    assert 0.1 < cal.shrink(0.1, evidence=10) < 0.9


def test_calibration_rejects_equal_nominal_and_floor():
    with pytest.raises(ValueError):
        Calibration(nominal=1.0, floor=1.0)


def test_default_config_covers_every_indicator():
    cfg = HealthConfig()
    assert set(cfg.calibrations) == set(cfg.weights) == set(INDICATORS)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(failed_below=0.8, degraded_below=0.6),
        dict(window_s=0.0),
        dict(tau_up_s=-1.0),
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        HealthConfig(**kwargs)
