import math

import pytest

from simulation.clock import SimulationClock


def test_no_float_drift_after_many_ticks():
    clock = SimulationClock(0.05)
    for _ in range(100_000):
        clock.tick()
    assert math.isclose(clock.time, 5000.0, rel_tol=1e-12)


def test_reset_and_steps_for():
    clock = SimulationClock(0.1)
    clock.tick()
    clock.reset()
    assert clock.time == 0.0
    assert clock.steps_for(2.0) == 20


def test_invalid_dt():
    with pytest.raises(ValueError):
        SimulationClock(0.0)
