import math

import numpy as np
import pytest

from simulation.shapes import Circle, Rectangle


def test_circle_distance_and_inside():
    c = Circle(2.0)
    assert math.isclose(float(c.distance_rel(5.0, 0.0)), 3.0)
    assert float(c.distance_rel(1.0, 0.0)) == 0.0


def test_circle_closest_point_on_surface():
    px, py = Circle(2.0).closest_point_rel(10.0, 0.0)
    assert math.isclose(px, 2.0) and math.isclose(py, 0.0)


def test_rectangle_distance_side_and_corner():
    r = Rectangle(4.0, 2.0)  # half sizes 2 and 1
    assert math.isclose(float(r.distance_rel(5.0, 0.0)), 3.0)
    assert math.isclose(float(r.distance_rel(5.0, 5.0)), math.hypot(3.0, 4.0))
    assert float(r.distance_rel(0.5, 0.5)) == 0.0


def test_rectangle_closest_point_clamped():
    assert Rectangle(4.0, 2.0).closest_point_rel(9.0, -9.0) == (2.0, -1.0)


def test_distance_accepts_numpy_arrays():
    d = Circle(1.0).distance_rel(np.array([0.0, 3.0]), np.array([0.0, 0.0]))
    assert np.allclose(d, [0.0, 2.0])


def test_invalid_shapes_rejected():
    with pytest.raises(ValueError):
        Circle(0.0)
    with pytest.raises(ValueError):
        Rectangle(1.0, -1.0)
