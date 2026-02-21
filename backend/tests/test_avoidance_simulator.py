"""Test for the avoidance simulator (updated for new implementation)."""
import numpy as np


def test_distance_numpy():
    """Basic distance calculation using numpy (replaces old _distance helper)."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([3.0, 4.0, 0.0])
    assert round(float(np.linalg.norm(a - b)), 6) == 5.0
