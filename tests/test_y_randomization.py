"""
Tests for Y-Randomization scrambling tests.
"""

import numpy as np
import pytest
from qsarcert.core.y_randomization import perform_y_randomization


def test_y_randomization_true_signal():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, size=(50, 4))
    y = x @ np.array([2.0, -1.5, 3.0, 0.5]) + rng.normal(0, 0.2, 50)

    res = perform_y_randomization(x, y, original_r2=0.95, n_iterations=30)
    assert res.n_iterations == 30
    assert res.mean_scrambled_r2 < 0.25
    assert res.cr2_p > 0.50
    assert res.status == "PASS"
