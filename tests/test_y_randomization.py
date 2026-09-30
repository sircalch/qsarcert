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


def test_surrogate_expected_chance_r2_and_not_applicable():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(101, 10))
    y = rng.normal(size=101)
    res = perform_y_randomization(x, y, n_iterations=400)
    assert res.expected_chance_r2 == pytest.approx(10 / 100)
    assert res.mean_scrambled_r2 == pytest.approx(0.10, abs=0.01)  # E[R^2] = p/(n-1) under the null
    wide = perform_y_randomization(rng.normal(size=(30, 40)), rng.normal(size=30))
    assert wide.status == "NOT_APPLICABLE"


def test_estimator_uses_cross_validated_r2():
    from sklearn.ensemble import RandomForestRegressor
    rng = np.random.default_rng(3)
    x = rng.integers(0, 2, size=(80, 300)).astype(float)
    y = x[:, :5].sum(1) + rng.normal(0, 0.3, 80)
    res = perform_y_randomization(x, y, n_iterations=5, estimator=RandomForestRegressor(n_estimators=50, random_state=0))
    assert res.procedure == "estimator"
    assert res.original_r2 > 0.3
    assert res.mean_scrambled_r2 < 0.1
