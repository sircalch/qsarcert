"""
Tests for applicability domain and Williams plot calculations.
"""

import numpy as np
import pytest
from qsarcert.core.applicability_domain import calculate_applicability_domain


def test_applicability_domain_in_domain():
    rng = np.random.default_rng(42)
    n_tr = 50
    n_ev = 15
    p = 4

    x_train = rng.normal(0, 1, size=(n_tr, p))
    x_eval = rng.normal(0, 1, size=(n_ev, p))

    weights = np.array([1.0, 2.0, -1.0, 0.5])
    y_true = x_eval @ weights
    y_pred = y_true + rng.normal(0, 0.1, size=n_ev)

    res = calculate_applicability_domain(x_train, x_eval, y_true, y_pred)
    assert res.n_train_samples == 50
    assert res.n_features == 4
    assert res.warning_leverage > 0.0
    assert res.pct_in_domain >= 80.0
    assert res.status in ["PASS", "WARNING"]


def test_applicability_domain_outlier_detection():
    rng = np.random.default_rng(42)
    n_tr = 40
    n_ev = 10
    p = 3

    x_train = rng.normal(0, 1, size=(n_tr, p))
    x_eval = rng.normal(0, 1, size=(n_ev, p))
    # Create extreme outlier
    x_eval[0] = x_eval[0] * 10.0

    y_true = np.ones(n_ev) * 5.0
    y_pred = y_true.copy()
    y_pred[0] = -50.0  # Extreme residual

    res = calculate_applicability_domain(x_train, x_eval, y_true, y_pred)
    assert res.n_influential_outliers >= 1
    assert res.status == "FAIL"
