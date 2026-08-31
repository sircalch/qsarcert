"""
Tests for OECD validation metrics and Tropsha criteria.
"""

import numpy as np
import pytest
from qsarcert.core.oecd_metrics import calculate_oecd_metrics


def test_oecd_metrics_perfect_fit():
    y_true = np.linspace(1.0, 10.0, 20)
    y_pred = y_true.copy()

    res = calculate_oecd_metrics(y_true, y_pred)
    assert np.isclose(res.r2, 1.0)
    assert np.isclose(res.q2_ext, 1.0)
    assert np.isclose(res.ccc, 1.0)
    assert np.isclose(res.k_slope, 1.0)
    assert res.tropsha_passed is True
    assert res.status == "PASS"


def test_oecd_metrics_poor_fit():
    y_true = np.linspace(1.0, 10.0, 20)
    # Random uncaught predictions
    rng = np.random.default_rng(42)
    y_pred = rng.normal(5.0, 2.0, 20)

    res = calculate_oecd_metrics(y_true, y_pred)
    assert res.r2 < 0.30
    assert res.tropsha_passed is False
    assert res.status == "FAIL"
