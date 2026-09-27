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


def test_metrics_match_independent_formulas():
    """Q2_F1/F2/F3, Pearson r^2, k, k', r_m^2 and CCC against textbook formulas."""
    rng = np.random.default_rng(3)
    y_train = rng.normal(5.0, 1.5, 80)
    y_true = rng.normal(5.3, 1.2, 30)
    y_pred = 0.9 * y_true + 0.4 + rng.normal(0, 0.5, 30)
    r = calculate_oecd_metrics(y_true, y_pred, y_train=y_train)

    press = np.sum((y_true - y_pred) ** 2)
    assert r.q2_f1 == pytest.approx(1 - press / np.sum((y_true - y_train.mean()) ** 2))
    assert r.q2_f2 == pytest.approx(1 - press / np.sum((y_true - y_true.mean()) ** 2))
    assert r.q2_f3 == pytest.approx(1 - (press / 30) / np.var(y_train))
    rp = np.corrcoef(y_true, y_pred)[0, 1] ** 2
    assert r.r2_pearson == pytest.approx(rp)
    k = np.sum(y_true * y_pred) / np.sum(y_pred ** 2)
    assert r.k_slope == pytest.approx(k)
    r0 = 1 - np.sum((y_true - k * y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2)
    assert r.r_m_2 == pytest.approx(rp * (1 - np.sqrt(abs(rp - r0))))
    ccc = 2 * np.cov(y_true, y_pred, bias=True)[0, 1] / (
        y_true.var() + y_pred.var() + (y_true.mean() - y_pred.mean()) ** 2)
    assert r.ccc == pytest.approx(ccc)


def test_q2_f3_requires_training_set():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = calculate_oecd_metrics(y, y + 0.1)
    assert np.isnan(r.q2_f3)
    assert r.q2_f1 == pytest.approx(r.q2_f2)


def test_pearson_r2_differs_from_coefficient_of_determination():
    """A biased but perfectly correlated prediction has r^2 = 1 but R^2 < 1."""
    y = np.linspace(1.0, 10.0, 20)
    r = calculate_oecd_metrics(y, y + 2.0)
    assert r.r2_pearson == pytest.approx(1.0)
    assert r.r2 < 0.9
