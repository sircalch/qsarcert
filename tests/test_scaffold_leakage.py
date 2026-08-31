"""
Tests for split leakage and descriptor duplication.
"""

import numpy as np
import pytest
from qsarcert.core.scaffold_leakage import check_split_leakage


def test_split_leakage_clean():
    rng = np.random.default_rng(42)
    x_tr = rng.normal(0, 1, size=(40, 5))
    x_te = rng.normal(5, 1, size=(10, 5))

    res = check_split_leakage(x_tr, x_te)
    assert res.n_exact_duplicates == 0
    assert res.status == "PASS"


def test_split_leakage_duplicate():
    rng = np.random.default_rng(42)
    x_tr = rng.normal(0, 1, size=(20, 5))
    # Copy first 5 training rows to test
    x_te = x_tr[:5].copy()

    res = check_split_leakage(x_tr, x_te)
    assert res.n_exact_duplicates == 5
    assert res.status == "FAIL"
