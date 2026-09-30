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


def test_raw_descriptors_are_not_flagged_by_cosine():
    rng = np.random.default_rng(0)
    # molecular-weight-like column dominates the raw vectors
    x_tr = np.column_stack([rng.normal(300, 50, 100), rng.normal(0, 1, (100, 3))])
    x_te = np.column_stack([rng.normal(300, 50, 20), rng.normal(0, 1, (20, 3))])
    res = check_split_leakage(x_tr, x_te)
    assert res.n_exact_duplicates == 0
    assert res.status == "PASS"


def test_fingerprints_use_tanimoto_and_smiles_duplicates():
    fp_tr = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]])
    fp_te = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    res = check_split_leakage(fp_tr, fp_te)
    assert res.similarity == "tanimoto"
    assert res.n_exact_duplicates == 1
    assert res.max_nn_similarity == 1.0
    pytest.importorskip("rdkit")
    res = check_split_leakage(fp_tr, fp_te, smiles_train=["CCO", "c1ccccc1O", "CCN"], smiles_test=["OCC", "c1ccccc1N"])
    assert res.n_duplicate_structures == 1
    assert res.frac_shared_scaffold == 1.0
    assert res.status == "FAIL"
