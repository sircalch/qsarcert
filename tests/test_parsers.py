"""
Tests for CSV parsers in QSARCert.
"""

import os
import tempfile
import numpy as np
import pytest
from qsarcert.parsers.predictions_csv import parse_predictions_csv
from qsarcert.parsers.feature_matrix import parse_feature_matrix_csv


def test_predictions_csv_parser():
    content = """id,y_true,y_pred,split,feat1,feat2
mol1,5.2,5.1,train,0.5,1.2
mol2,6.1,6.0,train,0.8,1.5
mol3,4.8,4.9,test,0.4,1.1
mol4,7.2,7.0,test,1.0,2.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        f_path = f.name

    try:
        data = parse_predictions_csv(f_path)
        assert data["n_train"] == 2
        assert data["n_eval"] == 2
        assert len(data["y_train"]) == 2
        assert len(data["y_eval"]) == 2
        assert data["x_train"].shape == (2, 2)
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)


def test_feature_matrix_parser():
    content = """feat1,feat2,feat3
0.1,0.2,0.3
0.4,0.5,0.6
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(content)
        f_path = f.name

    try:
        mat = parse_feature_matrix_csv(f_path)
        assert mat.shape == (2, 3)
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)
