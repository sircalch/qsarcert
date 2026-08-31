"""
Parser for chemical feature and descriptor matrices.
"""

from typing import Optional
import os
import pandas as pd
import numpy as np


def parse_feature_matrix_csv(filepath: str) -> np.ndarray:
    """
    Parses a CSV/TSV feature matrix file, stripping non-numeric ID columns.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    features : np.ndarray (n_samples, n_features)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    if filepath.endswith(".npy"):
        return np.load(filepath)

    sep = "\t" if filepath.endswith(".tsv") else ","
    df = pd.read_csv(filepath, sep=sep)

    # Select numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError(f"No numeric feature columns detected in {filepath}")

    return numeric_df.values
