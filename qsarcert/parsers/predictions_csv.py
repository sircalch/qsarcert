"""
Parser for QSAR prediction CSV files.
"""

from typing import Dict, Any, Optional
import os
import pandas as pd
import numpy as np


def parse_predictions_csv(filepath: str) -> Dict[str, Any]:
    """
    Parses a CSV file containing observed (y_true) and predicted (y_pred) values.
    Supports optional columns: 'split' ('train', 'test'), 'id', 'smiles', and feature columns.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    data : dict
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    sep = "\t" if filepath.endswith(".tsv") else ","
    df = pd.read_csv(filepath, sep=sep)

    col_map = {}
    for c in df.columns:
        c_low = c.lower().strip()
        if c_low in ["y_true", "observed", "actual", "y_exp", "experimental", "target", "label", "activity"]:
            col_map[c] = "y_true"
        elif c_low in ["y_pred", "predicted", "prediction", "y_hat", "pred"]:
            col_map[c] = "y_pred"
        elif c_low in ["split", "set", "subset", "dataset"]:
            col_map[c] = "split"

    df = df.rename(columns=col_map)

    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"CSV must contain observed and predicted columns (e.g. 'y_true' and 'y_pred'). Found: {list(df.columns)}")

    # Extract optional feature columns (all other numeric columns)
    known_meta = {"y_true", "y_pred", "split", "id", "name", "smiles", "compound"}
    feature_cols = [c for c in df.columns if c not in known_meta and pd.api.types.is_numeric_dtype(df[c])]

    x_features = df[feature_cols].values if feature_cols else None

    # Handle splits
    if "split" in df.columns:
        tr_mask = df["split"].astype(str).str.lower().isin(["train", "training"])
        ev_mask = df["split"].astype(str).str.lower().isin(["test", "eval", "evaluation", "ext", "external", "val", "validation"])

        if tr_mask.sum() > 0 and ev_mask.sum() > 0:
            return {
                "y_train": df.loc[tr_mask, "y_true"].values,
                "y_train_pred": df.loc[tr_mask, "y_pred"].values,
                "y_eval": df.loc[ev_mask, "y_true"].values,
                "y_eval_pred": df.loc[ev_mask, "y_pred"].values,
                "x_train": x_features[tr_mask] if x_features is not None else None,
                "x_eval": x_features[ev_mask] if x_features is not None else None,
                "n_train": int(tr_mask.sum()),
                "n_eval": int(ev_mask.sum())
            }

    return {
        "y_true": df["y_true"].values,
        "y_pred": df["y_pred"].values,
        "x_features": x_features,
        "n_samples": len(df)
    }
