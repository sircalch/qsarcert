"""
Generates synthetic QSAR dataset CSV with features, splits, and targets for tutorials.
"""

import os
import pandas as pd
import numpy as np


def generate_sample_qsar_data(output_dir: str = "sample_qsar_dataset"):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    n_train = 120
    n_test = 30
    p = 6

    # Random features (MW, LogP, TPSA, HBD, HBA, RotB)
    feature_names = ["MW", "LogP", "TPSA", "HBD", "HBA", "RotB"]
    weights = np.array([0.5, 1.2, -0.8, 0.4, -0.6, 0.3])

    x_tr = rng.normal(0, 1.0, size=(n_train, p))
    y_tr = x_tr @ weights + 5.0 + rng.normal(0, 0.25, size=n_train)
    y_pred_tr = y_tr + rng.normal(0, 0.20, size=n_train)

    x_te = rng.normal(0, 1.0, size=(n_test, p))
    # 1 outlier compound
    x_te[-1] = x_te[-1] * 2.5
    y_te = x_te @ weights + 5.0 + rng.normal(0, 0.30, size=n_test)
    y_pred_te = y_te + rng.normal(0, 0.25, size=n_test)

    rows = []
    # Train
    for i in range(n_train):
        d = {
            "id": f"mol_tr_{i:03d}",
            "y_true": float(y_tr[i]),
            "y_pred": float(y_pred_tr[i]),
            "split": "train"
        }
        for j, fn in enumerate(feature_names):
            d[fn] = float(x_tr[i, j])
        rows.append(d)

    # Test
    for i in range(n_test):
        d = {
            "id": f"mol_te_{i:03d}",
            "y_true": float(y_te[i]),
            "y_pred": float(y_pred_te[i]),
            "split": "test"
        }
        for j, fn in enumerate(feature_names):
            d[fn] = float(x_te[i, j])
        rows.append(d)

    df = pd.DataFrame(rows)
    csv_p = os.path.join(output_dir, "qsar_predictions_with_features.csv")
    df.to_csv(csv_p, index=False)
    print(f"Generated sample QSAR dataset ({n_train} train, {n_test} test) at: {os.path.abspath(csv_p)}")


if __name__ == "__main__":
    generate_sample_qsar_data()
