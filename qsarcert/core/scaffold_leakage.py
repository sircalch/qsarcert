"""
Train/Test split leakage, descriptor duplication, and similarity audit.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class SplitLeakageResult:
    n_train: int
    n_test: int
    n_exact_duplicates: int
    n_high_similarity_pairs: int  # Cosine similarity > 0.95
    mean_nn_similarity: float
    max_nn_similarity: float
    status: str  # 'PASS', 'WARNING', 'FAIL'
    diagnostic_message: str


def check_split_leakage(
    x_train: np.ndarray,
    x_test: np.ndarray,
    similarity_threshold: float = 0.95
) -> SplitLeakageResult:
    """
    Evaluates potential data leakage and extreme similarity between train and test splits.

    Parameters
    ----------
    x_train : np.ndarray
        Training feature matrix.
    x_test : np.ndarray
        Test/external feature matrix.
    similarity_threshold : float, default 0.95

    Returns
    -------
    result : SplitLeakageResult
    """
    tr = np.asarray(x_train, dtype=float)
    te = np.asarray(x_test, dtype=float)

    n_tr = len(tr)
    n_te = len(te)

    # Normalize vectors for cosine similarity
    norm_tr = np.linalg.norm(tr, axis=1, keepdims=True)
    norm_te = np.linalg.norm(te, axis=1, keepdims=True)
    norm_tr[norm_tr == 0] = 1.0
    norm_te[norm_te == 0] = 1.0

    tr_u = tr / norm_tr
    te_u = te / norm_te

    # Cosine similarity matrix (n_test x n_train)
    cos_sim = te_u @ tr_u.T

    # Maximum similarity per test sample
    max_sim_per_test = np.max(cos_sim, axis=1)
    
    # Exact duplicates (similarity >= 0.9999)
    exact_dups = int(np.sum(max_sim_per_test >= 0.9999))
    high_sim = int(np.sum(max_sim_per_test >= similarity_threshold))

    mean_nn = float(np.mean(max_sim_per_test))
    max_nn = float(np.max(max_sim_per_test))

    if exact_dups == 0 and high_sim <= max(1, int(0.05 * n_te)):
        status = "PASS"
        diag = f"Clean train/test split without data leakage (0 exact duplicates, Mean NN similarity = {mean_nn:.3f})."
    elif exact_dups == 0 and high_sim <= int(0.20 * n_te):
        status = "WARNING"
        diag = f"Moderate train/test similarity ({high_sim} test compounds share > {similarity_threshold*100:.0f}% similarity with train). Scaffold-based split recommended."
    else:
        status = "FAIL"
        diag = f"Data leakage detected! Found {exact_dups} exact duplicate(s) and {high_sim} highly identical compounds in test set. Metrics may be overly optimistic."

    return SplitLeakageResult(
        n_train=n_tr,
        n_test=n_te,
        n_exact_duplicates=exact_dups,
        n_high_similarity_pairs=high_sim,
        mean_nn_similarity=mean_nn,
        max_nn_similarity=max_nn,
        status=status,
        diagnostic_message=diag
    )
