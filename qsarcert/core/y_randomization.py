"""
Y-Randomization / Y-Scrambling test for chance correlation assessment.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class YRandomizationResult:
    n_iterations: int
    original_r2: float
    mean_scrambled_r2: float
    std_scrambled_r2: float
    max_scrambled_r2: float
    cr2_p: float  # cR^2_p = R * sqrt(R^2 - R_r^2) > 0.50
    scrambled_r2_distribution: List[float]
    status: str  # 'PASS', 'WARNING', 'FAIL'
    diagnostic_message: str


def perform_y_randomization(
    x_features: np.ndarray,
    y_true: np.ndarray,
    original_r2: float,
    n_iterations: int = 100,
    random_seed: int = 42
) -> YRandomizationResult:
    """
    Executes Y-randomization test by permuting target response Y across n_iterations.

    Parameters
    ----------
    x_features : np.ndarray
        Feature matrix (n_samples, n_features).
    y_true : np.ndarray
        True observed response vector.
    original_r2 : float
        The R^2 of the true unpermuted model.
    n_iterations : int, default 100
    random_seed : int, default 42

    Returns
    -------
    result : YRandomizationResult
    """
    x = np.asarray(x_features, dtype=float)
    y = np.asarray(y_true, dtype=float)
    n, p = x.shape
    rng = np.random.default_rng(random_seed)

    scrambled_r2s = []

    # Standardize X for fast ridge least squares
    x_aug = np.hstack([np.ones((n, 1)), x])
    xtx = x_aug.T @ x_aug + 1e-3 * np.eye(p + 1)
    inv_xtx = np.linalg.pinv(xtx)

    for _ in range(n_iterations):
        y_perm = rng.permutation(y)
        
        # Fit analytical weights
        weights = inv_xtx @ (x_aug.T @ y_perm)
        y_pred_perm = x_aug @ weights

        ss_res = np.sum((y_perm - y_pred_perm)**2)
        ss_tot = np.sum((y_perm - np.mean(y_perm))**2)
        r2_perm = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        scrambled_r2s.append(float(np.clip(r2_perm, 0.0, 1.0)))

    scrambled_arr = np.array(scrambled_r2s)
    mean_scrambled = float(np.mean(scrambled_arr))
    std_scrambled = float(np.std(scrambled_arr))
    max_scrambled = float(np.max(scrambled_arr))

    # cR^2_p = R * sqrt(R^2 - \bar{R}_r^2)
    orig_r = np.sqrt(max(0.0, original_r2))
    delta_r = max(0.0, original_r2 - mean_scrambled)
    cr2_p = float(orig_r * np.sqrt(delta_r))

    if cr2_p >= 0.50 and mean_scrambled <= 0.20:
        status = "PASS"
        diag = f"Y-randomization certified robust against chance correlation (cR^2_p = {cr2_p:.3f} >= 0.50, Mean scrambled R^2 = {mean_scrambled:.3f} <= 0.20)."
    elif cr2_p >= 0.35:
        status = "WARNING"
        diag = f"Moderate chance correlation risk (cR^2_p = {cr2_p:.3f}, Mean scrambled R^2 = {mean_scrambled:.3f}). Increase sample size or reduce descriptor count."
    else:
        status = "FAIL"
        diag = f"High chance correlation / overfitting detected (cR^2_p = {cr2_p:.3f} < 0.35, Max scrambled R^2 = {max_scrambled:.3f}). Model learning is uncertified."

    return YRandomizationResult(
        n_iterations=n_iterations,
        original_r2=original_r2,
        mean_scrambled_r2=mean_scrambled,
        std_scrambled_r2=std_scrambled,
        max_scrambled_r2=max_scrambled,
        cr2_p=cr2_p,
        scrambled_r2_distribution=scrambled_r2s,
        status=status,
        diagnostic_message=diag
    )
