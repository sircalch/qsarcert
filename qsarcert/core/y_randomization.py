"""
Y-Randomization / Y-Scrambling test for chance correlation assessment.

Two procedures are available:

* surrogate (default): ordinary least squares on the descriptors, with the training R^2 of the
  original and of the permuted responses, as in the classical test for MLR models. For random
  responses its expected R^2 is p/(n - 1) (Ruecker et al. 2007), so it is informative only when the
  descriptors are few compared with the compounds; with p >= (n - 1)/2 it is NOT_APPLICABLE.
* estimator: the user's own model (any object with fit/predict, e.g. scikit-learn) is refitted on
  the permuted responses and scored by k-fold cross-validated R^2. This is required for flexible
  models such as random forests, whose training R^2 is high even for random responses.
"""

import copy
from typing import List, Optional, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class YRandomizationResult:
    n_iterations: int
    original_r2: float
    mean_scrambled_r2: float
    std_scrambled_r2: float
    max_scrambled_r2: float
    cr2_p: float  # cR^2_p = R * sqrt(R^2 - mean R_r^2) > 0.50
    scrambled_r2_distribution: List[float]
    status: str  # 'PASS', 'WARNING', 'FAIL', 'NOT_APPLICABLE'
    diagnostic_message: str
    procedure: str = "surrogate"  # 'surrogate' (OLS training R^2) or 'estimator' (k-fold CV R^2)
    expected_chance_r2: float = float("nan")  # p/(n - 1) for the OLS surrogate
    p_value: float = float("nan")  # (1 + #{R_r^2 >= R^2}) / (1 + n_iterations)


def _ols_r2(x_aug, y):
    try:
        coef, *_ = np.linalg.lstsq(x_aug, y, rcond=None)
    except np.linalg.LinAlgError:  # SVD-based driver did not converge; QR with pivoting instead
        from scipy.linalg import lstsq
        coef, *_ = lstsq(x_aug, y, lapack_driver="gelsy")
    res = np.sum((y - x_aug @ coef) ** 2)
    tot = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - res / tot) if tot > 0 else 0.0


def _cv_r2(estimator, x, y, folds):
    pred = np.empty_like(y)
    for k in range(folds.max() + 1):
        te = folds == k
        try:
            from sklearn.base import clone
            m = clone(estimator)
        except Exception:  # not a scikit-learn estimator
            m = copy.deepcopy(estimator)
        m.fit(x[~te], y[~te])
        pred[te] = np.asarray(m.predict(x[te]), dtype=float).ravel()
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2))


def perform_y_randomization(
    x_features: np.ndarray,
    y_true: np.ndarray,
    original_r2: Optional[float] = None,
    n_iterations: int = 100,
    random_seed: int = 42,
    estimator: Any = None,
    cv_folds: int = 5
) -> YRandomizationResult:
    """
    Executes the Y-randomization test by permuting the responses n_iterations times.

    Parameters
    ----------
    x_features : np.ndarray
        Training descriptor matrix (n_samples, n_features).
    y_true : np.ndarray
        Training responses.
    original_r2 : float, optional
        Ignored; kept for backward compatibility. The reference R^2 is always recomputed with the
        same procedure as the permuted ones, so that cR^2_p compares like with like.
    n_iterations : int, default 100
    random_seed : int, default 42
    estimator : object with fit/predict, optional
        The user's model. When given, the original and permuted responses are scored by k-fold
        cross-validated R^2 of this model instead of the OLS surrogate.
    cv_folds : int, default 5

    Returns
    -------
    result : YRandomizationResult
    """
    x = np.asarray(x_features, dtype=float)
    y = np.asarray(y_true, dtype=float)
    n, p = x.shape
    rng = np.random.default_rng(random_seed)

    if estimator is not None:
        procedure = "estimator"
        folds = rng.permutation(np.arange(n) % cv_folds)
        expected = float("nan")

        def score(y_vec):
            return _cv_r2(estimator, x, y_vec, folds)
    else:
        procedure = "surrogate"
        sd = np.std(x, axis=0)
        sd[sd == 0] = 1.0
        x_aug = np.hstack([np.ones((n, 1)), (x - np.mean(x, axis=0)) / sd])
        try:
            rank = int(np.linalg.matrix_rank(x_aug))
        except np.linalg.LinAlgError:
            from scipy.linalg import svd
            sv = svd(x_aug, compute_uv=False, lapack_driver="gesvd")
            rank = int(np.sum(sv > sv.max() * max(x_aug.shape) * np.finfo(float).eps))
        expected = float((rank - 1) / (n - 1)) if n > 1 else float("nan")

        def score(y_vec):
            return _ols_r2(x_aug, y_vec)

        if expected >= 0.5:
            return YRandomizationResult(
                n_iterations=0, original_r2=float("nan"), mean_scrambled_r2=float("nan"),
                std_scrambled_r2=float("nan"), max_scrambled_r2=float("nan"), cr2_p=float("nan"),
                scrambled_r2_distribution=[], status="NOT_APPLICABLE",
                diagnostic_message=(f"The OLS surrogate cannot test chance correlation with {rank - 1} independent"
                                    f" descriptors and {n} compounds: random responses already give R^2 ="
                                    f" {expected:.2f} on average. Pass the model as `estimator` to use"
                                    f" cross-validated R^2."),
                procedure=procedure, expected_chance_r2=expected)

    r2_orig = score(y)
    scrambled = np.array([score(rng.permutation(y)) for _ in range(n_iterations)])
    mean_s = float(np.mean(scrambled))
    std_s = float(np.std(scrambled))
    max_s = float(np.max(scrambled))
    p_value = float((1 + np.sum(scrambled >= r2_orig)) / (1 + n_iterations))

    # cR^2_p = R sqrt(R^2 - mean R_r^2) (Todeschini); negative CV R^2 values are floored at zero
    r2_pos = max(0.0, r2_orig)
    cr2_p = float(np.sqrt(r2_pos) * np.sqrt(max(0.0, r2_pos - max(0.0, mean_s))))
    what = "5-fold CV R^2 of the model" if procedure == "estimator" and cv_folds == 5 else \
        (f"{cv_folds}-fold CV R^2 of the model" if procedure == "estimator" else "OLS training R^2")

    if cr2_p >= 0.50 and mean_s <= 0.20:
        status = "PASS"
        diag = (f"No chance correlation ({what}): R^2 = {r2_orig:.3f} against a mean of {mean_s:.3f} for"
                f" permuted responses (cR^2_p = {cr2_p:.3f} >= 0.50, p = {p_value:.3f}).")
    elif cr2_p >= 0.35:
        status = "WARNING"
        diag = (f"Moderate chance-correlation risk ({what}): R^2 = {r2_orig:.3f}, permuted mean {mean_s:.3f},"
                f" cR^2_p = {cr2_p:.3f}.")
    else:
        status = "FAIL"
        diag = (f"Chance correlation cannot be excluded ({what}): R^2 = {r2_orig:.3f}, permuted mean"
                f" {mean_s:.3f} (max {max_s:.3f}), cR^2_p = {cr2_p:.3f} < 0.35.")

    return YRandomizationResult(
        n_iterations=n_iterations,
        original_r2=r2_orig,
        mean_scrambled_r2=mean_s,
        std_scrambled_r2=std_s,
        max_scrambled_r2=max_s,
        cr2_p=cr2_p,
        scrambled_r2_distribution=scrambled.tolist(),
        status=status,
        diagnostic_message=diag,
        procedure=procedure,
        expected_chance_r2=expected,
        p_value=p_value,
    )
