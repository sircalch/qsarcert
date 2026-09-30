"""
Statistical validation metrics, Tropsha-Golbraikh criteria, and Roy modified r_m^2 metrics.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class OECDValidationResult:
    n_samples: int
    r2: float  # Coefficient of determination of the evaluation set, 1 - SS_res/SS_tot
    r2_pearson: float  # Squared Pearson correlation r^2 (used by Golbraikh-Tropsha and Roy r_m^2)
    q2_ext: float  # Predictive squared correlation (R^2_pred)
    q2_f1: float
    q2_f2: float
    q2_f3: float  # NaN unless the training set is given (needs the training variance)
    ccc: float  # Concordance Correlation Coefficient
    mae: float
    rmse: float
    k_slope: float  # Slope of regression through origin y vs y_pred (0.85 <= k <= 1.15)
    k_prime_slope: float  # Slope of regression through origin y_pred vs y (0.85 <= k' <= 1.15)
    r0_2: float
    r0_prime_2: float
    tropsha_r2_diff: float  # |r^2 - R0^2| / r^2 (< 0.1), r^2 = squared Pearson
    tropsha_r2_prime_diff: float  # |r^2 - R'0^2| / r^2 (< 0.1)
    r_m_2: float
    r_m_prime_2: float
    r_m_average: float  # (r_m^2 + r'_m^2)/2 > 0.5
    delta_r_m_2: float  # |r_m^2 - r'_m^2| < 0.2
    r0_diff: float  # |R0^2 - R'0^2| < 0.3
    tropsha_passed: bool
    status: str  # 'PASS', 'WARNING', 'FAIL'
    diagnostic_message: str


def calculate_oecd_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_mean: Optional[float] = None,
    min_pass_q2: float = 0.60,
    min_warn_q2: float = 0.50,
    y_train: Optional[np.ndarray] = None
) -> OECDValidationResult:
    """
    Computes all standard OECD & Tropsha-Golbraikh QSAR validation metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Observed experimental values.
    y_pred : np.ndarray
        Model predicted values.
    y_train_mean : float, optional
        Mean of the training set (used for Q2_F1). Taken from y_train when that is given.
    y_train : np.ndarray, optional
        Training responses. Needed for Q2_F1 (training mean) and Q2_F3 (training variance);
        without it Q2_F1 falls back to the evaluation mean (= Q2_F2) and Q2_F3 is NaN.

    Notes
    -----
    Definitions follow Consonni, Ballabio & Todeschini (2009) for Q2_F1/F2/F3, Lin (1989) for
    CCC, Golbraikh & Tropsha (2002) and Roy et al. (2012) for the regression-through-origin
    criteria, where r^2 is the squared Pearson correlation between observed and predicted values.
    min_pass_q2 : float, default 0.60
    min_warn_q2 : float, default 0.50

    Returns
    -------
    result : OECDValidationResult
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    n = len(y_t)

    if n < 3:
        raise ValueError("At least 3 samples are required for OECD statistical validation.")

    mean_t = float(np.mean(y_t))
    mean_p = float(np.mean(y_p))
    y_tr = np.asarray(y_train, dtype=float) if y_train is not None else None
    if y_tr is not None and len(y_tr) > 1:
        mean_tr = float(np.mean(y_tr))
    else:
        mean_tr = float(y_train_mean) if y_train_mean is not None else mean_t

    ss_res = float(np.sum((y_t - y_p)**2))
    ss_tot = float(np.sum((y_t - mean_t)**2))
    ss_tot_train = float(np.sum((y_t - mean_tr)**2))

    # 1. R^2 & Q^2_ext
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    q2_ext = 1.0 - (ss_res / ss_tot_train) if ss_tot_train > 0 else 0.0

    # 2. Q2_F1, Q2_F2, Q2_F3
    q2_f1 = q2_ext
    q2_f2 = 1.0 - (ss_res / float(np.sum((y_t - mean_t)**2))) if ss_tot > 0 else 0.0
    if y_tr is not None and len(y_tr) > 1 and float(np.var(y_tr)) > 0:
        q2_f3 = 1.0 - ((ss_res / n) / float(np.mean((y_tr - np.mean(y_tr)) ** 2)))
    else:
        q2_f3 = float("nan")   # requires the training variance; the evaluation variance would give Q2_F2

    # 3. MAE & RMSE
    mae = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p)**2)))

    # 4. Concordance Correlation Coefficient (CCC)
    cov_tp = float(np.mean((y_t - mean_t) * (y_p - mean_p)))
    var_t = float(np.var(y_t))
    var_p = float(np.var(y_p))
    ccc_denom = var_t + var_p + (mean_t - mean_p)**2
    ccc = float(2.0 * cov_tp / ccc_denom) if ccc_denom > 0 else 0.0

    # 5. Slopes of regression through origin:
    # k = \sum (y_t * y_p) / \sum (y_p^2)
    # k' = \sum (y_t * y_p) / \sum (y_t^2)
    sum_prod = float(np.sum(y_t * y_p))
    sum_p2 = float(np.sum(y_p**2))
    sum_t2 = float(np.sum(y_t**2))

    k = sum_prod / sum_p2 if sum_p2 > 0 else 1.0
    k_prime = sum_prod / sum_t2 if sum_t2 > 0 else 1.0

    # 6. R0^2 & R'0^2
    ss_r0 = float(np.sum((y_t - k * y_p)**2))
    ss_r0_prime = float(np.sum((y_p - k_prime * y_t)**2))
    
    r0_2 = 1.0 - (ss_r0 / ss_tot) if ss_tot > 0 else 0.0
    r0_prime_2 = 1.0 - (ss_r0_prime / float(np.sum((y_p - mean_p)**2))) if float(np.sum((y_p - mean_p)**2)) > 0 else 0.0

    # 7. Squared Pearson correlation and Tropsha differences
    sd_t, sd_p = float(np.std(y_t)), float(np.std(y_p))
    r2_pearson = float((cov_tp / (sd_t * sd_p)) ** 2) if sd_t > 0 and sd_p > 0 else 0.0
    diff_r0 = abs(r2_pearson - r0_2) / r2_pearson if r2_pearson > 0 else float("inf")
    diff_r0_prime = abs(r2_pearson - r0_prime_2) / r2_pearson if r2_pearson > 0 else float("inf")
    r0_diff = abs(r0_2 - r0_prime_2)

    # 8. Roy r_m^2 metrics (with r^2 = squared Pearson correlation)
    term1 = np.sqrt(abs(r2_pearson - r0_2))
    term2 = np.sqrt(abs(r2_pearson - r0_prime_2))

    r_m_2 = float(r2_pearson * (1.0 - term1))
    r_m_prime_2 = float(r2_pearson * (1.0 - term2))
    r_m_avg = float((r_m_2 + r_m_prime_2) / 2.0)
    delta_r_m_2 = float(abs(r_m_2 - r_m_prime_2))

    # 9. Tropsha acceptance evaluation
    c1 = (q2_ext > min_warn_q2)
    c2 = (r2_pearson > 0.60)                     # Golbraikh & Tropsha: r^2 > 0.6
    c3 = (0.85 <= k <= 1.15) or (0.85 <= k_prime <= 1.15)
    c4 = ((diff_r0 < 0.10) or (diff_r0_prime < 0.10)) and (r0_diff < 0.30)
    c5 = (delta_r_m_2 < 0.20) and (r_m_avg > min_warn_q2)
    c6 = (ccc >= 0.80)

    tropsha_passed = bool(c1 and c2 and c3 and c4 and c5 and c6)

    failed_criteria = []
    if not c1:
        failed_criteria.append(f"Q^2_ext = {q2_ext:.2f} <= {min_warn_q2}")
    if not c2:
        failed_criteria.append(f"r^2 = {r2_pearson:.2f} <= 0.60")
    if not c3:
        failed_criteria.append(f"Slope k = {k:.2f} out of [0.85, 1.15]")
    if not c4:
        failed_criteria.append(f"|r^2 - R0^2|/r^2 = {diff_r0:.2f} (>= 0.10) or |R0^2 - R'0^2| = {r0_diff:.2f} (>= 0.30)")
    if not c5:
        failed_criteria.append(f"r_m^2 avg = {r_m_avg:.2f} <= 0.50 or delta_r_m = {delta_r_m_2:.2f} >= 0.20")
    if not c6:
        failed_criteria.append(f"CCC = {ccc:.2f} < 0.80")

    if tropsha_passed and q2_ext >= min_pass_q2 and ccc >= 0.85:
        status = "PASS"
        diag = f"Model satisfies all OECD Principle 4 & Tropsha-Golbraikh validation criteria (Q^2_ext = {q2_ext:.3f}, CCC = {ccc:.3f}, r_m^2 avg = {r_m_avg:.3f}, k = {k:.3f})."
    elif q2_ext >= min_warn_q2:
        status = "WARNING"
        f_str = ", ".join(failed_criteria) if failed_criteria else "Borderline predictability"
        diag = f"Moderate QSAR predictivity (Q^2_ext = {q2_ext:.3f}, CCC = {ccc:.3f}). Warning on: {f_str}."
    else:
        status = "FAIL"
        f_str = ", ".join(failed_criteria) if failed_criteria else f"Q^2_ext = {q2_ext:.3f} < 0.50"
        diag = f"Model fails OECD validation standards ({f_str}). Inadequate external predictivity."

    return OECDValidationResult(
        n_samples=n,
        r2=r2,
        r2_pearson=r2_pearson,
        q2_ext=q2_ext,
        q2_f1=q2_f1,
        q2_f2=q2_f2,
        q2_f3=q2_f3,
        ccc=ccc,
        mae=mae,
        rmse=rmse,
        k_slope=k,
        k_prime_slope=k_prime,
        r0_2=r0_2,
        r0_prime_2=r0_prime_2,
        tropsha_r2_diff=diff_r0,
        tropsha_r2_prime_diff=diff_r0_prime,
        r_m_2=r_m_2,
        r_m_prime_2=r_m_prime_2,
        r_m_average=r_m_avg,
        delta_r_m_2=delta_r_m_2,
        r0_diff=r0_diff,
        tropsha_passed=tropsha_passed,
        status=status,
        diagnostic_message=diag
    )
