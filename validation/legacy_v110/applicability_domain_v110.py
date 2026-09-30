"""
Applicability Domain (AD) assessment, Hat matrix leverage, and Williams Plot.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class CompoundADClassification:
    index: int
    leverage: float
    standardized_residual: float
    category: str  # 'IN_DOMAIN', 'HIGH_LEVERAGE_ACCURATE', 'RESPONSE_OUTLIER', 'INFLUENTIAL_OUTLIER'
    is_in_domain: bool


@dataclass
class ApplicabilityDomainResult:
    n_train_samples: int
    n_features: int
    n_evaluated_samples: int
    warning_leverage: float  # h* = 3(p+1)/n
    max_leverage: float
    mean_leverage: float
    leverages: List[float]
    standardized_residuals: List[float]
    classifications: List[CompoundADClassification]
    pct_in_domain: float
    n_influential_outliers: int
    n_response_outliers: int
    n_high_leverage_accurate: int
    status: str  # 'PASS', 'WARNING', 'FAIL'
    diagnostic_message: str


def calculate_applicability_domain(
    x_train: np.ndarray,
    x_eval: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train_residuals: Optional[np.ndarray] = None,
    residual_threshold: float = 3.0,
    min_pass_pct_in_domain: float = 85.0,
    min_warn_pct_in_domain: float = 70.0
) -> ApplicabilityDomainResult:
    """
    Computes Hat matrix leverage h_i, warning threshold h*, standardized residuals,
    and Williams Plot domain classifications.

    Parameters
    ----------
    x_train : np.ndarray
        Training feature matrix (n_train, p_features).
    x_eval : np.ndarray
        Evaluated / test feature matrix (n_eval, p_features).
    y_true : np.ndarray
        True observed response values for evaluated set.
    y_pred : np.ndarray
        Model predicted response values for evaluated set.
    y_train_residuals : np.ndarray, optional
        Residuals from training set (y_train - y_train_pred) to calculate reference s.
    residual_threshold : float, default 3.0
        Standardized residual cutoff (+-3 sigma).
    min_pass_pct_in_domain : float, default 85.0%
    min_warn_pct_in_domain : float, default 70.0%

    Returns
    -------
    result : ApplicabilityDomainResult
    """
    x_tr = np.asarray(x_train, dtype=float)
    x_ev = np.asarray(x_eval, dtype=float)
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    n_tr, p = x_tr.shape
    n_ev = len(y_t)

    # Standardize / center features based on training set statistics
    tr_mean = np.mean(x_tr, axis=0)
    tr_std = np.std(x_tr, axis=0)
    tr_std[tr_std == 0] = 1.0  # Avoid division by zero

    x_tr_norm = (x_tr - tr_mean) / tr_std
    x_ev_norm = (x_ev - tr_mean) / tr_std

    # Add intercept column (ones)
    x_tr_aug = np.hstack([np.ones((n_tr, 1)), x_tr_norm])
    x_ev_aug = np.hstack([np.ones((n_ev, 1)), x_ev_norm])

    # Warning leverage: h* = 3(p + 1)/n
    p_eff = p
    h_star = float(3.0 * (p_eff + 1) / max(1, n_tr))

    # Hat matrix kernel: (X_tr^T X_tr)^{-1}
    xtx = x_tr_aug.T @ x_tr_aug
    try:
        inv_xtx = np.linalg.pinv(xtx)
    except Exception:
        inv_xtx = np.linalg.inv(xtx + 1e-4 * np.eye(xtx.shape[0]))

    # Compute leverages for evaluated set: h_i = diag(X_ev (X_tr^T X_tr)^{-1} X_ev^T)
    leverages = np.einsum('ij,jk,ik->i', x_ev_aug, inv_xtx, x_ev_aug)
    leverages = np.clip(leverages, 0.0, 10.0)

    # Raw residuals
    residuals = y_t - y_p

    # Reference standard deviation s
    if y_train_residuals is not None and len(y_train_residuals) > 0:
        s_ref = float(np.std(y_train_residuals, ddof=1))
    else:
        median_res = np.median(np.abs(residuals))
        s_ref = float(1.4826 * median_res) if median_res > 1e-4 else float(np.std(residuals, ddof=1))
        
    if s_ref < 1e-6:
        s_ref = float(np.std(residuals)) if np.std(residuals) > 1e-6 else 1.0

    # Standardized studentized residuals
    std_residuals = []
    for idx in range(n_ev):
        e_i = residuals[idx]
        h_i = leverages[idx]
        if h_i < 0.99:
            denom_i = s_ref * np.sqrt(max(1e-4, 1.0 - h_i))
        else:
            denom_i = s_ref
        std_residuals.append(float(e_i / denom_i))

    std_residuals = np.asarray(std_residuals)

    classifications = []
    n_in_domain = 0
    n_high_lev_acc = 0
    n_resp_outliers = 0
    n_influential = 0

    for idx in range(n_ev):
        h = float(leverages[idx])
        res = float(std_residuals[idx])

        is_high_lev = (h > h_star)
        is_high_res = (abs(res) > residual_threshold)

        if not is_high_lev and not is_high_res:
            cat = "IN_DOMAIN"
            in_dom = True
            n_in_domain += 1
        elif is_high_lev and not is_high_res:
            cat = "HIGH_LEVERAGE_ACCURATE"
            in_dom = True  # Extrapolation with accurate prediction
            n_high_lev_acc += 1
        elif not is_high_lev and is_high_res:
            cat = "RESPONSE_OUTLIER"
            in_dom = False
            n_resp_outliers += 1
        else:
            cat = "INFLUENTIAL_OUTLIER"
            in_dom = False
            n_influential += 1

        classifications.append(CompoundADClassification(
            index=idx,
            leverage=h,
            standardized_residual=res,
            category=cat,
            is_in_domain=in_dom
        ))

    pct_in_domain = float(100.0 * (n_in_domain + n_high_lev_acc) / max(1, n_ev))
    max_lev = float(np.max(leverages)) if len(leverages) > 0 else 0.0
    mean_lev = float(np.mean(leverages)) if len(leverages) > 0 else 0.0

    if n_influential > 0:
        status = "FAIL"
        diag = f"Applicability Domain violation: {n_influential} influential outlier(s) detected (h > {h_star:.3f}, |residual| > 3.0). Predictions outside domain are inaccurate."
    elif pct_in_domain >= min_pass_pct_in_domain:
        status = "PASS"
        diag = f"High Applicability Domain coverage ({pct_in_domain:.1f}% in domain >= {min_pass_pct_in_domain:.1f}%). No influential outliers detected (h* = {h_star:.3f})."
    elif pct_in_domain >= min_warn_pct_in_domain:
        status = "WARNING"
        diag = f"Moderate Applicability Domain coverage ({pct_in_domain:.1f}% in domain). Detected {n_resp_outliers} response outlier(s)."
    else:
        status = "FAIL"
        diag = f"Poor Applicability Domain coverage ({pct_in_domain:.1f}% < {min_warn_pct_in_domain:.1f}%)."

    return ApplicabilityDomainResult(
        n_train_samples=n_tr,
        n_features=p,
        n_evaluated_samples=n_ev,
        warning_leverage=h_star,
        max_leverage=max_lev,
        mean_leverage=mean_lev,
        leverages=leverages.tolist(),
        standardized_residuals=std_residuals.tolist(),
        classifications=classifications,
        pct_in_domain=pct_in_domain,
        n_influential_outliers=n_influential,
        n_response_outliers=n_resp_outliers,
        n_high_leverage_accurate=n_high_lev_acc,
        status=status,
        diagnostic_message=diag
    )
