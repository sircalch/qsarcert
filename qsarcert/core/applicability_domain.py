"""
Applicability Domain (AD) assessment, Hat matrix leverage, and Williams Plot.

The domain is structural: a compound is inside it when its leverage does not exceed the warning
leverage h* = 3(p + 1)/n (Gramatica 2007). Residuals do not decide membership, since the domain
must be defined for compounds whose response is unknown; they are used only to label response
outliers on the Williams plot.
"""

from typing import List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class CompoundADClassification:
    index: int
    leverage: float
    standardized_residual: float
    category: str  # 'IN_DOMAIN', 'HIGH_LEVERAGE' (outside, accurate), 'RESPONSE_OUTLIER', 'INFLUENTIAL_OUTLIER'
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
    pct_in_domain: float  # percentage with h <= h*
    n_influential_outliers: int
    n_response_outliers: int
    n_high_leverage_accurate: int
    status: str  # 'PASS', 'WARNING', 'FAIL', 'NOT_APPLICABLE'
    diagnostic_message: str
    design_rank: int = 0  # rank of the centred training design matrix (with intercept)
    residual_scale: float = float("nan")  # s used to standardise residuals
    residual_scale_source: str = ""  # 'training' or 'evaluation (robust)'


def _thin_svd(x):
    """Singular values and right singular vectors; falls back to the slower but more robust LAPACK
    driver gesvd when the default divide-and-conquer driver does not converge."""
    try:
        _, sv, vt = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd
        _, sv, vt = svd(x, full_matrices=False, lapack_driver="gesvd")
    return sv, vt


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
    Computes leverages h_i = x_i (X^T X)^+ x_i^T of the evaluated compounds with respect to the
    training design (standardised descriptors plus intercept), the warning leverage h*, standardised
    residuals and the Williams plot categories.

    Parameters
    ----------
    x_train, x_eval : np.ndarray
        Descriptor matrices (n, p) used by the model.
    y_true, y_pred : np.ndarray
        Observed and predicted responses of the evaluated compounds (Williams plot only).
    y_train_residuals : np.ndarray, optional
        Training residuals. Their standard deviation (n - p - 1 degrees of freedom) is the residual
        scale s, as in the Williams plot. Without them a robust scale of the evaluated residuals is
        used and reported as such.
    residual_threshold : float, default 3.0
    min_pass_pct_in_domain, min_warn_pct_in_domain : float
        Coverage (percentage with h <= h*) for PASS and WARNING.

    Notes
    -----
    Leverage bounds the domain only when the design has many more compounds than descriptors.
    When h* >= 1 (3(p + 1) >= n) every training compound is below h* by construction (training
    leverages cannot exceed 1), the threshold no longer describes the training data and the status is
    NOT_APPLICABLE; reduce the descriptors (e.g. to those of the model or to principal components)
    or use a distance-based domain.
    """
    x_tr = np.asarray(x_train, dtype=float)
    x_ev = np.asarray(x_eval, dtype=float)
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    n_tr, p = x_tr.shape
    n_ev = len(y_t)

    tr_mean = np.mean(x_tr, axis=0)
    tr_std = np.std(x_tr, axis=0)
    tr_std[tr_std == 0] = 1.0
    x_tr_aug = np.hstack([np.ones((n_tr, 1)), (x_tr - tr_mean) / tr_std])
    x_ev_aug = np.hstack([np.ones((n_ev, 1)), (x_ev - tr_mean) / tr_std])

    # Leverage from the thin SVD of the design, X = U S V^T: (X^T X)^+ = V S^-2 V^T over the non-zero
    # singular values. Working on X rather than X^T X avoids squaring its condition number.
    sv, vt = _thin_svd(x_tr_aug)
    tol = sv.max() * max(x_tr_aug.shape) * np.finfo(float).eps if sv.size else 0.0
    keep = sv > tol
    rank = int(keep.sum())
    z = (x_ev_aug @ vt[keep].T) / sv[keep]
    leverages = np.sum(z ** 2, axis=1)
    # h* = 3(p + 1)/n, with p + 1 replaced by the rank of the design when descriptors are collinear
    # (the mean training leverage is rank/n, so h* stays three times the mean).
    h_star = float(3.0 * rank / max(1, n_tr))

    residuals = y_t - y_p
    if y_train_residuals is not None and len(y_train_residuals) > p + 1:
        r_tr = np.asarray(y_train_residuals, dtype=float)
        s_ref = float(np.sqrt(np.sum(r_tr ** 2) / (len(r_tr) - p - 1)))
        s_src = "training"
    else:
        s_ref = float(1.4826 * np.median(np.abs(residuals - np.median(residuals))))
        s_src = "evaluation (robust)"
    if not np.isfinite(s_ref) or s_ref < 1e-12:
        s_ref = float(np.std(residuals)) if np.std(residuals) > 1e-12 else 1.0

    # Williams plot convention: residual scaled by s sqrt(1 - h); for extrapolated compounds (h >= 1)
    # the prediction variance exceeds s^2 and the residual is scaled by s alone.
    denom = s_ref * np.sqrt(np.clip(1.0 - leverages, 1e-12, None))
    denom = np.where(leverages < 1.0, denom, s_ref)
    std_residuals = residuals / denom

    classifications = []
    counts = {"IN_DOMAIN": 0, "HIGH_LEVERAGE": 0, "RESPONSE_OUTLIER": 0, "INFLUENTIAL_OUTLIER": 0}
    for idx in range(n_ev):
        h = float(leverages[idx])
        res = float(std_residuals[idx])
        high_lev = h > h_star
        high_res = abs(res) > residual_threshold
        cat = ("INFLUENTIAL_OUTLIER" if high_res else "HIGH_LEVERAGE") if high_lev else \
              ("RESPONSE_OUTLIER" if high_res else "IN_DOMAIN")
        counts[cat] += 1
        classifications.append(CompoundADClassification(index=idx, leverage=h, standardized_residual=res,
                                                        category=cat, is_in_domain=not high_lev))

    n_in = counts["IN_DOMAIN"] + counts["RESPONSE_OUTLIER"]
    pct_in_domain = float(100.0 * n_in / max(1, n_ev))
    max_lev = float(np.max(leverages)) if n_ev else 0.0
    mean_lev = float(np.mean(leverages)) if n_ev else 0.0
    extra = (f" {counts['INFLUENTIAL_OUTLIER']} compound(s) outside the domain are also badly predicted"
             f" (|residual| > {residual_threshold:g} s).") if counts["INFLUENTIAL_OUTLIER"] else ""

    if h_star >= 1.0:
        status = "NOT_APPLICABLE"
        diag = (f"Leverage cannot bound the domain: h* = {h_star:.2f} >= 1 with p = {p} descriptors (design rank {rank})"
                f" and n = {n_tr} training compounds. Use the model's own descriptors, principal components"
                f" or a distance-based domain.")
    elif pct_in_domain >= min_pass_pct_in_domain:
        status = "PASS"
        diag = f"{pct_in_domain:.1f}% of the compounds lie inside the leverage domain (h <= h* = {h_star:.3f})." + extra
    elif pct_in_domain >= min_warn_pct_in_domain:
        status = "WARNING"
        diag = (f"{pct_in_domain:.1f}% of the compounds lie inside the leverage domain (h* = {h_star:.3f});"
                f" predictions for the rest are extrapolations." + extra)
    else:
        status = "FAIL"
        diag = (f"Only {pct_in_domain:.1f}% of the compounds lie inside the leverage domain (h* = {h_star:.3f});"
                f" most predictions are extrapolations." + extra)

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
        n_influential_outliers=counts["INFLUENTIAL_OUTLIER"],
        n_response_outliers=counts["RESPONSE_OUTLIER"],
        n_high_leverage_accurate=counts["HIGH_LEVERAGE"],
        status=status,
        diagnostic_message=diag,
        design_rank=rank,
        residual_scale=s_ref,
        residual_scale_source=s_src,
    )
