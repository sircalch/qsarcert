"""
OECD Validation Principles scoring engine, certification rules, and report aggregator.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import numpy as np

from qsarcert.core.applicability_domain import ApplicabilityDomainResult, calculate_applicability_domain
from qsarcert.core.oecd_metrics import OECDValidationResult, calculate_oecd_metrics
from qsarcert.core.y_randomization import YRandomizationResult, perform_y_randomization
from qsarcert.core.scaffold_leakage import SplitLeakageResult, check_split_leakage


@dataclass
class QSARValidationReport:
    overall_status: str  # 'PASS', 'WARNING', 'FAIL'
    validation_score: str
    metadata: Dict[str, Any]
    oecd_metrics: OECDValidationResult
    applicability_domain: Optional[ApplicabilityDomainResult]
    y_randomization: Optional[YRandomizationResult]
    split_leakage: Optional[SplitLeakageResult]
    recommendations: List[str]
    provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def assess_qsar_quality(
    metadata: Dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    x_train: Optional[np.ndarray] = None,
    x_eval: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    run_y_scrambling: bool = True,
    n_scrambling_iterations: int = 100
) -> QSARValidationReport:
    """
    Evaluates QSAR & Molecular ML model against OECD Validation Principles.

    Parameters
    ----------
    metadata : dict
        Endpoint (e.g. pIC50, LogP, Toxicity), algorithm (e.g. Random Forest, XGBoost, GNN), descriptor type.
    y_true : np.ndarray
        Observed experimental values (evaluated/test set).
    y_pred : np.ndarray
        Model predicted values (evaluated/test set).
    x_train : np.ndarray, optional
        Training feature matrix for applicability domain & Y-randomization.
    x_eval : np.ndarray, optional
        Evaluation/test feature matrix for applicability domain.
    y_train : np.ndarray, optional
        Training target vector for Y-randomization.
    run_y_scrambling : bool, default True
    n_scrambling_iterations : int, default 100

    Returns
    -------
    report : QSARValidationReport
    """
    statuses = []
    recommendations = []

    # 1. OECD Principle 4: Goodness-of-fit & External Predictivity
    oecd_res = calculate_oecd_metrics(y_true, y_pred)
    statuses.append(oecd_res.status)
    if oecd_res.status != "PASS":
        recommendations.append(oecd_res.diagnostic_message)

    # 2. OECD Principle 3: Applicability Domain (Williams Plot)
    ad_res = None
    if x_train is not None and x_eval is not None:
        ad_res = calculate_applicability_domain(x_train, x_eval, y_true, y_pred)
        statuses.append(ad_res.status)
        if ad_res.status != "PASS":
            recommendations.append(ad_res.diagnostic_message)

    # 3. OECD Principle 4: Robustness against chance correlation (Y-randomization)
    y_rand_res = None
    if run_y_scrambling:
        # Determine appropriate feature matrix and response vector with matching lengths
        if x_train is not None and y_train is not None and len(x_train) == len(y_train):
            x_for_rand = x_train
            y_for_rand = y_train
        elif x_eval is not None and len(x_eval) == len(y_true):
            x_for_rand = x_eval
            y_for_rand = y_true
        elif x_train is not None and len(x_train) == len(y_true):
            x_for_rand = x_train
            y_for_rand = y_true
        else:
            x_for_rand = None
            y_for_rand = None

        if x_for_rand is not None and y_for_rand is not None:
            y_rand_res = perform_y_randomization(
                x_for_rand,
                y_for_rand,
                original_r2=oecd_res.r2,
                n_iterations=n_scrambling_iterations
            )
            statuses.append(y_rand_res.status)
            if y_rand_res.status != "PASS":
                recommendations.append(y_rand_res.diagnostic_message)

    # 4. Split Leakage Check
    leakage_res = None
    if x_train is not None and x_eval is not None:
        leakage_res = check_split_leakage(x_train, x_eval)
        statuses.append(leakage_res.status)
        if leakage_res.status != "PASS":
            recommendations.append(leakage_res.diagnostic_message)

    # Overall Decision
    if "FAIL" in statuses:
        overall_status = "FAIL"
        validation_score = "QSAR MODEL VALIDATION = FAILED / NON-COMPLIANT"
    elif "WARNING" in statuses:
        overall_status = "WARNING"
        validation_score = "QSAR MODEL VALIDATION = COMPLIANT WITH WARNINGS"
    else:
        overall_status = "PASS"
        validation_score = "QSAR MODEL VALIDATION = FULLY OECD COMPLIANT (HIGH CONFIDENCE)"

    return QSARValidationReport(
        overall_status=overall_status,
        validation_score=validation_score,
        metadata=metadata,
        oecd_metrics=oecd_res,
        applicability_domain=ad_res,
        y_randomization=y_rand_res,
        split_leakage=leakage_res,
        recommendations=recommendations,
        provenance={
            "tool": "QSARCert",
            "version": "1.0.0",
            "citation": "Monreal-Hernández, A. (2026). QSARCert: An Open-Source Toolkit for OECD Validation Principles, Applicability Domain Assessment, Y-Randomization, and Reproducibility Certification of QSAR and Molecular Machine Learning Models."
        }
    )
