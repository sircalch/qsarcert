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
    n_scrambling_iterations: int = 100,
    y_train_pred: Optional[np.ndarray] = None,
    estimator: Any = None,
    smiles_train: Optional[List[str]] = None,
    smiles_eval: Optional[List[str]] = None
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
    y_train_pred : np.ndarray, optional
        Training predictions; the training residuals give the residual scale of the Williams plot.
    estimator : object with fit/predict, optional
        The model, refitted on permuted responses for Y-randomization (cross-validated R^2).
        Without it an OLS surrogate is used, which is valid only for few descriptors.
    smiles_train, smiles_eval : list of str, optional
        Structures for the duplicate-structure and scaffold checks (requires RDKit).

    Returns
    -------
    report : QSARValidationReport
    """
    statuses = []
    recommendations = []

    # 1. OECD Principle 4: Goodness-of-fit & External Predictivity
    oecd_res = calculate_oecd_metrics(y_true, y_pred, y_train=y_train)
    statuses.append(oecd_res.status)
    if oecd_res.status != "PASS":
        recommendations.append(oecd_res.diagnostic_message)

    # 2. OECD Principle 3: Applicability Domain (Williams Plot)
    ad_res = None
    if x_train is not None and x_eval is not None:
        train_res = None
        if y_train is not None and y_train_pred is not None:
            train_res = np.asarray(y_train, dtype=float) - np.asarray(y_train_pred, dtype=float)
        ad_res = calculate_applicability_domain(x_train, x_eval, y_true, y_pred, y_train_residuals=train_res)
        if ad_res.status != "NOT_APPLICABLE":
            statuses.append(ad_res.status)
        if ad_res.status != "PASS":
            recommendations.append(ad_res.diagnostic_message)

    # 3. OECD Principle 4: Robustness against chance correlation (Y-randomization)
    y_rand_res = None
    if run_y_scrambling:
        # Y-randomization tests the training data: the responses the model was fitted to.
        if x_train is not None and y_train is not None and len(x_train) == len(y_train):
            x_for_rand, y_for_rand = x_train, y_train
        else:
            x_for_rand, y_for_rand = None, None
            recommendations.append("Y-randomization skipped: it needs the training descriptors and responses.")

        if x_for_rand is not None and y_for_rand is not None:
            y_rand_res = perform_y_randomization(
                x_for_rand,
                y_for_rand,
                n_iterations=n_scrambling_iterations,
                estimator=estimator
            )
            if y_rand_res.status != "NOT_APPLICABLE":
                statuses.append(y_rand_res.status)
            if y_rand_res.status != "PASS":
                recommendations.append(y_rand_res.diagnostic_message)

    # 4. Split Leakage Check
    leakage_res = None
    if x_train is not None and x_eval is not None:
        leakage_res = check_split_leakage(x_train, x_eval, smiles_train=smiles_train, smiles_test=smiles_eval)
        statuses.append(leakage_res.status)
        if leakage_res.status != "PASS":
            recommendations.append(leakage_res.diagnostic_message)

    # Overall Decision
    if "FAIL" in statuses:
        overall_status = "FAIL"
        validation_score = "NUMERICAL CHECKS (OECD PRINCIPLES 3-4) = AT LEAST ONE FAILED"
    elif "WARNING" in statuses:
        overall_status = "WARNING"
        validation_score = "NUMERICAL CHECKS (OECD PRINCIPLES 3-4) = PASSED WITH WARNINGS"
    else:
        overall_status = "PASS"
        validation_score = "NUMERICAL CHECKS (OECD PRINCIPLES 3-4) = ALL PASSED"

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
            "version": __import__("qsarcert").__version__,
            "citation": "Monreal-Hernández, A. (2026). QSARCert: An Open-Source Toolkit for OECD Validation Principles, Applicability Domain Assessment, Y-Randomization, and Reproducibility Certification of QSAR and Molecular Machine Learning Models."
        }
    )
