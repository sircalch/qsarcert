"""
QSARCert: Automated OECD Validation Principles, Applicability Domain (Williams Plot),
Y-Randomization, and Scaffold Leakage Certification for QSAR & Molecular ML Models.
"""

__version__ = "1.0.0"
__author__ = "Andres Monreal-Hernández"
__license__ = "MIT"

from qsarcert.core.applicability_domain import (
    calculate_applicability_domain,
    ApplicabilityDomainResult
)
from qsarcert.core.oecd_metrics import (
    calculate_oecd_metrics,
    OECDValidationResult
)
from qsarcert.core.y_randomization import (
    perform_y_randomization,
    YRandomizationResult
)
from qsarcert.core.scaffold_leakage import (
    check_split_leakage,
    SplitLeakageResult
)
from qsarcert.core.scoring import assess_qsar_quality, QSARValidationReport

__all__ = [
    "__version__",
    "calculate_applicability_domain",
    "ApplicabilityDomainResult",
    "calculate_oecd_metrics",
    "OECDValidationResult",
    "perform_y_randomization",
    "YRandomizationResult",
    "check_split_leakage",
    "SplitLeakageResult",
    "assess_qsar_quality",
    "QSARValidationReport"
]
