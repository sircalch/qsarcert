# QSARCert

[![CI](https://github.com/amonreal/qsarcert/actions/workflows/test.yml/badge.svg)](https://github.com/amonreal/qsarcert/actions)
[![PyPI version](https://img.shields.io/pypi/v/qsarcert.svg?color=blue)](https://pypi.org/project/qsarcert/)
[![Python versions](https://img.shields.io/pypi/pyversions/qsarcert.svg)](https://pypi.org/project/qsarcert/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234580.svg)](https://doi.org/10.5281/zenodo.1234580)

> **Automated OECD Validation Principles, Applicability Domain (Williams Plot), Y-Randomization, and Scaffold Leakage Certification for QSAR & Molecular ML Models.**

---

## Overview

**QSARCert** is an open-source scientific software package designed to systematically validate, audit, and certify machine learning (Random Forest, XGBoost, Neural Networks, Graph Neural Networks) and QSAR/QSPR models against the **5 OECD Validation Principles for (Q)SAR Models**.

In computational drug discovery, cheminformatics, and regulatory toxicology (REACH, FDA, OECD), proving external predictivity, absence of chance correlation, and domain bounding is essential:

- 🎯 **Applicability Domain & Williams Plot (OECD Principle 3)**:
  - Exact Hat matrix leverage calculation: $\mathbf{H} = \mathbf{X} (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$.
  - Warning leverage cutoff: $h^* = \frac{3(p + 1)}{n}$.
  - Standardized studentized residuals: $\delta_i = \frac{y_i - \hat{y}_i}{s \sqrt{|1 - h_i|}}$.
  - Automatic detection of influential outliers and response outliers.
- 📐 **Tropsha-Golbraikh & Roy Statistical Metrics (OECD Principle 4)**:
  - Coefficients: $R^2$, $Q^2_{\text{ext}} / R^2_{\text{pred}}$, $Q^2_{F1}, Q^2_{F2}, Q^2_{F3}$, Concordance Correlation Coefficient ($\text{CCC} \ge 0.85$).
  - Regressions through the origin: slopes $k, k' \in [0.85, 1.15]$, $R_0^2, {R'}_0^2$.
  - Modified determination parameters: $r_m^2, {r'}_m^2, \bar{r}_m^2 \ge 0.50$, and $\Delta r_m^2 \le 0.20$.
- 🎲 **Robustness against Chance Correlation: Y-Randomization (OECD Principle 4)**:
  - $N = 100$ response permutations ($Y$-scrambling).
  - Penalization parameter: $cR^2_p = R \sqrt{R^2 - \bar{R}_r^2} > 0.50$.
- 🛡️ **Train/Test Data Leakage & Overlap Audit**:
  - Detects duplicate feature vectors and extreme chemical similarity ($> 95\%$) between train and test splits to prevent over-optimistic performance reports.
- 📑 **Publication Deliverables**:
  - Interactive self-contained `report.html` dashboard.
  - Publication vector plots (Williams Plot, Observed vs Predicted scatter, Y-scrambling histogram) in SVG, PDF, PNG (300 DPI).
  - Ready-to-compile LaTeX summary tables (`.tex`).
  - Draft **Methods** text snippet and BibTeX citation (`citation.bib`).

```
    Predictions CSV / Feature Matrices (Train & Test)
                           │
                           ▼
  ┌───────────────────────────────────────────────────────────┐
  │                         QSARCert                          │
  │  ├── Applicability Domain & Williams Plot (h* = 3(p+1)/n) │
  │  ├── Tropsha-Golbraikh & Roy r_m^2 Statistical Metrics    │
  │  ├── Y-Randomization Scrambling Engine (cR^2_p > 0.50)    │
  │  └── Train/Test Scaffold Data Leakage Detection           │
  └───────────────────────────────────────────────────────────┘
                           │
                           ▼
  ┌───────────────────────────────────────────────────────────┐
  │                   Publication Deliverables                │
  │  ├── report.html (Interactive Dashboard & Badges)         │
  │  ├── qsarcert_williams_plot.pdf/svg/png                   │
  │  ├── qsarcert_observed_vs_predicted.pdf/svg/png           │
  │  ├── qsarcert_y_randomization.pdf/svg/png                 │
  │  ├── qsarcert_summary_table.tex / .csv                    │
  │  ├── methods_snippet.txt (Ready for Manuscript)           │
  │  └── citation.bib (BibTeX Reference)                      │
  └───────────────────────────────────────────────────────────┘
```

---

## Installation

### From PyPI
```bash
pip install qsarcert
```

### From Source
```bash
git clone https://github.com/amonreal/qsarcert.git
cd qsarcert
pip install -e .[dev]
```

---

## Quickstart (CLI)

### 1. Run Benchmark Demo (Random Forest Kinase pIC50 QSAR Audit)
```bash
qsarcert demo -o my_qsar_audit/
```
Open `my_qsar_audit/report.html` in any browser to inspect the interactive report!

### 2. Assess Predictions CSV
```bash
qsarcert assess -i predictions.csv --features-train x_train.csv --features-test x_test.csv -o qsar_report/
```

---

## Python API Usage

```python
from qsarcert import assess_qsar_quality
from qsarcert.parsers import parse_predictions_csv
from qsarcert.reporters import generate_qsar_figures, generate_qsar_manuscript_assets, generate_qsar_html_report

# 1. Parse prediction results
data = parse_predictions_csv("qsar_predictions.csv")

# 2. Assess OECD principles and applicability domain
report = assess_qsar_quality(
    metadata={"endpoint": "pIC50", "algorithm": "Random Forest"},
    y_true=data["y_eval"],
    y_pred=data["y_eval_pred"],
    x_train=data["x_train"],
    x_eval=data["x_eval"],
    run_y_scrambling=True,
    n_scrambling_iterations=100
)

print(f"Overall OECD Certification: {report.overall_status}")
print(f"Predictivity: Q^2_ext = {report.oecd_metrics.q2_ext:.3f}, CCC = {report.oecd_metrics.ccc:.3f}")
print(f"Applicability Domain: {report.applicability_domain.pct_in_domain:.1f}% In-Domain (h* = {report.applicability_domain.warning_leverage:.3f})")
print(f"Y-Randomization: cR^2_p = {report.y_randomization.cr2_p:.3f} > 0.50")

# 3. Export all publication assets
generate_qsar_figures(report, "output_dir/", y_true=data["y_eval"], y_pred=data["y_eval_pred"])
generate_qsar_manuscript_assets(report, "output_dir/")
generate_qsar_html_report(report, "output_dir/report.html")
```

---

## Citation

If you use QSARCert in your research, please cite:

```bibtex
@software{monreal2026qsarcert,
  author = {Monreal-Hern{\'a}ndez, Andre},
  title = {{QSARCert: An Open-Source Toolkit for OECD Validation Principles, Applicability Domain Assessment, Y-Randomization, and Reproducibility Certification of QSAR and Molecular Machine Learning Models}},
  year = {2026},
  version = {1.0.0},
  publisher = {Zenodo},
  url = {https://github.com/amonreal/qsarcert}
}
```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
