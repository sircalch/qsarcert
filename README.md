# QSARCert

[![CI](https://github.com/sircalch/qsarcert/actions/workflows/test.yml/badge.svg)](https://github.com/sircalch/qsarcert/actions)
[![PyPI version](https://img.shields.io/pypi/v/qsarcert.svg?color=blue)](https://pypi.org/project/qsarcert/)
[![Python versions](https://img.shields.io/pypi/pyversions/qsarcert.svg)](https://pypi.org/project/qsarcert/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22217582.svg)](https://doi.org/10.5281/zenodo.22217582)

> **Numerical checks of OECD validation principles 3 and 4 for QSAR and molecular machine-learning regression models: applicability domain, external predictivity, Y-randomization and train/test leakage.**

---

## Overview

**QSARCert** is an open-source package that computes the numerical checks of OECD validation principles 3 (applicability domain) and 4 (goodness-of-fit, robustness, predictivity) for regression QSAR/QSPR and machine-learning models, and reports for each check whether it passes, warns, fails or is not applicable to the model given. Principles 1, 2 and 5 (endpoint, algorithm, mechanistic interpretation) are documentation requirements that no software can verify.

> **Version 1.2.0** was validated on 180 real models (MLR, ridge regression and random forests on ESOL, FreeSolv and Lipophilicity, random and scaffold splits); see `validation/` and the CHANGELOG for the defects this exposed in 1.1.0.

What it computes:

- **Applicability domain (Principle 3)**: leverage $h = x(X^TX)^{+}x^T$ of each evaluated compound on the standardised training design, warning leverage $h^* = 3p'/n$ ($p'$ = rank of the design, i.e. $p+1$ for non-collinear descriptors), Williams plot. Membership is decided by leverage alone; residuals, standardised with the training residual scale, only label response outliers. The check is *not applicable* when $h^* \ge 1$ (e.g. fingerprints with more bits than compounds).
- **External predictivity (Principle 4)**: $Q^2_{F1}$, $Q^2_{F2}$, $Q^2_{F3}$ (Consonni et al. 2009), CCC (Lin 1989), RMSE, MAE; Golbraikh–Tropsha criteria with the squared Pearson correlation and regressions through the origin; Roy's $r_m^2$ and $\Delta r_m^2$.
- **Y-randomization (Principle 4)**: $cR^2_p = R\sqrt{R^2 - \bar R_r^2}$ with 100 permutations. By default an OLS surrogate on the model's descriptors (expected chance $R^2 = p/(n-1)$; *not applicable* when that exceeds 0.5); pass the model as `estimator` to refit it and use 5-fold cross-validated $R^2$.
- **Train/test leakage**: identical input vectors; Tanimoto nearest neighbours for binary fingerprints; duplicate structures and shared Bemis–Murcko scaffolds when SMILES are given.
- 📑 **Publication Deliverables**:
  - Interactive self-contained `report.html` dashboard.
  - Publication vector plots (Williams Plot, Observed vs Predicted scatter, Y-scrambling histogram) in SVG, PDF, PNG (300 DPI).
  - Ready-to-compile LaTeX summary tables (`.tex`).
  - Draft **Methods** paragraph that reports the actual results (including failed checks) and BibTeX citation (`citation.bib`).

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
> **Note:** PyPI release pending. Until then, install from the tagged GitHub release:

```bash
pip install "git+https://github.com/sircalch/qsarcert@v1.2.0"
```

### From Source
```bash
git clone https://github.com/sircalch/qsarcert.git
cd qsarcert
pip install -e .[dev]
```

---

## Quickstart (CLI)

### 1. Run the demo (synthetic data)
```bash
qsarcert demo -o my_qsar_audit/
```
Open `my_qsar_audit/report.html` in any browser to inspect the interactive report!

### 2. Assess Predictions CSV
```bash
qsarcert assess -i predictions.csv -o qsar_report/
```
`predictions.csv` has columns `y_true`, `y_pred`, `split` (`train`/`test`), optionally `smiles`, and one column per descriptor. The training rows are needed for $Q^2_{F1}$, $Q^2_{F3}$, the residual scale and Y-randomization.

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
    y_train=data["y_train"],
    y_train_pred=data["y_train_pred"],
    smiles_train=data["smiles_train"],
    smiles_eval=data["smiles_eval"],
    estimator=None,  # or your fitted scikit-learn model, refitted in Y-randomization
    n_scrambling_iterations=100
)

print(f"Overall status: {report.overall_status}")
print(f"Predictivity: Q^2_F1 = {report.oecd_metrics.q2_f1:.3f}, CCC = {report.oecd_metrics.ccc:.3f}")
print(f"Applicability domain: {report.applicability_domain.status}, {report.applicability_domain.pct_in_domain:.1f}% inside (h* = {report.applicability_domain.warning_leverage:.3f})")
print(f"Y-randomization: {report.y_randomization.status}, cR^2_p = {report.y_randomization.cr2_p:.3f}")

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
  version = {1.2.0},
  publisher = {Zenodo},
  url = {https://github.com/sircalch/qsarcert}
}
```

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

