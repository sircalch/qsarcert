"""
Manuscript Methods snippet generator, summary tables, and BibTeX citations for QSARCert.
"""

from typing import Dict, Any, Optional
import os
import pandas as pd
from qsarcert.core.scoring import QSARValidationReport


def generate_qsar_manuscript_assets(
    report: QSARValidationReport,
    output_dir: str
) -> Dict[str, str]:
    """
    Generates manuscript Methods paragraph, summary CSV/LaTeX tables, and BibTeX citations.

    Parameters
    ----------
    report : QSARValidationReport
    output_dir : str

    Returns
    -------
    paths : dict
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = {}

    # 1. Summary DataFrame
    rows = []
    meta = report.metadata
    om = report.oecd_metrics

    rows.append({"Parameter": "Target Endpoint / Algorithm", "Value": f"{meta.get('endpoint', 'Activity')} ({meta.get('algorithm', 'Machine Learning')})", "Status": "PASS"})
    rows.append({"Parameter": "Evaluated Samples", "Value": f"{om.n_samples} compounds", "Status": "PASS"})
    rows.append({"Parameter": "Coefficient of Determination (R^2)", "Value": f"{om.r2:.3f}", "Status": "PASS" if om.r2 >= 0.60 else "WARNING"})
    rows.append({"Parameter": "Predictive Squared Correlation (Q^2_ext)", "Value": f"{om.q2_ext:.3f}", "Status": "PASS" if om.q2_ext >= 0.60 else "WARNING"})
    rows.append({"Parameter": "Concordance Correlation (CCC)", "Value": f"{om.ccc:.3f}", "Status": "PASS" if om.ccc >= 0.85 else "WARNING"})
    rows.append({"Parameter": "Regression Slope (k / k')", "Value": f"{om.k_slope:.3f} / {om.k_prime_slope:.3f}", "Status": "PASS" if (0.85 <= om.k_slope <= 1.15) else "WARNING"})
    rows.append({"Parameter": "Modified Roy Metric (r_m^2 avg)", "Value": f"{om.r_m_average:.3f} (delta: {om.delta_r_m_2:.3f})", "Status": "PASS" if om.r_m_average >= 0.50 else "WARNING"})

    if report.applicability_domain:
        ad = report.applicability_domain
        rows.append({"Parameter": "Applicability Domain (Williams Plot)", "Value": f"{ad.pct_in_domain:.1f}% in-domain (h* = {ad.warning_leverage:.3f})", "Status": ad.status})

    if report.y_randomization:
        yr = report.y_randomization
        rows.append({"Parameter": "Y-Randomization (cR^2_p)", "Value": f"cR^2_p = {yr.cr2_p:.3f} (Scrambled R^2 = {yr.mean_scrambled_r2:.3f})", "Status": yr.status})

    if report.split_leakage:
        sl = report.split_leakage
        rows.append({"Parameter": "Train/Test Data Leakage Check", "Value": f"{sl.n_exact_duplicates} duplicates (Mean NN sim: {sl.mean_nn_similarity:.3f})", "Status": sl.status})

    df_summary = pd.DataFrame(rows)

    # CSV Table
    csv_path = os.path.join(output_dir, "qsarcert_summary_table.csv")
    df_summary.to_csv(csv_path, index=False)
    generated["summary_csv"] = csv_path

    # LaTeX Table
    tex_path = os.path.join(output_dir, "qsarcert_summary_table.tex")
    tex_content = df_summary.to_latex(index=False, escape=False)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("% QSARCert OECD Principles & Model Validation Table\n")
        f.write(tex_content)
    generated["summary_tex"] = tex_path

    # 2. Methods Text Snippet
    methods_path = os.path.join(output_dir, "methods_snippet.txt")
    ep_str = meta.get("endpoint", "biological activity")
    alg_str = meta.get("algorithm", "machine learning regression")

    ad_str = ""
    if report.applicability_domain:
        ad = report.applicability_domain
        if ad.status == "NOT_APPLICABLE":
            ad_str = (f"Leverage could not bound the applicability domain (h* = {ad.warning_leverage:.2f} >= 1 with"
                      f" {ad.n_features} descriptors and {ad.n_train_samples} training compounds). ")
        else:
            ad_str = (f"The applicability domain was defined by leverage (h* = {ad.warning_leverage:.3f}); {ad.pct_in_domain:.1f}%"
                      f" of the evaluated compounds lie inside it, and {ad.n_influential_outliers} compound(s) outside it"
                      f" have standardised residuals above 3. ")

    yr_str = ""
    if report.y_randomization:
        yr = report.y_randomization
        if yr.status == "NOT_APPLICABLE":
            yr_str = ("Y-randomization with a least-squares surrogate was not applicable because of the number of"
                      " descriptors. ")
        else:
            how = "cross-validated R^2 of the model" if yr.procedure == "estimator" else "training R^2 of a least-squares surrogate"
            yr_str = (f"In Y-randomization ({yr.n_iterations} permutations, {how}) the original R^2 was {yr.original_r2:.3f}"
                      f" against a mean of {yr.mean_scrambled_r2:.3f} for permuted responses (cR^2_p = {yr.cr2_p:.3f};"
                      f" {yr.status}). ")

    tropsha = "met" if om.tropsha_passed else "did not meet"
    full_methods = (
        f"Quantitative Structure-Activity Relationship (QSAR) modeling for {ep_str} was conducted using {alg_str}. "
        f"External predictivity and the OECD validation principles were assessed with QSARCert v{__import__('qsarcert').__version__} (Monreal-Hernández, 2026). "
        f"On the evaluation set, Q^2_F1 = {om.q2_f1:.3f}, Q^2_F2 = {om.q2_f2:.3f}, CCC = {om.ccc:.3f} and r_m^2 (average) = {om.r_m_average:.3f};"
        f" the model {tropsha} the Golbraikh-Tropsha criteria. "
        f"{ad_str}{yr_str}"
        f"Overall QSARCert status: {report.overall_status}."
    )

    with open(methods_path, "w", encoding="utf-8") as f:
        f.write(full_methods + "\n")
    generated["methods_text"] = methods_path

    # 3. BibTeX Citation
    bib_path = os.path.join(output_dir, "citation.bib")
    bib_content = """@software{monreal2026qsarcert,
  author = {Monreal-Hern\\'andez, Andre},
  title = {{QSARCert: An Open-Source Toolkit for OECD Validation Principles, Applicability Domain Assessment, Y-Randomization, and Reproducibility Certification of QSAR and Molecular Machine Learning Models}},
  year = {2026},
  version = {1.0.0},
  publisher = {Zenodo},
  url = {https://github.com/sircalch/qsarcert}
}
"""
    with open(bib_path, "w", encoding="utf-8") as f:
        f.write(bib_content)
    generated["citation_bib"] = bib_path

    return generated

