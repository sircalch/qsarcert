"""
Quickstart API tutorial for QSARCert.
"""

import os
import sys

# Ensure current script dir is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qsarcert import assess_qsar_quality
from qsarcert.parsers import parse_predictions_csv
from qsarcert.reporters import (
    generate_qsar_figures,
    generate_qsar_manuscript_assets,
    generate_qsar_html_report
)
from generate_sample_qsar_data import generate_sample_qsar_data


def main():
    print("Running QSARCert Python API quickstart tutorial...")
    raw_data_dir = "sample_qsar_dataset"
    generate_sample_qsar_data(raw_data_dir)

    output_dir = "quickstart_qsarcert_output"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Parse prediction CSV
    csv_file = os.path.join(raw_data_dir, "qsar_predictions_with_features.csv")
    data = parse_predictions_csv(csv_file)

    # 2. Assess OECD principles and applicability domain
    report = assess_qsar_quality(
        metadata={"endpoint": "pIC50 (Enzyme Inhibition)", "algorithm": "Random Forest Regressor"},
        y_true=data["y_eval"],
        y_pred=data["y_eval_pred"],
        x_train=data["x_train"],
        x_eval=data["x_eval"],
        run_y_scrambling=True,
        n_scrambling_iterations=50
    )

    print(f"\nOverall OECD Certification: {report.overall_status}")
    print(f"Validation Score: {report.validation_score}")
    print(f"External Predictivity: Q^2_ext = {report.oecd_metrics.q2_ext:.3f}, CCC = {report.oecd_metrics.ccc:.3f}")
    print(f"Applicability Domain: {report.applicability_domain.pct_in_domain:.1f}% In-Domain (h* = {report.applicability_domain.warning_leverage:.3f})")
    print(f"Y-Randomization: cR^2_p = {report.y_randomization.cr2_p:.3f} > 0.50")

    # 3. Export all publication assets
    generate_qsar_figures(report, output_dir, y_true=data["y_eval"], y_pred=data["y_eval_pred"])
    assets = generate_qsar_manuscript_assets(report, output_dir)

    with open(assets["methods_text"], "r", encoding="utf-8") as f:
        methods_txt = f.read()
    with open(assets["citation_bib"], "r", encoding="utf-8") as f:
        bib_txt = f.read()

    html_p = os.path.join(output_dir, "report.html")
    generate_qsar_html_report(report, html_p, methods_text=methods_txt, citation_bib=bib_txt)

    print(f"\nCompleted! HTML report available at: {os.path.abspath(html_p)}")


if __name__ == "__main__":
    main()
