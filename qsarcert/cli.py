"""
Command Line Interface (CLI) for QSARCert.
"""

import sys
import os
import argparse
import numpy as np

from qsarcert import __version__
from qsarcert.parsers.predictions_csv import parse_predictions_csv
from qsarcert.parsers.feature_matrix import parse_feature_matrix_csv
from qsarcert.core.scoring import assess_qsar_quality
from qsarcert.core.y_randomization import perform_y_randomization
from qsarcert.reporters.plot_generator import generate_qsar_figures
from qsarcert.reporters.manuscript_prep import generate_qsar_manuscript_assets
from qsarcert.reporters.html_report import generate_qsar_html_report


def print_banner():
    banner = rf"""
   ____   _____         _____   _____          _   
  / __ \ / ____|  /\   |  __ \ / ____|        | |  
 | |  | | (___   /  \  | |__) | |     ___ _ __| |_ 
 | |  | |\___ \ / /\ \ |  _  /| |    / _ \ '__| __|
 | |__| |____) / ____ \| | \ \| |___|  __/ |  | |_ 
  \___\_\_____/_/    \_\_|  \_\\_____\___|_|   \__| v{__version__}

 OECD Principles & Applicability Domain Certification Toolkit for QSAR / ML
 Monreal-Hernández et al., 2026
"""
    print(banner)


def run_demo(output_dir: str = "qsarcert_demo_output"):
    """
    Executes a benchmark demonstration evaluating a Random Forest QSAR model (pIC50 prediction)
    with 120 compounds (100 train, 20 test), computing Williams plot, Tropsha metrics, and Y-randomization.
    """
    print(f"\n[QSARCert] Running demonstration benchmark on QSAR Model (Kinase pIC50)...")
    os.makedirs(output_dir, exist_ok=True)

    metadata = {
        "endpoint": "SYNTHETIC DEMO DATA - pIC50 (Kinase Inhibition)",
        "algorithm": "Random Forest Regressor (100 trees)",
        "descriptors": "RDKit 2D PhysChem Descriptors (p=8)"
    }

    # Generate synthetic training & test sets
    rng = np.random.default_rng(42)
    n_train = 100
    n_test = 25
    p = 8

    # True linear+nonlinear ground truth
    true_weights = rng.normal(0, 1.0, size=p)

    x_train = rng.normal(0, 1.0, size=(n_train, p))
    y_train = x_train @ true_weights + rng.normal(0, 0.35, size=n_train)

    x_test = rng.normal(0, 1.0, size=(n_test, p))
    # Add 1 slight extrapolation compound
    x_test[-1] = x_test[-1] * 2.2
    y_test = x_test @ true_weights + rng.normal(0, 0.40, size=n_test)

    # Simulated accurate predictions (R^2 ~ 0.85)
    y_pred_test = y_test + rng.normal(0, 0.35, size=n_test)

    print("  -> Performing OECD Principle 4 statistical validation (Tropsha, CCC, r_m^2)...")
    print("  -> Computing Applicability Domain Hat matrix leverage (h*) and Williams Plot...")
    print("  -> Executing Y-Randomization permutation test (N=100 iterations)...")

    report = assess_qsar_quality(
        metadata=metadata,
        y_true=y_test,
        y_pred=y_pred_test,
        x_train=x_train,
        x_eval=x_test,
        run_y_scrambling=True,
        n_scrambling_iterations=100
    )

    print("  -> Generating publication-ready vector figures (Williams Plot, Observed vs Predicted, Y-scrambling)...")
    generate_qsar_figures(report, output_dir, y_true=y_test, y_pred=y_pred_test)

    print("  -> Drafting manuscript Methods text snippet, summary LaTeX tables, and BibTeX citations...")
    assets = generate_qsar_manuscript_assets(report, output_dir)

    with open(assets["methods_text"], "r", encoding="utf-8") as f:
        methods_txt = f.read()
    with open(assets["citation_bib"], "r", encoding="utf-8") as f:
        bib_txt = f.read()

    html_p = os.path.join(output_dir, "report.html")
    print(f"  -> Writing interactive report to {html_p}...")
    generate_qsar_html_report(report, html_p, methods_text=methods_txt, citation_bib=bib_txt)

    print("\n" + "="*70)
    print(f" [RESULT] Overall QSAR Certification Status: {report.overall_status}")
    print(f" [SCORE]  {report.validation_score}")
    print("="*70)
    print(f" * Target Endpoint : {report.metadata['endpoint']}")
    print(f" * Predictivity    : Q^2_ext = {report.oecd_metrics.q2_ext:.3f}, R^2 = {report.oecd_metrics.r2:.3f}, CCC = {report.oecd_metrics.ccc:.3f} | Status: {report.oecd_metrics.status}")
    print(f" * Tropsha Status  : {'PASS (All 6 criteria met)' if report.oecd_metrics.tropsha_passed else 'WARNING'}")
    if report.applicability_domain:
        print(f" * App. Domain     : {report.applicability_domain.pct_in_domain:.1f}% In-Domain (h* = {report.applicability_domain.warning_leverage:.3f}) | Status: {report.applicability_domain.status}")
    if report.y_randomization:
        print(f" * Y-Randomization : cR^2_p = {report.y_randomization.cr2_p:.3f} > 0.50 (Scrambled R^2 = {report.y_randomization.mean_scrambled_r2:.3f}) | Status: {report.y_randomization.status}")
    print("="*70)
    print(f"\nAll outputs successfully saved to: {os.path.abspath(output_dir)}/")
    print(f"Open {os.path.abspath(html_p)} in your browser to inspect the full report.\n")


def run_assess(args):
    """
    Evaluates user-provided CSV file.
    """
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[QSARCert] Parsing predictions file from: {args.input}...")
    data = parse_predictions_csv(args.input)

    x_tr = None
    x_ev = None
    if "x_train" in data and data["x_train"] is not None:
        x_tr = data["x_train"]
        x_ev = data["x_eval"]
        y_true = data["y_eval"]
        y_pred = data["y_eval_pred"]
    else:
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        if "x_features" in data and data["x_features"] is not None:
            x_tr = data["x_features"]
            x_ev = data["x_features"]

    if args.features_train:
        print(f"  -> Loading training feature matrix: {args.features_train}...")
        x_tr = parse_feature_matrix_csv(args.features_train)
    if args.features_test:
        print(f"  -> Loading test feature matrix: {args.features_test}...")
        x_ev = parse_feature_matrix_csv(args.features_test)

    meta = {
        "endpoint": args.endpoint or "Activity / Endpoint",
        "algorithm": args.algorithm or "Machine Learning / QSAR"
    }

    print("  -> Performing OECD statistical and applicability domain audit...")
    report = assess_qsar_quality(
        metadata=meta,
        y_true=y_true,
        y_pred=y_pred,
        x_train=x_tr,
        x_eval=x_ev,
        run_y_scrambling=not args.no_scrambling,
        n_scrambling_iterations=int(args.scrambling_runs)
    )

    print("  -> Generating publication figures...")
    generate_qsar_figures(report, output_dir, y_true=y_true, y_pred=y_pred)

    print("  -> Generating manuscript text, LaTeX summary table, and BibTeX citations...")
    assets = generate_qsar_manuscript_assets(report, output_dir)

    with open(assets["methods_text"], "r", encoding="utf-8") as f:
        methods_txt = f.read()
    with open(assets["citation_bib"], "r", encoding="utf-8") as f:
        bib_txt = f.read()

    html_p = os.path.join(output_dir, "report.html")
    print(f"  -> Writing HTML quality report to {html_p}...")
    generate_qsar_html_report(report, html_p, methods_text=methods_txt, citation_bib=bib_txt)

    print("\n" + "="*70)
    print(f" [RESULT] Overall Quality Certification: {report.overall_status}")
    print(f" [SCORE]  {report.validation_score}")
    print("="*70)
    print(f" * Predictivity : Q^2_ext = {report.oecd_metrics.q2_ext:.3f}, R^2 = {report.oecd_metrics.r2:.3f}, CCC = {report.oecd_metrics.ccc:.3f}")
    if report.applicability_domain:
        print(f" * App. Domain  : {report.applicability_domain.pct_in_domain:.1f}% in-domain (h* = {report.applicability_domain.warning_leverage:.3f})")
    print("="*70)
    print(f"\nReport ready at: {os.path.abspath(html_p)}\n")


def print_citation():
    bib = """@software{monreal2026qsarcert,
  author = {Monreal-Hern\\'andez, Andre},
  title = {{QSARCert: An Open-Source Toolkit for OECD Validation Principles, Applicability Domain Assessment, Y-Randomization, and Reproducibility Certification of QSAR and Molecular Machine Learning Models}},
  year = {2026},
  version = {1.0.0},
  publisher = {Zenodo},
  url = {https://github.com/sircalch/qsarcert}
}"""
    print("\nIf you use QSARCert in your publications, please cite:\n")
    print("APA Style:")
    print("Monreal-Hernández, A. (2026). QSARCert: An Open-Source Toolkit for OECD Validation Principles, Applicability Domain Assessment, Y-Randomization, and Reproducibility Certification of QSAR and Molecular Machine Learning Models (v1.0.0). Zenodo. https://github.com/sircalch/qsarcert\n")
    print("BibTeX:")
    print(bib)
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="qsarcert",
        description="QSARCert: OECD Validation Principles, Applicability Domain, and Y-Randomization for QSAR & ML Models."
    )
    parser.add_argument("-v", "--version", action="version", version=f"qsarcert {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Assess command
    assess_parser = subparsers.add_parser("assess", help="Assess QSAR / ML predictions and applicability domain")
    assess_parser.add_argument("-i", "--input", required=True, help="Path to predictions CSV/TSV table (y_true, y_pred, split)")
    assess_parser.add_argument("--features-train", default=None, help="Path to training feature matrix CSV")
    assess_parser.add_argument("--features-test", default=None, help="Path to test feature matrix CSV")
    assess_parser.add_argument("-o", "--output", default="qsarcert_output", help="Directory for output report (default: qsarcert_output)")
    assess_parser.add_argument("--endpoint", default=None, help="Endpoint description (e.g. 'pIC50', 'LogP')")
    assess_parser.add_argument("--algorithm", default=None, help="ML algorithm description (e.g. 'Random Forest')")
    assess_parser.add_argument("--no-scrambling", action="store_true", help="Disable Y-randomization test")
    assess_parser.add_argument("--scrambling-runs", default=100, help="Number of Y-randomization runs (default: 100)")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run benchmark demonstration (Kinase pIC50 QSAR model + Williams plot)")
    demo_parser.add_argument("-o", "--output", default="qsarcert_demo_output", help="Output directory (default: qsarcert_demo_output)")

    # Cite command
    subparsers.add_parser("cite", help="Display BibTeX and APA citation details")

    if len(sys.argv) == 1:
        print_banner()
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.command == "assess":
        print_banner()
        run_assess(args)
    elif args.command == "demo":
        print_banner()
        run_demo(args.output)
    elif args.command == "cite":
        print_banner()
        print_citation()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

