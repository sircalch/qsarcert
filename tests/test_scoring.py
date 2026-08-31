"""
Tests for scoring, manuscript reporting, and CLI execution in QSARCert.
"""

import os
import tempfile
import numpy as np
import pytest
from qsarcert.core.scoring import assess_qsar_quality
from qsarcert.reporters.plot_generator import generate_qsar_figures
from qsarcert.reporters.manuscript_prep import generate_qsar_manuscript_assets
from qsarcert.reporters.html_report import generate_qsar_html_report
from qsarcert.cli import run_demo


def test_full_qsar_validation_pipeline():
    meta = {
        "endpoint": "LogP",
        "algorithm": "XGBoost"
    }

    rng = np.random.default_rng(42)
    n_tr = 50
    n_ev = 15
    p = 6

    x_tr = rng.normal(0, 1, size=(n_tr, p))
    x_ev = rng.normal(0, 1, size=(n_ev, p))

    weights = np.array([2.0, -1.0, 0.5, 1.2, -0.8, 0.3])
    y_tr = x_tr @ weights
    y_ev = x_ev @ weights
    y_pred = y_ev + rng.normal(0, 0.1, size=n_ev)

    report = assess_qsar_quality(
        metadata=meta,
        y_true=y_ev,
        y_pred=y_pred,
        x_train=x_tr,
        x_eval=x_ev,
        run_y_scrambling=True,
        n_scrambling_iterations=20
    )

    assert report.overall_status in ["PASS", "WARNING"]
    assert report.oecd_metrics.status == "PASS"

    with tempfile.TemporaryDirectory() as tmpdir:
        # Vector figures
        plots = generate_qsar_figures(report, tmpdir, y_true=y_ev, y_pred=y_pred, formats=["png", "svg"])
        assert len(plots) > 0
        for p_file in plots:
            assert os.path.exists(p_file)

        # Manuscript assets
        assets = generate_qsar_manuscript_assets(report, tmpdir)
        assert os.path.exists(assets["summary_csv"])
        assert os.path.exists(assets["summary_tex"])
        assert os.path.exists(assets["methods_text"])
        assert os.path.exists(assets["citation_bib"])

        # HTML report
        html_p = os.path.join(tmpdir, "report.html")
        generate_qsar_html_report(report, html_p, methods_text="Sample methods", citation_bib="@software{}")
        assert os.path.exists(html_p)
        assert os.path.getsize(html_p) > 500


def test_cli_demo_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_demo(output_dir=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, "report.html"))
        assert os.path.exists(os.path.join(tmpdir, "qsarcert_summary_table.csv"))
        assert os.path.exists(os.path.join(tmpdir, "citation.bib"))
