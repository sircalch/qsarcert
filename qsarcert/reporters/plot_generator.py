"""
Publication-ready vector figures for QSAR and molecular machine learning models.
"""

from typing import List, Optional
import os
import numpy as np
import matplotlib.pyplot as plt
from qsarcert.core.scoring import QSARValidationReport

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'lines.linewidth': 2.0,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})


def generate_qsar_figures(
    report: QSARValidationReport,
    output_dir: str,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    formats: List[str] = ("png", "svg", "pdf")
) -> List[str]:
    """
    Generates publication figures: Williams Plot, Observed vs Predicted, and Y-Scrambling Histogram.

    Parameters
    ----------
    report : QSARValidationReport
    output_dir : str
    y_true : np.ndarray, optional
    y_pred : np.ndarray, optional
    formats : list of str

    Returns
    -------
    saved_paths : list of str
    """
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    # 1. Williams Plot (Applicability Domain)
    if report.applicability_domain is not None:
        ad = report.applicability_domain
        h_vals = np.asarray(ad.leverages)
        res_vals = np.asarray(ad.standardized_residuals)
        h_star = ad.warning_leverage

        fig, ax = plt.subplots(figsize=(7, 5.5))

        # Color categories
        colors = []
        for c in ad.classifications:
            if c.category == "IN_DOMAIN":
                colors.append("#0284c7")  # Blue
            elif c.category == "HIGH_LEVERAGE_ACCURATE":
                colors.append("#16a34a")  # Green
            elif c.category == "RESPONSE_OUTLIER":
                colors.append("#f59e0b")  # Yellow/Orange
            else:
                colors.append("#dc2626")  # Red (Influential Outlier)

        ax.scatter(h_vals, res_vals, c=colors, alpha=0.75, edgecolors="none", s=45)

        # Boundary lines
        ax.axvline(h_star, color="#dc2626", linestyle="--", linewidth=1.5, label=rf"Warning Leverage $h^* = {h_star:.3f}$")
        ax.axhline(3.0, color="#f59e0b", linestyle=":", linewidth=1.2, label=r"$\pm 3\sigma$ Residual Boundary")
        ax.axhline(-3.0, color="#f59e0b", linestyle=":", linewidth=1.2)
        ax.axhline(0.0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

        # Shaded In-Domain quadrant
        max_h = max(h_star * 1.5, float(np.max(h_vals) * 1.1))
        ax.fill_between([0, h_star], -3.0, 3.0, color="#0284c7", alpha=0.08, label="Applicability Domain")

        ax.set_xlim(0, max_h)
        ax.set_ylim(min(-4.0, float(np.min(res_vals) - 0.5)), max(4.0, float(np.max(res_vals) + 0.5)))
        ax.set_xlabel("Hat Matrix Leverage ($h_i$)")
        ax.set_ylabel(r"Standardized Residual ($\delta_i$)")
        ax.set_title(f"Williams Plot — Applicability Domain ({ad.pct_in_domain:.1f}% In-Domain)")
        ax.grid(True)
        ax.legend(loc="upper right", frameon=True, fontsize=9)

        plt.tight_layout()
        for fmt in formats:
            p = os.path.join(output_dir, f"qsarcert_williams_plot.{fmt}")
            plt.savefig(p, dpi=300, bbox_inches="tight")
            saved_files.append(p)
        plt.close()

    # 2. Observed vs Predicted Scatter Plot
    if y_true is not None and y_pred is not None:
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)
        om = report.oecd_metrics

        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(y_t, y_p, color="#0284c7", alpha=0.7, edgecolors="white", s=50, label="Compounds")

        # 1:1 line
        min_v = min(np.min(y_t), np.min(y_p))
        max_v = max(np.max(y_t), np.max(y_p))
        pad = 0.05 * (max_v - min_v)
        lims = [min_v - pad, max_v + pad]

        ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.5, label="Ideal 1:1 Diagonal")
        
        # Origin regression line: y_pred = k * y_true (or inverse)
        x_line = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_line, om.k_prime_slope * x_line, color="#dc2626", linestyle=":", linewidth=1.2, label=rf"Regression Slope $k' = {om.k_prime_slope:.3f}$")

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Observed {report.metadata.get('endpoint', 'Activity')}")
        ax.set_ylabel(f"Predicted {report.metadata.get('endpoint', 'Activity')}")
        ax.set_title(rf"Observed vs Predicted — $R^2 = {om.r2:.3f}$, $Q^2_{{\mathrm{{ext}}}} = {om.q2_ext:.3f}$, $\mathrm{{CCC}} = {om.ccc:.3f}$")
        ax.grid(True)
        ax.legend(loc="upper left", frameon=True, fontsize=9)

        plt.tight_layout()
        for fmt in formats:
            p = os.path.join(output_dir, f"qsarcert_observed_vs_predicted.{fmt}")
            plt.savefig(p, dpi=300, bbox_inches="tight")
            saved_files.append(p)
        plt.close()

    # 3. Y-Randomization Scrambled R^2 Distribution Histogram
    if report.y_randomization is not None:
        yr = report.y_randomization
        scrambled_arr = np.asarray(yr.scrambled_r2_distribution)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(scrambled_arr, bins=15, color="#94a3b8", edgecolor="black", alpha=0.7, label=rf"Scrambled $R^2_r$ ($N={yr.n_iterations}$)")
        
        ax.axvline(yr.mean_scrambled_r2, color="#f59e0b", linestyle="-", linewidth=2.0, label=rf"Mean Scrambled $\bar{{R}}^2_r = {yr.mean_scrambled_r2:.3f}$")
        ax.axvline(yr.original_r2, color="#0284c7", linestyle="--", linewidth=2.5, label=rf"Original Model $R^2 = {yr.original_r2:.3f}$")

        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r"Coefficient of Determination ($R^2$)")
        ax.set_ylabel("Frequency")
        ax.set_title(rf"Y-Randomization Test — $cR^2_p = {yr.cr2_p:.3f}$ (Threshold: $>0.50$)")
        ax.grid(True)
        ax.legend(loc="upper right", frameon=True, fontsize=9)

        plt.tight_layout()
        for fmt in formats:
            p = os.path.join(output_dir, f"qsarcert_y_randomization.{fmt}")
            plt.savefig(p, dpi=300, bbox_inches="tight")
            saved_files.append(p)
        plt.close()

    return saved_files
