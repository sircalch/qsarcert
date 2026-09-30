"""
Figures and tables for the QSARCert v2 manuscript, built only from validation/results/.

    python validation/make_figures.py

Style: Elsevier double-column width 190 mm, 8 pt sans-serif, one fixed colour and marker per series.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from qsarcert.core import calculate_oecd_metrics  # noqa: E402

RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
TAB = os.path.join(HERE, "tables")
MM = 1 / 25.4
DOUBLE = 190 * MM
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NEW, OLD, REF, AUX = "#2a78d6", "#e87ba4", "#1baf7a", "#eda100"
DS = {"esol": "ESOL", "freesolv": "FreeSolv", "lipo": "Lipophilicity"}
MODEL = {"mlr": "MLR (6 descriptors)", "ridge": "ridge (all descriptors)", "rf": "random forest (fingerprints)"}
MARK = {"esol": "o", "freesolv": "s", "lipo": "^"}


def setup():
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 6.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.2, "lines.markersize": 4,
        "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42})


def panel(ax, letter, x=-0.16):
    ax.text(x, 1.03, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", color=INK)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def tropsha_pair(y, p, y_mean_tr):
    """Golbraikh-Tropsha verdicts with centred (QSARCert) and uncentred R0^2."""
    m = calculate_oecd_metrics(y, p, y_train_mean=y_mean_tr)
    r2 = pearsonr(y, p)[0] ** 2
    r0u = 1 - np.sum((y - m.k_slope * p) ** 2) / np.sum(y ** 2)
    r0pu = 1 - np.sum((p - m.k_prime_slope * y) ** 2) / np.sum(p ** 2)
    c3 = (0.85 <= m.k_slope <= 1.15) or (0.85 <= m.k_prime_slope <= 1.15)
    c4 = ((abs(r2 - r0u) / r2 < 0.1) or (abs(r2 - r0pu) / r2 < 0.1)) and abs(r0u - r0pu) < 0.3
    rm = r2 * (1 - np.sqrt(abs(r2 - r0u))), r2 * (1 - np.sqrt(abs(r2 - r0pu)))
    unc = bool(m.q2_ext > 0.5 and r2 > 0.6 and c3 and c4 and abs(rm[0] - rm[1]) < 0.2 and np.mean(rm) > 0.5
               and m.ccc >= 0.8)
    return m.tropsha_passed, unc


def offset_scan(shifts):
    """Pass rate of the two conventions when the response of every model is shifted by a constant
    (in units of the dataset's standard deviation). Every other validation statistic is unchanged."""
    out = []
    for f in sorted(glob.glob(os.path.join(RES, "runs", "*.npz"))):
        ds = os.path.basename(f).split("_")[0]
        d = np.load(f)
        y = d["y"]
        sd = y.std()
        yte, pte, ytr = y[d["test"]], d["pred_test"], y[d["train"]]
        base = yte.mean()
        for s in shifts:
            c = -base + s * sd  # shift so that the test mean sits s standard deviations from zero
            a, b = tropsha_pair(yte + c, pte + c, ytr.mean() + c)
            out.append({"run": os.path.basename(f), "dataset": ds, "shift_sd": s, "centred": a, "uncentred": b})
    return pd.DataFrame(out)


def fig_metrics(d, scan):
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 68 * MM), gridspec_kw={"width_ratios": [1, 1.1]})
    g = scan.groupby("shift_sd")[["centred", "uncentred"]].mean()
    a.plot(g.index, g.centred, color=NEW, marker="o", ms=3, label="centred $R_0^2$ (Golbraikh & Tropsha)")
    a.plot(g.index, g.uncentred, color=OLD, marker="^", ms=3, label="uncentred $R_0^2$")
    a.axvline(0, color=INK2, lw=0.5, ls=":")
    a.set(xlabel="test-set mean after shifting the response (dataset SD from zero)",
          ylabel="models passing all criteria", ylim=(-0.02, 0.62))
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a.legend(loc="lower right")
    panel(a, "a")

    for m, mk in (("mlr", "o"), ("ridge", "s"), ("rf", "^")):
        for sp, col in (("random", NEW), ("scaffold", AUX)):
            v = d[(d.model == m) & (d.split == sp)]
            b.scatter(v.q2_f1, v.q2_f3, s=10, marker=mk, color=col, lw=0, alpha=0.85,
                      label=f"{'MLR' if m == 'mlr' else ('RF' if m == 'rf' else 'ridge')}, {sp}")
    b.plot([-0.5, 1], [-0.5, 1], color=INK2, lw=0.6, ls="--")
    b.set(xlim=(-0.25, 1.0), ylim=(-1.5, 1.0), xlabel="$Q^2_{F1}$ (training mean)", ylabel="$Q^2_{F3}$ (training variance)")
    b.legend(loc="lower right", ncol=2, handletextpad=0.2, columnspacing=0.8)
    panel(b, "b")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig2_metrics")


def fig_ad(d):
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 68 * MM), gridspec_kw={"width_ratios": [1.15, 1]})
    e = d[d.model != "rf"].copy()
    e["ratio"] = e.ad_mae_out / e.ad_mae_in
    groups = [(m, s) for m in ("mlr", "ridge") for s in ("random", "scaffold")]
    rng = np.random.default_rng(0)
    for i, (m, s) in enumerate(groups):
        for j, ds in enumerate(DS):
            v = e[(e.model == m) & (e.split == s) & (e.dataset == ds)].ratio.dropna()
            a.scatter(i + (j - 1) * 0.22 + rng.uniform(-0.05, 0.05, len(v)), v, s=9, marker=MARK[ds],
                      color=[NEW, AUX, REF][j], lw=0, alpha=0.85, label=DS[ds] if i == 0 else None)
    a.axhline(1, color=INK2, lw=0.7, ls="--")
    a.set_yscale("log")
    a.set_yticks([0.5, 1, 2, 4], ["0.5", "1", "2", "4"])
    a.minorticks_off()
    a.set_xticks(range(4), [f"{m if m == 'ridge' else 'MLR'}\n{s}" for m, s in groups])
    a.set_ylabel("MAE outside / inside the leverage domain")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    a.set_ylim(0.5, 5)
    panel(a, "a")

    for m, c, mk in (("mlr", NEW, "o"), ("ridge", AUX, "s")):
        v = d[d.model == m]
        b.scatter(v.ad110_pct_in, v.ad_pct_in, s=10, marker=mk, color=c, lw=0, alpha=0.85, label=MODEL[m])
    b.plot([0, 100], [0, 100], color=INK2, lw=0.6, ls="--")
    b.set(xlim=(0, 102), ylim=(0, 102), xlabel="compounds in domain, version 1.1.0 (%)",
          ylabel="compounds with $h \\leq h^*$, version 1.2 (%)")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    panel(b, "b")
    fig.tight_layout(w_pad=2.5)
    save(fig, "fig3_ad")


def fig_yrand_leak(d, est):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(DOUBLE, 64 * MM), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for m, col, mk in (("mlr", NEW, "o"), ("ridge", AUX, "s"), ("rf", OLD, "^")):
        v = d[d.model == m]
        a.scatter(v.yr_expected, v.yr110_mean, s=10, marker=mk, color=col, lw=0, alpha=0.85, label=MODEL[m].split(" (")[0])
    a.plot([1e-3, 1], [1e-3, 1], color=INK2, lw=0.6, ls="--")
    a.axhline(0.5, color=INK2, lw=0.5, ls=":")
    a.set(xscale="log", yscale="log", xlim=(1e-3, 1.2), ylim=(1e-3, 1.2), xlabel="expected chance $R^2 = p/(n-1)$",
          ylabel="mean $R^2$ of permuted responses")
    a.legend(loc="upper left")
    panel(a, "a", -0.2)

    if est is not None and len(est):
        for j, ds in enumerate(("esol", "freesolv")):
            v = est[est.dataset == ds]
            for k, (_, r) in enumerate(v.iterrows()):
                xx = j * 2 + (0 if r.split == "random" else 1)
                b.scatter(xx - 0.08, r.cv_r2, s=12, marker="o", color=NEW, lw=0, label="original (CV $R^2$)" if (j + k) == 0 else None)
                b.scatter(xx + 0.08, r.mean_scrambled, s=12, marker="v", color=INK2, lw=0,
                          label="permuted, mean" if (j + k) == 0 else None)
        b.set_xticks(range(4), ["ESOL\nrandom", "ESOL\nscaffold", "FreeSolv\nrandom", "FreeSolv\nscaffold"])
        b.axhline(0, color=INK2, lw=0.5)
        b.set(ylabel="5-fold CV $R^2$ of the random forest", ylim=(-0.3, 1.0))
        b.legend(loc="center right")
    panel(b, "b", -0.2)

    rows = []
    for s in ("random", "scaffold"):
        v = d[d.split == s]
        rows.append((s, v[v.model != "rf"].lk110_exact.sum() / v[v.model != "rf"].n_test.sum(),
                     v[v.model != "rf"].lk_exact.sum() / v[v.model != "rf"].n_test.sum(),
                     v[v.model == "mlr"].ref_dup_smiles.sum() / v[v.model == "mlr"].n_test.sum()))
    x = np.arange(2)
    w = 0.26
    c.bar(x - w, [r[1] for r in rows], w, color=OLD, label="1.1.0: 'duplicates' (raw cosine)", edgecolor="white", lw=0.8)
    c.bar(x, [r[2] for r in rows], w, color=NEW, label="1.2: identical inputs", edgecolor="white", lw=0.8)
    c.bar(x + w, [r[3] for r in rows], w, color=REF, label="duplicate structures (SMILES)", edgecolor="white", lw=0.8)
    c.set_yscale("symlog", linthresh=0.01)
    c.set_yticks([0, 0.01, 0.1, 1], ["0", "1%", "10%", "100%"])
    c.minorticks_off()
    c.set_xticks(x, ["random split", "scaffold split"])
    c.set_ylabel("test compounds flagged (descriptor models)")
    c.legend(loc="upper right")
    panel(c, "c", -0.2)
    fig.tight_layout(w_pad=1.8)
    save(fig, "fig4_yrand_leakage")


def tables(d, est, scan):
    # Table: status counts 1.1.0 vs 1.2 per check and model
    def counts(col, m):
        v = d[d.model == m][col].fillna("error").value_counts()
        return ", ".join(f"{k.replace('_', ' ').lower()} {v[k]}" for k in ("PASS", "WARNING", "FAIL", "NOT_APPLICABLE", "error") if k in v)
    lines = [r"\begin{tabular}{llll}", r"\toprule", r"Check & model & version 1.1.0 & version 1.2 \\", r"\midrule"]
    for chk, old, new in (("Applicability domain", "ad110_status", "ad_status"), ("Y-randomization", "yr110_status", "yr_status"),
                          ("Train/test leakage", "lk110_status", "lk_status")):
        for i, m in enumerate(("mlr", "ridge", "rf")):
            lines.append(f"{chk if i == 0 else ''} & {MODEL[m]} & {counts(old, m)} & {counts(new, m)} " + r"\\")
        lines.append(r"\addlinespace")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    with open(os.path.join(TAB, "table_status.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    # Table S: per dataset, split and model medians
    s = d.groupby(["dataset", "split", "model"]).agg(
        q2f1=("q2_f1", "median"), q2f2=("q2_f2", "median"), q2f3=("q2_f3", "median"), ccc=("ccc", "median"),
        rmse=("rmse", "median"), trop=("tropsha", "sum"), tropu=("tropsha_uncentred", "sum"),
        ad=("ad_pct_in", "median"), ratio=("ad_mae_out", "median")).reset_index()
    lines = [r"\begin{tabular}{lllrrrrrrr}", r"\toprule",
             r"Dataset & split & model & $Q^2_{F1}$ & $Q^2_{F2}$ & $Q^2_{F3}$ & CCC & RMSE & G--T pass (c/u) & in domain (\%) \\",
             r"\midrule"]
    for r in s.itertuples():
        ad = "n/a" if r.model == "rf" else f"{r.ad:.1f}"
        lines.append(f"{DS[r.dataset]} & {r.split} & {r.model if r.model != 'mlr' else 'MLR'} & {r.q2f1:.3f} & {r.q2f2:.3f} & {r.q2f3:.3f} & "
                     f"{r.ccc:.3f} & {r.rmse:.3f} & {int(r.trop)}/{int(r.tropu)} & {ad} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "table_s_models.tex"), "w") as fh:
        fh.write("\n".join(lines).replace("rf ", "RF ") + "\n")

    if est is not None and len(est):
        lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
                 r"Dataset & split & seed & CV $R^2$ & permuted mean & permuted max & cR$^2_p$ \\", r"\midrule"]
        for r in est.sort_values(["dataset", "split", "seed"]).itertuples():
            lines.append(f"{DS[r.dataset]} & {r.split} & {r.seed} & {r.cv_r2:.3f} & {r.mean_scrambled:.3f} & {r.max_scrambled:.3f} & {r.cr2p:.3f} " + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(TAB, "table_s_yrand_rf.tex"), "w") as fh:
            fh.write("\n".join(lines) + "\n")

    scan.to_csv(os.path.join(RES, "offset_scan.csv"), index=False)


def main():
    for p in (FIG, TAB):
        os.makedirs(p, exist_ok=True)
    setup()
    d = pd.read_csv(os.path.join(RES, "benchmark.csv"))
    ep = os.path.join(RES, "yrand_estimator.csv")
    est = pd.read_csv(ep) if os.path.exists(ep) else None
    sp = os.path.join(RES, "offset_scan.csv")
    scan = pd.read_csv(sp) if os.path.exists(sp) else offset_scan(np.round(np.arange(-3, 3.01, 0.5), 2))
    fig_metrics(d, scan)
    fig_ad(d)
    fig_yrand_leak(d, est)
    tables(d, est, scan)
    print("figures written")


if __name__ == "__main__":
    main()
