"""
Applies QSARCert 1.1.0 and the current version to every model built by run_models.py and compares
each computation with an independent reference.

    python validation/benchmark.py

References:
    metrics      scikit-learn r2_score / mean_absolute_error, scipy pearsonr, direct formulas of
                 Consonni et al. (2009) and Lin (1989)
    leverage     QR decomposition of the training design (h = ||R^-T x||^2)
    Y-random.    expected chance R^2 = p/(n - 1) for OLS (Ruecker et al. 2007)
    leakage      canonical SMILES and Bemis-Murcko scaffolds (RDKit)
Output: validation/results/benchmark.csv and yrand_estimator.csv
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process; the pool provides the parallelism

import ctypes
import glob
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(HERE, "legacy_v110"))
warnings.filterwarnings("ignore")

from qsarcert.core import (calculate_applicability_domain, calculate_oecd_metrics,  # noqa: E402
                           check_split_leakage, perform_y_randomization)
import applicability_domain_v110 as ad110  # noqa: E402
import scaffold_leakage_v110 as lk110  # noqa: E402
import y_randomization_v110 as yr110  # noqa: E402

RUNS = os.path.join(HERE, "results", "runs")
CACHE = os.path.join(HERE, "results", "bench_cache")
FEAT = os.path.join(HERE, "results", "features")
N_PERM = 100


def tropsha_uncentred(y, p):
    """Golbraikh-Tropsha verdict with R0^2 computed with uncentred total sums of squares, the
    convention of regression-through-origin routines (e.g. Excel, statsmodels without constant)."""
    r2 = pearsonr(y, p)[0] ** 2
    k = np.sum(y * p) / np.sum(p ** 2)
    kp = np.sum(y * p) / np.sum(y ** 2)
    r0 = 1 - np.sum((y - k * p) ** 2) / np.sum(y ** 2)
    r0p = 1 - np.sum((p - kp * y) ** 2) / np.sum(p ** 2)
    return r0, r0p, r2


def safe_call(row, key, fn, *a, **k):
    """Runs one check; a numerical failure is recorded in the row instead of stopping the benchmark."""
    try:
        return fn(*a, **k)
    except Exception as e:  # noqa: BLE001
        row[key + "_error"] = f"{type(e).__name__}: {e}"
        return None


def one(path):
    cache = os.path.join(CACHE, os.path.basename(path)[:-4] + ".json")
    if os.path.exists(cache):
        return json.load(open(cache))
    row = _one(path)
    row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    json.dump(row, open(cache, "w"))
    return row


def _one(path):
    name = os.path.basename(path)[:-4]
    ds, split, seed, model = name.split("_")
    d = np.load(path, allow_pickle=True)
    f = np.load(os.path.join(FEAT, f"{ds}.npz"), allow_pickle=True)
    tr, te = d["train"], d["test"]
    y = d["y"]
    ytr, yte, ptr, pte = y[tr], y[te], d["pred_train"], d["pred_test"]
    xtr, xte = d["x_train"], d["x_test"]
    row = {"dataset": ds, "split": split, "seed": int(seed), "model": model, "n_train": len(tr), "n_test": len(te),
           "p": xtr.shape[1]}

    # --- metrics against independent implementations
    m = calculate_oecd_metrics(yte, pte, y_train=ytr)
    q2f1 = 1 - np.sum((yte - pte) ** 2) / np.sum((yte - ytr.mean()) ** 2)
    q2f3 = 1 - (np.sum((yte - pte) ** 2) / len(yte)) / (np.sum((ytr - ytr.mean()) ** 2) / len(ytr))
    ccc = 2 * np.cov(yte, pte, bias=True)[0, 1] / (yte.var() + pte.var() + (yte.mean() - pte.mean()) ** 2)
    row.update(q2_f1=m.q2_f1, q2_f2=m.q2_f2, q2_f3=m.q2_f3, r2_pearson=m.r2_pearson, ccc=m.ccc, rmse=m.rmse,
               rm2=m.r_m_average, delta_rm2=m.delta_r_m_2, k=m.k_slope, k_prime=m.k_prime_slope,
               r0_2=m.r0_2, r0p_2=m.r0_prime_2, tropsha=m.tropsha_passed, metric_status=m.status)
    row["dev_metrics"] = max(abs(m.q2_f2 - r2_score(yte, pte)), abs(m.q2_f1 - q2f1), abs(m.q2_f3 - q2f3),
                             abs(m.r2_pearson - pearsonr(yte, pte)[0] ** 2), abs(m.ccc - ccc),
                             abs(m.mae - mean_absolute_error(yte, pte)))
    r0u, r0pu, r2 = tropsha_uncentred(yte, pte)
    c4u = ((abs(r2 - r0u) / r2 < 0.1) or (abs(r2 - r0pu) / r2 < 0.1)) and abs(r0u - r0pu) < 0.3
    rmu = (r2 * (1 - np.sqrt(abs(r2 - r0u))) + r2 * (1 - np.sqrt(abs(r2 - r0pu)))) / 2
    drmu = abs(r2 * np.sqrt(abs(r2 - r0u)) - r2 * np.sqrt(abs(r2 - r0pu)))
    c3 = (0.85 <= m.k_slope <= 1.15) or (0.85 <= m.k_prime_slope <= 1.15)
    row["tropsha_uncentred"] = bool(m.q2_ext > 0.5 and r2 > 0.6 and c3 and c4u and drmu < 0.2 and rmu > 0.5
                                    and m.ccc >= 0.8)
    row.update(r0_2_unc=r0u, r0p_2_unc=r0pu, rm2_unc=rmu)

    # --- applicability domain
    err = np.abs(yte - pte)
    new = calculate_applicability_domain(xtr, xte, yte, pte, ytr - ptr)
    old = safe_call(row, "ad110", ad110.calculate_applicability_domain, xtr, xte, yte, pte, ytr - ptr)
    if old is not None:
        row.update(ad110_status=old.status, ad110_pct_in=old.pct_in_domain, ad110_hstar=old.warning_leverage)
    lev = np.array(new.leverages)
    out = lev > new.warning_leverage
    row.update(ad_status=new.status, ad_pct_in=new.pct_in_domain, ad_hstar=new.warning_leverage, ad_rank=new.design_rank,
               ad_mae_in=err[~out].mean() if (~out).any() else np.nan, ad_mae_out=err[out].mean() if out.any() else np.nan,
               ad_rho_lev_err=spearmanr(lev, err)[0])
    if new.status != "NOT_APPLICABLE":
        mu, sd = xtr.mean(0), xtr.std(0)
        sd[sd == 0] = 1
        a = np.hstack([np.ones((len(xtr), 1)), (xtr - mu) / sd])
        b = np.hstack([np.ones((len(xte), 1)), (xte - mu) / sd])
        _, r = np.linalg.qr(a)
        diag = np.abs(np.diag(r))
        if diag.min() > 1e-8 * diag.max():  # full column rank: the QR reference is well defined
            ref = np.sum(np.linalg.solve(r.T, b.T) ** 2, axis=0)
            row["dev_leverage_rel"] = float(np.max(np.abs(lev - ref) / ref))

    # --- Y-randomization (surrogate)
    yn = perform_y_randomization(xtr, ytr, n_iterations=N_PERM, random_seed=int(seed))
    yo = safe_call(row, "yr110", yr110.perform_y_randomization, xtr, ytr, 0.0, n_iterations=N_PERM, random_seed=int(seed))
    row.update(yr_status=yn.status, yr_mean=yn.mean_scrambled_r2, yr_expected=yn.expected_chance_r2, yr_cr2p=yn.cr2_p)
    if yo is not None:
        row.update(yr110_status=yo.status, yr110_mean=yo.mean_scrambled_r2, yr110_orig=yo.original_r2, yr110_cr2p=yo.cr2_p)

    # --- leakage
    smi = f["smiles"]
    ln = check_split_leakage(xtr, xte, smiles_train=list(smi[tr]), smiles_test=list(smi[te]))
    lo = safe_call(row, "lk110", lk110.check_split_leakage, xtr, xte)
    fp = f["fp"]
    rows_tr = {r_.tobytes() for r_ in fp[tr]}
    row.update(lk_status=ln.status, lk_exact=ln.n_exact_duplicates, lk_high=ln.n_high_similarity_pairs,
               lk_dup_struct=ln.n_duplicate_structures, lk_scaffold=ln.frac_shared_scaffold, lk_mean_nn=ln.mean_nn_similarity,
               ref_dup_smiles=int(np.isin(smi[te], smi[tr]).sum()),
               ref_identical_fp=int(sum(r_.tobytes() in rows_tr for r_ in fp[te])))
    if lo is not None:
        row.update(lk110_status=lo.status, lk110_exact=lo.n_exact_duplicates, lk110_high=lo.n_high_similarity_pairs,
                   lk110_mean_nn=lo.mean_nn_similarity)
    return row


def yrand_estimator(path):
    """Y-randomization of the random forest with the model itself (5-fold CV R^2), 20 permutations,
    100 trees."""
    name = os.path.basename(path)[:-4]
    ds, split, seed, model = name.split("_")
    d = np.load(path)
    ytr = d["y"][d["train"]]
    est = RandomForestRegressor(n_estimators=100, random_state=int(seed), n_jobs=1)
    res = perform_y_randomization(d["x_train"], ytr, n_iterations=20, random_seed=int(seed), estimator=est)
    return {"dataset": ds, "split": split, "seed": int(seed), "status": res.status, "cv_r2": res.original_r2,
            "mean_scrambled": res.mean_scrambled_r2, "max_scrambled": res.max_scrambled_r2, "cr2p": res.cr2_p,
            "p_value": res.p_value}


def main():
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # keep the machine awake
    paths = sorted(glob.glob(os.path.join(RUNS, "*.npz")))
    os.makedirs(CACHE, exist_ok=True)
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, paths, chunksize=2))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "benchmark.csv"), index=False)
    print(len(rows), "runs")
    # the model-based test refits a random forest 5 x 21 times per model; it is run on the two smaller
    # datasets (Lipophilicity would take about 14 h per model on this machine)
    rf = [p for p in paths if p.endswith("_rf.npz") and int(os.path.basename(p).split("_")[2]) < 3
          and not os.path.basename(p).startswith("lipo")]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        est = list(ex.map(yrand_estimator, rf))
    pd.DataFrame(est).to_csv(os.path.join(HERE, "results", "yrand_estimator.csv"), index=False)
    print(len(est), "estimator runs")


if __name__ == "__main__":
    main()
