"""
Builds real QSAR models on three public MoleculeNet sets and stores everything QSARCert needs.

    python validation/run_models.py

Datasets (validation/data, downloaded from the MoleculeNet/DeepChem mirror; SHA-256 in README):
    ESOL (Delaney 2004, log solubility, 1128), FreeSolv (Mobley & Guthrie 2014, hydration free
    energy, 642), Lipophilicity (AstraZeneca/ChEMBL logD7.4, 4200).
Splits: random 80/20 and Bemis-Murcko scaffold 80/20, 10 seeds each.
Models: MLR on 6 descriptors chosen by forward selection on the training set (classic QSAR);
    ridge regression on all RDKit 2D descriptors; random forest on Morgan fingerprints.
Output: validation/results/runs/<dataset>_<split>_<seed>_<model>.npz
"""
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results", "runs")
FEAT = os.path.join(HERE, "results", "features")
SETS = {"esol": ("delaney-processed.csv", "smiles", "measured log solubility in mols per litre"),
        "freesolv": ("SAMPL.csv", "smiles", "expt"),
        "lipo": ("Lipophilicity.csv", "smiles", "exp")}
SEEDS = range(10)
N_MLR = 6


def featurize(name):
    path = os.path.join(FEAT, f"{name}.npz")
    if os.path.exists(path):
        return dict(np.load(path, allow_pickle=True))
    fn, scol, ycol = SETS[name]
    df = pd.read_csv(os.path.join(DATA, fn))
    mols = [Chem.MolFromSmiles(s) for s in df[scol]]
    keep = [i for i, m in enumerate(mols) if m is not None]
    mols = [mols[i] for i in keep]
    y = df[ycol].to_numpy(float)[keep]
    smiles = np.array([Chem.MolToSmiles(m) for m in mols])
    names = [n for n, _ in Descriptors.descList]
    desc = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=float)
    ok = np.isfinite(desc).all(0) & (np.nanstd(desc, 0) > 0) & (np.abs(desc).max(0) < 1e8)
    desc, names = desc[:, ok], np.array(names)[ok]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8)
    scaf = np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols])
    d = {"y": y, "smiles": smiles, "desc": desc, "desc_names": names, "fp": fp, "scaffold": scaf}
    os.makedirs(FEAT, exist_ok=True)
    np.savez_compressed(path, **d)
    return d


def split(n, scaf, kind, seed, frac=0.8):
    rng = np.random.default_rng(seed)
    if kind == "random":
        idx = rng.permutation(n)
        k = int(round(frac * n))
        return np.sort(idx[:k]), np.sort(idx[k:])
    groups = {}
    for i, s in enumerate(scaf):
        groups.setdefault(s, []).append(i)
    sets = list(groups.values())
    rng.shuffle(sets)
    # big scaffold groups first (as in DeepChem), random order within equal sizes
    sets.sort(key=len, reverse=True)
    train, test = [], []
    for g in sets:
        (train if len(train) + len(g) <= frac * n else test).extend(g)
    return np.sort(train), np.sort(test)


def forward_select(x, y, k):
    chosen = []
    for _ in range(k):
        best, best_rss = None, np.inf
        for j in range(x.shape[1]):
            if j in chosen:
                continue
            a = np.column_stack([np.ones(len(y)), x[:, chosen + [j]]])
            coef, rss, rank, _ = np.linalg.lstsq(a, y, rcond=None)
            r = float(rss[0]) if len(rss) else float(np.sum((y - a @ coef) ** 2))
            if rank == a.shape[1] and r < best_rss:
                best, best_rss = j, r
        chosen.append(best)
    return chosen


def run(args):
    name, kind, seed = args
    d = featurize(name)
    y, desc, fp = d["y"], d["desc"], d["fp"].astype(float)
    tr, te = split(len(y), d["scaffold"], kind, seed)
    done = []
    for model in ("mlr", "ridge", "rf"):
        out = os.path.join(OUT, f"{name}_{kind}_{seed}_{model}.npz")
        if os.path.exists(out):
            continue
        if model == "mlr":
            sel = forward_select(desc[tr], y[tr], N_MLR)
            x = desc[:, sel]
            m = LinearRegression().fit(x[tr], y[tr])
            extra = {"selected": np.array(d["desc_names"][sel])}
        elif model == "ridge":
            sc = StandardScaler().fit(desc[tr])
            x = sc.transform(desc)
            m = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(x[tr], y[tr])
            extra = {"alpha": m.alpha_}
            x = desc  # QSARCert standardizes internally; give it the raw descriptors
        else:
            x = fp
            m = RandomForestRegressor(n_estimators=500, n_jobs=1, random_state=seed, oob_score=True).fit(x[tr], y[tr])
            extra = {"oob_r2": m.oob_score_}
        xm = sc.transform(desc) if model == "ridge" else x
        np.savez_compressed(out, train=tr, test=te, y=y, pred_train=m.predict(xm[tr]), pred_test=m.predict(xm[te]),
                            x_train=x[tr], x_test=x[te], **extra)
        done.append(model)
    return name, kind, seed, done


def main():
    os.makedirs(OUT, exist_ok=True)
    names = sys.argv[1:] or list(SETS)
    for n in names:
        featurize(n)
    jobs = [(n, k, s) for n in names for k in ("random", "scaffold") for s in SEEDS]
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        for r in ex.map(run, jobs):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
