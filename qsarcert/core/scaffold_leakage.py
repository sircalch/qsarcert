"""
Train/test split leakage: duplicated inputs, near-identical compounds and shared scaffolds.

Similarity between feature vectors is measured on a scale that suits the features:

* binary fingerprints: Tanimoto (Jaccard) similarity;
* continuous descriptors: no similarity threshold is applied. Cosine similarity of raw descriptors
  is dominated by a few large-valued descriptors (molecular weight, surface areas) and is close to
  1 for almost every pair of molecules, and in a space of a few standardised descriptors the
  nearest neighbour is almost always within any fixed angle. Only identical input vectors are
  counted, and the nearest-neighbour cosine similarity after standardisation is reported for
  information. Pass SMILES for a structural check.

When SMILES are given (and RDKit is installed) the check also reports compounds of the test set
that duplicate a training compound and the fraction that share a Bemis-Murcko scaffold with it.
"""

from typing import Optional, Sequence
from dataclasses import dataclass
import numpy as np


@dataclass
class SplitLeakageResult:
    n_train: int
    n_test: int
    n_exact_duplicates: int  # test rows whose feature vector equals a training row
    n_high_similarity_pairs: int  # test rows with Tanimoto NN similarity >= threshold (-1: continuous features)
    mean_nn_similarity: float
    max_nn_similarity: float
    status: str  # 'PASS', 'WARNING', 'FAIL'
    diagnostic_message: str
    similarity: str = ""  # 'tanimoto' or 'cosine (standardised)'
    n_duplicate_structures: int = -1  # test compounds whose canonical SMILES is in training (-1: not checked)
    frac_shared_scaffold: float = float("nan")  # fraction of test compounds with a training scaffold


def _structure_checks(smiles_train, smiles_test):
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        return -1, float("nan")
    RDLogger.DisableLog("rdApp.*")

    def canon(smiles):
        out = []
        for s in smiles:
            m = Chem.MolFromSmiles(s)
            out.append((Chem.MolToSmiles(m), MurckoScaffold.MurckoScaffoldSmiles(mol=m)) if m else (None, None))
        return out

    tr, te = canon(smiles_train), canon(smiles_test)
    can_tr = {c for c, _ in tr if c}
    scaf_tr = {s for _, s in tr if s}  # acyclic compounds have an empty scaffold and are not counted
    dup = sum(1 for c, _ in te if c and c in can_tr)
    shared = [s in scaf_tr for _, s in te if s]
    return int(dup), float(np.mean(shared)) if shared else float("nan")


def check_split_leakage(
    x_train: np.ndarray,
    x_test: np.ndarray,
    similarity_threshold: float = 0.95,
    smiles_train: Optional[Sequence[str]] = None,
    smiles_test: Optional[Sequence[str]] = None
) -> SplitLeakageResult:
    """
    Evaluates duplicated inputs and extreme similarity between the training and test sets.

    Parameters
    ----------
    x_train, x_test : np.ndarray
        Feature matrices used by the model.
    similarity_threshold : float, default 0.95
    smiles_train, smiles_test : sequence of str, optional
        Structures, for duplicate-structure and scaffold checks (requires RDKit).
    """
    tr = np.asarray(x_train, dtype=float)
    te = np.asarray(x_test, dtype=float)
    n_tr, n_te = len(tr), len(te)

    # exact duplicates of the input vector (the model cannot tell these compounds apart)
    tr_rows = {row.tobytes() for row in np.ascontiguousarray(tr)}
    exact_dups = int(sum(row.tobytes() in tr_rows for row in np.ascontiguousarray(te)))

    binary = bool(np.isin(tr, (0.0, 1.0)).all() and np.isin(te, (0.0, 1.0)).all())
    if binary:
        kind = "tanimoto"
        inter = te @ tr.T
        union = te.sum(1)[:, None] + tr.sum(1)[None, :] - inter
        sim = np.divide(inter, union, out=np.ones_like(inter), where=union > 0)
    else:
        kind = "cosine (standardised)"
        mu, sd = tr.mean(0), tr.std(0)
        sd[sd == 0] = 1.0
        a, b = (tr - mu) / sd, (te - mu) / sd
        na, nb = np.linalg.norm(a, axis=1, keepdims=True), np.linalg.norm(b, axis=1, keepdims=True)
        na[na == 0] = 1.0
        nb[nb == 0] = 1.0
        sim = (b / nb) @ (a / na).T
    nn = sim.max(axis=1) if n_tr else np.zeros(n_te)
    high_sim = int(np.sum(nn >= similarity_threshold)) if binary else -1
    mean_nn = float(np.mean(nn)) if n_te else 0.0
    max_nn = float(np.max(nn)) if n_te else 0.0

    dup_struct, frac_scaf = (-1, float("nan"))
    if smiles_train is not None and smiles_test is not None:
        dup_struct, frac_scaf = _structure_checks(smiles_train, smiles_test)

    parts = [f"{exact_dups} test compound(s) have the same feature vector as a training compound"]
    if binary:
        parts.append(f"{high_sim} have a training neighbour with Tanimoto similarity >= {similarity_threshold:g}")
    if dup_struct >= 0:
        parts.append(f"{dup_struct} duplicate a training structure")
    txt = "; ".join(parts) + "."
    if np.isfinite(frac_scaf):
        txt += f" {frac_scaf:.0%} of the ring-containing test compounds share a Bemis-Murcko scaffold with training."
    if not binary and dup_struct < 0:
        txt += " Continuous descriptors: only identical inputs are detected; pass SMILES for a structural check."

    frac_dup = exact_dups / max(1, n_te)
    frac_high = max(0, high_sim) / max(1, n_te)
    if dup_struct > 0 or frac_dup > 0.05 or frac_high > 0.20:
        status = "FAIL"
        diag = "Train/test leakage: " + txt + " External metrics may be optimistic."
    elif exact_dups > 0 or frac_high > 0.05:
        status = "WARNING"
        diag = "Near-duplicates between training and test sets: " + txt
    else:
        status = "PASS"
        diag = "No duplicated compounds between training and test sets: " + txt

    return SplitLeakageResult(
        n_train=n_tr,
        n_test=n_te,
        n_exact_duplicates=exact_dups,
        n_high_similarity_pairs=high_sim,
        mean_nn_similarity=mean_nn,
        max_nn_similarity=max_nn,
        status=status,
        diagnostic_message=diag,
        similarity=kind,
        n_duplicate_structures=dup_struct,
        frac_shared_scaffold=frac_scaf,
    )
