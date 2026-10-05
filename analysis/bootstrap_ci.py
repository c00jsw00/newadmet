#!/usr/bin/env python3
"""Paired bootstrap 95% CI for delta-R2 between two models (manuscript v2.0 §2.5).

Resamples test-set ROWS with common random numbers (each bootstrap replicate
uses the same row indices for both models — this is what makes the CI "paired"
and what removes test-sampling error from the comparison; inter-seed SD alone
does not).  10,000 replicates, percentile method.

Usage
-----
    python analysis/bootstrap_ci.py PAMPA preds_baseline_PAMPA.npz preds_tabpfn_v2_desc.npz

Input .npz contract (produced by dump_predictions.py):
    y      : full label vector (all rows)
    te     : integer test-row indices (canonical v4.2 split)
    one or more prediction keys named p* : full-length prediction vectors
Any two keys are compared pairwise; `y_*` and index keys are ignored.

Output: JSON on stdout with mean delta, CI bounds, and P(delta<=0).
"""
import json
import sys

import numpy as np

N_BOOT = 10_000
SEED = 20_261_005  # fixed bootstrap seed for reproducibility


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def r2_on(y_true, y_pred, rows):
    yt, yp = y_true[rows], y_pred[rows]
    sst = ((yt - yt.mean()) ** 2).sum()
    return 1.0 - ((yt - yp) ** 2).sum() / sst


def paired_delta_ci(y, p_a, p_b, te, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(te)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        rows = te[rng.integers(0, n, n)]
        deltas[b] = r2_on(y, p_a, rows) - r2_on(y, p_b, rows)
    return {
        "delta_point": float(r2_on(y, p_a, te) - r2_on(y, p_b, te)),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "p_delta_le_0": float((deltas <= 0).mean()),
        "n_boot": n_boot,
    }


def main(path_a, path_b, label=""):
    A, B = load_npz(path_a), load_npz(path_b)
    # locate a shared index vector and label vector
    y = A["y"] if "y" in A else B["y"]
    te = A["te"] if "te" in A else B["te"]
    pa_keys = [k for k in A if k.startswith("p")]
    pb_keys = [k for k in B if k.startswith("p")]
    out = {"comparison": f"{path_a} vs {path_b}", "note": label, "pairs": {}}
    for ka in pa_keys:
        for kb in pb_keys:
            res = paired_delta_ci(y, A[ka], B[kb], te)
            out["pairs"][f"{ka} - {kb}"] = res
            print(f"{ka} vs {kb}: dR2={res['delta_point']:+.4f} "
                  f"95%CI[{res['ci_lo']:+.4f},{res['ci_hi']:+.4f}] "
                  f"P(d<=0)={res['p_delta_le_0']:.4f}", file=sys.stderr)
    return out


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1], sys.argv[2]), indent=2))
