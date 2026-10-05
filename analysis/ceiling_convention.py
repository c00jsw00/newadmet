#!/usr/bin/env python3
"""Oracle-ceiling convention, made exact and reproducible (manuscript §2.5).

DECLARED CONVENTION (single, used for both endpoints):
    oracle prediction p_i = y_i           for un-censored test rows
                      p_i = mean(y_test)  for floor test rows (y_i <= -10.0)
    ceiling R2 = 1 - SSE_oracle / SST_test   on the v4.2 test partition.

This module reproduces the published values EXACTLY:
    PAMPA  R2 = 0.5387  (manuscript: 0.539)
    Caco-2 R2 = 0.5696  (manuscript: 0.570)

It also reports every nearby convention so the choice is auditable:
    floor -> train-set mean        0.5426 / 0.5721
    floor -> full-table mean       0.5422 / 0.5724
    floor -> train censored mean   1.0 (DEGENERATE: the train censored mean IS
                                 the floor value -10.0; never use this)
    floor -> test floor mean       1.0 (degenerate for the same reason)

Requires ONLY numpy + pandas + the two committed CSVs (no torch, no feature
cache, no GPU). Runtime < 10 s.

Run:  python analysis/ceiling_convention.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FLOOR = -10.0
SEED = 42  # the single canonical v4.2 split seed

ENDPOINTS = {
    "PAMPA": ("data/pepadmet_pampa_mdck.csv", "PAMPA_MDCK"),
    "Caco-2": ("data/pepadmet_caco2.csv", "Caco2"),
}


def canonical_split(smiles_list, seed=SEED, frac=(0.70, 0.10, 0.20)):
    """Verbatim v4.2 unique-SMILES split (same algorithm as analysis/common.py)."""
    uniq, inv = np.unique(np.asarray(smiles_list, dtype=object), return_inverse=True)
    n = len(smiles_list)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_tr = int(round(len(uniq) * frac[0]))
    n_va = int(round(len(uniq) * frac[1]))
    ids = {"tr": set(perm[:n_tr].tolist()),
           "va": set(perm[n_tr:n_tr + n_va].tolist()),
           "te": set(perm[n_tr + n_va:].tolist())}
    return tuple(np.array([i for i in range(n) if inv[i] in ids[k]], dtype=np.int64)
                 for k in ("tr", "va", "te"))


def r2(y_true, y_pred):
    sst = ((y_true - y_true.mean()) ** 2).sum()
    sse = ((y_true - y_pred) ** 2).sum()
    return 1.0 - sse / sst


def main():
    out = {}
    for name, (path, col) in ENDPOINTS.items():
        df = pd.read_csv(os.path.join(ROOT, path))
        y = df[col].to_numpy(dtype=np.float64)
        tr, va, te = canonical_split(df["smiles"].astype(str).tolist())
        y_te, y_tr = y[te], y[tr]
        fl = y <= FLOOR + 1e-9
        f_te = fl[te]
        sst_share = ((y_te[f_te] - y_te.mean()) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum()
        conv = {
            "test_mean (DECLARED)":      r2(y_te, np.where(f_te, y_te.mean(), y_te)),
            "train_mean":                r2(y_te, np.where(f_te, y_tr.mean(), y_te)),
            "full_table_mean":           r2(y_te, np.where(f_te, y.mean(), y_te)),
            "train_censored_mean":       r2(y_te, np.where(f_te, y_tr[fl[tr]].mean(), y_te)),
            "test_floor_mean":           r2(y_te, np.where(f_te, y_te[f_te].mean(), y_te)),
        }
        out[name] = dict(n_rows=len(y), n_test=len(te), n_floor_test=int(f_te.sum()),
                         floor_share_test_SST=round(float(sst_share), 4),
                         ceilings={k: round(v, 4) for k, v in conv.items()})
        print(f"{name}: test rows={len(te)}  floor rows={int(f_te.sum())}  "
              f"floor share of test SST={sst_share:.4f}")
        for k, v in conv.items():
            print(f"    ceiling ({k:<22}) R2 = {v:.4f}")
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(main(), indent=2))
