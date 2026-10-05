#!/usr/bin/env python3
"""Dump per-molecule test predictions for every re-runnable route.

Writes one .npz per route into results/predictions/ with the common schema

    y   : full label vector (aligned to data/pepadmet_pampa_mdck.csv row order)
    te  : canonical v4.2 test-row indices (seed 42)
    tr  : canonical train-row indices
    p   : full-length prediction vector (NaN outside the test rows is fine),
          or p<seed> keys for multi-seed routes

Anything that follows the schema can be fed to bootstrap_ci.py directly.

Re-runnable routes (no GPU training required):
  * baseline MLP      (needs models_v4/<ep>/admet_mlp.pt + scaler.pt +
                       data/molformer/molformer_emb_*.npz)
  * TabPFN v2 desc    (needs the 217-descriptor block; see run_tabpfn_v2.py)
  * LightGBM (route3) (trained here in <5 min)

KPGT routes must dump predictions from the fine-tuning loop
(`train_kpgt.py --dump-test-predictions results/predictions/kpgt_s<seed>.npz`).
"""
import os

import numpy as np
import pandas as pd

import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FLOOR = -10.0
SEED = 42
OUT = os.path.join(ROOT, "results", "predictions")


def canonical_split(smiles, seed=SEED):
    uniq, inv = np.unique(np.asarray(smiles, dtype=object), return_inverse=True)
    n = len(smiles)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    n_tr = int(round(len(uniq) * 0.70))
    n_va = int(round(len(uniq) * 0.10))
    ids = {"tr": set(perm[:n_tr].tolist()),
           "va": set(perm[n_tr:n_tr + n_va].tolist()),
           "te": set(perm[n_tr + n_va:].tolist())}
    return tuple(np.array([i for i in range(n) if inv[i] in ids[k]], dtype=np.int64)
                 for k in ("tr", "va", "te"))


def dump_lightgbm_endpoint(csv_path, col, name):
    """Route-3-style LightGBM regressor; cheap re-runnable comparator."""
    import lightgbm as lgb
    import feature_extractor

    df = pd.read_csv(os.path.join(ROOT, csv_path))
    sm = df["smiles"].astype(str).tolist()
    y = df[col].to_numpy(np.float64)
    X = feature_extractor.molecule_features(sm).astype(np.float32)
    tr, va, te = canonical_split(sm)
    params = dict(objective="regression", metric="l2", num_leaves=63,
                  learning_rate=0.05, num_threads=8, verbose=-1, seed=42)
    dtr = lgb.Dataset(X[tr], y[tr])
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    booster = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    p = np.full(len(y), np.nan)
    p[te] = booster.predict(X[te])
    np.savez(os.path.join(OUT, f"{name}.npz"), y=y, te=te, tr=tr, p=p)
    from sklearn.metrics import r2_score
    print(f"{name}: test R2 = {r2_score(y[te], p[te]):.4f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    dump_lightgbm_endpoint("data/pepadmet_pampa_mdck.csv", "PAMPA_MDCK", "lightgbm_pampa")
    dump_lightgbm_endpoint("data/pepadmet_caco2.csv", "Caco2", "lightgbm_caco2")
    print("Wrote route predictions to", OUT)


if __name__ == "__main__":
    main()
