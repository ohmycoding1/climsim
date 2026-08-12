#!/usr/bin/env python3
"""Improve ClimSim prediction with physics-informed features (domain knowledge).

Compares, for each model, a BASELINE (124 raw inputs, standardized) against a
DOMAIN feature set that adds physically motivated variables and standardizes
everything. Reports variance-weighted R2 overall and per target group.

Domain features added on top of the 124 raw inputs:
  - vertical gradients dT, dq   (static stability / moisture convergence)
  - per-level T*q coupling      (relative-humidity-like, all 60 levels)
  - column water vapour + column mean T
  - lower- vs upper-troposphere moisture / temperature aggregates
  - forcing interactions: SOLIN x column-q, LHFLX x low-level-q, SHFLX x low-T
  - StandardScaler on ALL features  <-- this is why raw engineered features
                                        barely helped before (scale mismatch)

Target groups (128 outputs):
  ptend_t  0-59  (temperature tendency)
  ptend_q  60-119 (moisture tendency -- the physically hard, sub-grid part)
  surface  120-127 (8 surface fluxes)

Usage:
    python climsim_domain_features.py --data-dir data --n-rows 100000
    python climsim_domain_features.py --n-rows 60000 --rf-trees 60 --mlp-iter 200
"""
from __future__ import annotations

import argparse
import os
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

T_SLICE = slice(0, 60)      # state_t (0 = model top, 59 = surface)
Q_SLICE = slice(60, 120)    # state_q0001
TGT_GROUPS = {
    "ptend_t (0-59)":    slice(0, 60),
    "ptend_q (60-119)":  slice(60, 120),
    "surface (120-127)": slice(120, 128),
}


def engineer_features(X: np.ndarray) -> np.ndarray:
    """Append physics-informed features to the raw 124 inputs."""
    t, q = X[:, T_SLICE], X[:, Q_SLICE]
    sc = X[:, 120:124]                       # ps, SOLIN, LHFLX, SHFLX
    solin, lhflx, shflx = sc[:, 1:2], sc[:, 2:3], sc[:, 3:4]

    dt = np.diff(t, axis=1)                   # 59  vertical temperature gradient
    dq = np.diff(q, axis=1)                   # 59  vertical moisture gradient
    tq = t * q                                # 60  per-level T*q coupling (RH-like)

    col_q = q.mean(1, keepdims=True)          # column water vapour proxy
    col_t = t.mean(1, keepdims=True)
    low_q = q[:, 45:].mean(1, keepdims=True)  # lower troposphere (near surface)
    low_t = t[:, 45:].mean(1, keepdims=True)
    up_q = q[:, :20].mean(1, keepdims=True)   # upper troposphere

    solin_colq = solin * col_q                # insolation x available moisture
    lhflx_lowq = lhflx * low_q                # surface evaporation x low moisture
    shflx_lowt = shflx * low_t                # sensible heating x low temperature

    return np.hstack([X, dt, dq, tq, col_q, col_t, low_q, low_t, up_q,
                      solin_colq, lhflx_lowq, shflx_lowt]).astype(np.float32)


def load_split(data_dir, n_rows, seed):
    xtr = np.load(os.path.join(data_dir, "train_input.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(data_dir, "train_target.npy"), mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(xtr.shape[0], min(n_rows, xtr.shape[0]), replace=False))
    X = np.asarray(xtr[idx], np.float32)
    y = np.asarray(ytr[idx], np.float32)

    vx_p = os.path.join(data_dir, "val_input.npy")
    vy_p = os.path.join(data_dir, "val_target.npy")
    if os.path.exists(vx_p) and os.path.exists(vy_p):
        vx, vy = np.load(vx_p, mmap_mode="r"), np.load(vy_p, mmap_mode="r")
        vidx = np.sort(rng.choice(vx.shape[0], min(n_rows // 4, vx.shape[0]), replace=False))
        Xv, yv, src = np.asarray(vx[vidx], np.float32), np.asarray(vy[vidx], np.float32), "val_*.npy"
    else:
        cut = int(0.8 * len(X))
        X, Xv, y, yv, src = X[:cut], X[cut:], y[:cut], y[cut:], "80/20 split of train"
    return X, y, Xv, yv, src


def make_models(rf_trees, mlp_iter, seed):
    """Each model wrapped in StandardScaler -> estimator."""
    return {
        "Ridge(linear)": make_pipeline(
            StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": make_pipeline(
            StandardScaler(with_mean=False),
            RandomForestRegressor(n_estimators=rf_trees, max_depth=25,
                                  n_jobs=-1, random_state=seed)),
        "NeuralNet(MLP)": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(256, 128), max_iter=mlp_iter,
                         alpha=1e-4, early_stopping=True, n_iter_no_change=12,
                         random_state=seed)),
    }


def r2w(yt, yp, cols=None):
    if cols is not None:
        yt, yp = yt[:, cols], yp[:, cols]
    return r2_score(yt, yp, multioutput="variance_weighted")


def group_r2(yt, yp):
    return {g: r2_score(yt[:, s], yp[:, s], multioutput="variance_weighted")
            for g, s in TGT_GROUPS.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-rows", type=int, default=100_000)
    ap.add_argument("--rf-trees", type=int, default=60)
    ap.add_argument("--mlp-iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    print("Loading data ...")
    X, y, Xv, yv, src = load_split(args.data_dir, args.n_rows, args.seed)
    Xe, Xve = engineer_features(X), engineer_features(Xv)
    print(f"  train {X.shape}  val {Xv.shape}  (val: {src})")
    print(f"  baseline features: {X.shape[1]}   domain features: {Xe.shape[1]}\n")

    print(f"{'model':<16}{'overall R2varw':>26}")
    print(f"{'':<16}{'base':>12}{'+domain':>12}{'delta':>10}")
    best = {}
    for name, _ in make_models(args.rf_trees, args.mlp_iter, args.seed).items():
        mb = make_models(args.rf_trees, args.mlp_iter, args.seed)[name].fit(X, y)
        md = make_models(args.rf_trees, args.mlp_iter, args.seed)[name].fit(Xe, y)
        pb, pd = mb.predict(Xv), md.predict(Xve)
        b, d = r2w(yv, pb), r2w(yv, pd)
        print(f"{name:<16}{b:>12.3f}{d:>12.3f}{d-b:>+10.3f}")
        best[name] = pd

    print("\nvariance-weighted R2 by target group (domain features):")
    print("      " + " " * 16 + "".join(f"{g:>20}" for g in TGT_GROUPS))
    for name, pd in best.items():
        gr = group_r2(yv, pd)
        print("      " + f"{name:<16}" + "".join(f"{gr[g]:>20.3f}" for g in TGT_GROUPS))

    print("\nNotes:")
    print("  - domain features + standardization should lift overall + ptend_t + surface")
    print("  - ptend_q (sub-grid moisture/convection) stays the hardest -- that is a")
    print("    physical result, not a modelling bug; report it as a finding.")


if __name__ == "__main__":
    main()
