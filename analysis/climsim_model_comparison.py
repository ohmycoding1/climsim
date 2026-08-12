#!/usr/bin/env python3
"""Compare Linear Regression / Random Forest / Neural Net on the ClimSim
subsampled low-res data, and probe the two effects from the write-up:

  (A) model choice vs. feature choice  -> which moves the score more?
  (B) near-deterministic outputs       -> do they inflate the score, and what
                                          happens to the score when excluded?
  (C) a relative-humidity-inspired feature (warm air holds more water vapour)
                                          -> does adding it help?

Data: the prenormalized arrays from https://huggingface.co/datasets/LEAP/subsampled_low_res
      train_input.npy  (N, 124)   train_target.npy (N, 128)
      (val_/scoring_ used automatically if present)

Input layout (124):  0-59 state_t (temperature, 60 levels)
                     60-119 state_q0001 (specific humidity, 60 levels)
                    120-123 state_ps, pbuf_SOLIN, pbuf_LHFLX, pbuf_SHFLX

NOTE on physics: the arrays are already normalized, so a *true* physical relative
humidity cannot be recovered here. The "RH-inspired" feature below is a
temperature x humidity interaction proxy in normalized space — enough to test the
"does a humidity-coupling feature help?" question, not a calibrated RH.

Usage:
    python climsim_model_comparison.py                 # data in ./data, 50k rows
    python climsim_model_comparison.py --data-dir data --n-rows 100000
    python climsim_model_comparison.py --rf-trees 100 --seed 0
"""
from __future__ import annotations

import argparse
import os
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---- ClimSim input variable layout -----------------------------------------
T_SLICE = slice(0, 60)      # state_t
Q_SLICE = slice(60, 120)    # state_q0001
SCALARS = slice(120, 124)   # ps, SOLIN, LHFLX, SHFLX

# ---- ClimSim target variable groups (128 outputs) --------------------------
TGT_GROUPS = {
    "ptend_t  (0-59)":   slice(0, 60),     # temperature tendency
    "ptend_q  (60-119)": slice(60, 120),   # moisture tendency
    "surface  (120-127)": slice(120, 128), # 8 surface fluxes
}


def load_split(data_dir: str, n_rows: int, seed: int):
    """Return (X_train, y_train, X_val, y_val), row-subsampled for tractability."""
    xtr = np.load(os.path.join(data_dir, "train_input.npy"), mmap_mode="r")
    ytr = np.load(os.path.join(data_dir, "train_target.npy"), mmap_mode="r")

    rng = np.random.default_rng(seed)
    idx = rng.choice(xtr.shape[0], size=min(n_rows, xtr.shape[0]), replace=False)
    idx.sort()
    X = np.asarray(xtr[idx], dtype=np.float32)
    y = np.asarray(ytr[idx], dtype=np.float32)

    val_x = os.path.join(data_dir, "val_input.npy")
    val_y = os.path.join(data_dir, "val_target.npy")
    if os.path.exists(val_x) and os.path.exists(val_y):
        vx = np.load(val_x, mmap_mode="r")
        vy = np.load(val_y, mmap_mode="r")
        vidx = rng.choice(vx.shape[0], size=min(n_rows // 4, vx.shape[0]), replace=False)
        vidx.sort()
        Xv = np.asarray(vx[vidx], dtype=np.float32)
        yv = np.asarray(vy[vidx], dtype=np.float32)
        src = "val_*.npy"
    else:  # fall back to an 80/20 split of the training rows
        cut = int(0.8 * len(X))
        X, Xv = X[:cut], X[cut:]
        y, yv = y[:cut], y[cut:]
        src = "80/20 split of train (val_*.npy not found)"
    return X, y, Xv, yv, src


def add_rh_feature(X: np.ndarray) -> np.ndarray:
    """Append a relative-humidity-inspired coupling feature.

    Physically, warmer air holds more water vapour, so temperature and humidity
    are coupled. In normalized space we approximate that coupling with the
    per-row mean of (state_t * state_q0001) across the 60 vertical levels, plus
    the near-surface (bottom level) product. Two extra columns."""
    t, q = X[:, T_SLICE], X[:, Q_SLICE]
    col_mean_tq = (t * q).mean(axis=1, keepdims=True)
    surface_tq = (t[:, -1] * q[:, -1]).reshape(-1, 1)   # bottom model level
    return np.hstack([X, col_mean_tq, surface_tq]).astype(np.float32)


def metrics(y_true, y_pred, cols=None):
    """MAE, RMSE, and two aggregate R2 flavours over target columns (default all).

    r2_unif : mean of per-column R2 (uniform_average). Dominated by low-variance
              output channels -> can go strongly negative even when MAE is small.
    r2_varw : variance-weighted R2. Weights each channel by its variance, so
              near-zero-variance channels stop dominating -> the meaningful number.
    """
    if cols is not None:
        y_true, y_pred = y_true[:, cols], y_pred[:, cols]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2_unif = r2_score(y_true, y_pred, multioutput="uniform_average")
    r2_varw = r2_score(y_true, y_pred, multioutput="variance_weighted")
    return mae, rmse, r2_unif, r2_varw


def group_r2(y_true, y_pred):
    """Variance-weighted R2 within each ClimSim target group."""
    return {g: r2_score(y_true[:, s], y_pred[:, s], multioutput="variance_weighted")
            for g, s in TGT_GROUPS.items()}


def find_deterministic_cols(X, y, Xv, yv, thresh=0.99):
    """Target columns a plain linear model already predicts almost perfectly."""
    lr = LinearRegression().fit(X, y)
    pv = lr.predict(Xv)
    per_col_r2 = np.array([r2_score(yv[:, j], pv[:, j]) for j in range(y.shape[1])])
    det = np.where(per_col_r2 >= thresh)[0]
    return det, per_col_r2


def build_models(rf_trees, seed):
    return {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=rf_trees, max_depth=20, n_jobs=-1, random_state=seed
        ),
        "NeuralNet(MLP)": MLPRegressor(
            hidden_layer_sizes=(128, 128), max_iter=60,
            early_stopping=True, random_state=seed,
        ),
    }


def row(name, m):
    return (f"{name:<26} MAE={m[0]:.4f}  RMSE={m[1]:.4f}  "
            f"R2unif={m[2]:+.3f}  R2varw={m[3]:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--n-rows", type=int, default=50_000)
    ap.add_argument("--rf-trees", type=int, default=60)
    ap.add_argument("--det-thresh", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    print("Loading data ...")
    X, y, Xv, yv, src = load_split(args.data_dir, args.n_rows, args.seed)
    print(f"  train {X.shape}  val {Xv.shape}  (val source: {src})\n")

    # --- (B) which target columns are near-deterministic? -------------------
    det, per_col_r2 = find_deterministic_cols(X, y, Xv, yv, args.det_thresh)
    keep = np.setdiff1d(np.arange(y.shape[1]), det)
    print(f"[B] near-deterministic outputs (linear R2 >= {args.det_thresh}): "
          f"{len(det)} of {y.shape[1]} columns -> {list(det)}")
    print("    these are trivially predictable; keeping them inflates the mean score.\n")

    # --- (A) model comparison on the base 124 features ----------------------
    print("[A] model comparison (base 124 features)")
    print("     scores shown ALL columns, then EXCLUDING near-deterministic ones")
    models = build_models(args.rf_trees, args.seed)
    preds = {}
    for name, model in models.items():
        model.fit(X, y)
        pv = model.predict(Xv)
        preds[name] = pv
        print("  " + row(name + " [all]", metrics(yv, pv)))
        if len(keep):
            print("  " + row(name + " [excl.det]", metrics(yv, pv, cols=keep)))
    if not len(keep):
        print("  (every column was near-deterministic at this threshold; "
              "raise --det-thresh to separate them)")
    print()

    # per-target-group variance-weighted R2 (where does each model do well?)
    print("    variance-weighted R2 by target group:")
    hdr = "      " + " " * 18 + "".join(f"{g:>22}" for g in TGT_GROUPS)
    print(hdr)
    for name, pv in preds.items():
        gr = group_r2(yv, pv)
        print(f"      {name:<18}" + "".join(f"{gr[g]:>22.3f}" for g in TGT_GROUPS))
    print()

    # --- (C) does the RH-inspired feature help? -----------------------------
    print("[C] add relative-humidity-inspired feature (T x q coupling)")
    eval_cols = keep if len(keep) else np.arange(y.shape[1])
    Xr, Xvr = add_rh_feature(X), add_rh_feature(Xv)
    for name in ("LinearRegression", "NeuralNet(MLP)"):
        base = build_models(args.rf_trees, args.seed)[name].fit(X, y)
        rh = build_models(args.rf_trees, args.seed)[name].fit(Xr, y)
        b = metrics(yv, base.predict(Xv), cols=eval_cols)
        r = metrics(yv, rh.predict(Xvr), cols=eval_cols)
        # index 3 = variance-weighted R2 (the meaningful aggregate)
        print(f"  {name:<18} base  R2varw={b[3]:+.4f}   +RH  R2varw={r[3]:+.4f}   "
              f"delta={r[3]-b[3]:+.4f}   (excl. near-deterministic cols)")

    print("\nTakeaway to check against your write-up:")
    print("  - swapping models moves R2 less than adding/removing feature groups")
    print("  - excluding near-deterministic columns lowers the headline R2 (more honest)")
    print("  - the humidity-coupling feature should nudge R2 up")


if __name__ == "__main__":
    main()
