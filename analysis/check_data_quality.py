#!/usr/bin/env python3
"""Data-quality report for the ClimSim subsampled .npy arrays.

Checks each available file (train/val/scoring x input/target) for:
  - shape / dtype
  - missing values (NaN) and non-finite values (+/-inf)   [full scan, chunked]
  - global min / max / mean / std                          [sampled]
  - zero-variance (constant) columns                       [sampled]
  - sparsity: fraction of exact zeros                      [sampled]
  - heavy-tailed / outlier columns (max |z| over a sample) [sampled]

NaN/inf are scanned over the FULL file in chunks (cheap); distribution stats use
a random row sample for speed. "Prenormalized" means the arrays are already
scaled, so values are typically O(1) but not guaranteed standard-normal.

Usage:
    python check_data_quality.py --data-dir data
    python check_data_quality.py --data-dir data --sample 200000
"""
from __future__ import annotations

import argparse
import os
import warnings
import numpy as np

FILES = [
    "train_input.npy", "train_target.npy",
    "val_input.npy", "val_target.npy",
    "scoring_input.npy", "scoring_target.npy",
]


def scan_nonfinite(a, chunk):
    """Full-file chunked scan for NaN and inf counts."""
    n = a.shape[0]
    nan = inf = 0
    for i in range(0, n, chunk):
        b = np.asarray(a[i:i + chunk], dtype=np.float64)
        nan += int(np.isnan(b).sum())
        inf += int(np.isinf(b).sum())
    return nan, inf


def report_file(path, sample, chunk):
    a = np.load(path, mmap_mode="r")
    n, d = a.shape
    print(f"\n=== {os.path.basename(path)} ===")
    print(f"  shape={a.shape}  dtype={a.dtype}")

    nan, inf = scan_nonfinite(a, chunk)
    tot = n * d
    flag = "  <-- has missing/!finite values!" if (nan or inf) else ""
    print(f"  NaN={nan} ({100*nan/tot:.4g}%)   inf={inf} ({100*inf/tot:.4g}%){flag}")

    # sampled distribution stats
    rng = np.random.default_rng(0)
    m = min(sample, n)
    idx = np.sort(rng.choice(n, m, replace=False))
    s = np.asarray(a[idx], dtype=np.float64)
    finite = s[np.isfinite(s)]
    print(f"  sample={m} rows | min={finite.min():.4g}  max={finite.max():.4g}  "
          f"mean={finite.mean():.4g}  std={finite.std():.4g}")

    col_std = np.nanstd(s, axis=0)
    const_cols = np.where(col_std == 0)[0]
    print(f"  zero-variance (constant) columns: {len(const_cols)}"
          + (f" -> {list(const_cols)}" if 0 < len(const_cols) <= 20 else ""))

    zero_frac = (s == 0).mean()
    print(f"  exact-zero fraction (sparsity): {100*zero_frac:.2f}%")

    # heavy-tail / outlier columns: standardized, look at largest |z|
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.abs((s - s.mean(0)) / np.where(col_std == 0, np.nan, col_std))
    max_abs_z = np.nanmax(z)
    n_extreme = int((z > 10).sum())
    print(f"  max |z| in sample: {max_abs_z:.1f}   points with |z|>10: {n_extreme}"
          f"  ({'heavy tails present' if max_abs_z > 10 else 'no strong outliers'})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--sample", type=int, default=200_000)
    ap.add_argument("--chunk", type=int, default=200_000)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    found = [f for f in FILES if os.path.exists(os.path.join(args.data_dir, f))]
    if not found:
        raise SystemExit(f"No ClimSim .npy files found in {args.data_dir!r}")
    print(f"Checking {len(found)} file(s) in {args.data_dir!r}: {found}")
    for f in found:
        report_file(os.path.join(args.data_dir, f), args.sample, args.chunk)

    print("\nHow to read this:")
    print("  - NaN/inf should be 0. Anything >0 must be cleaned before training.")
    print("  - constant columns carry no information (safe to drop).")
    print("  - high sparsity is expected for PRECSC/PRECC and upper-level moisture.")
    print("  - large max|z| = heavy tails (e.g. convective precip); not an error, but")
    print("    it is what drags the naive per-column R2 negative.")


if __name__ == "__main__":
    main()
