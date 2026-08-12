#!/usr/bin/env python3
"""Download the ClimSim subsampled low-resolution dataset.

Follows the ClimSim quickstart:
    https://leap-stc.github.io/ClimSim/quickstart.html

The subsampled + prenormalized low-res real-geography data used for training,
validation, and scoring in the ClimSim paper lives on the Hugging Face Hub at:

    https://huggingface.co/datasets/LEAP/subsampled_low_res

Files (.npy):
    train_input.npy    (~5.0 GB)   train_target.npy   (~5.2 GB)
    val_input.npy      (~0.7 GB)   val_target.npy     (~0.7 GB)
    scoring_input.npy  (~0.8 GB)   scoring_target.npy (~0.9 GB)

`scoring_*` can be treated as the test set.

Usage:
    pip install huggingface_hub
    python download_subsampled.py                 # all .npy files -> ./data
    python download_subsampled.py --split train    # only train_{input,target}.npy
    python download_subsampled.py --out /path/to/data
"""
from __future__ import annotations

import argparse
import sys

REPO_ID = "LEAP/subsampled_low_res"

SPLIT_FILES = {
    "train":   ["train_input.npy", "train_target.npy"],
    "val":     ["val_input.npy", "val_target.npy"],
    "scoring": ["scoring_input.npy", "scoring_target.npy"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="./data",
                   help="destination directory (default: ./data)")
    p.add_argument("--split", choices=[*SPLIT_FILES, "all"], default="all",
                   help="which split(s) to download (default: all)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("Missing dependency. Install it with:  pip install huggingface_hub")

    if args.split == "all":
        files = [f for group in SPLIT_FILES.values() for f in group]
    else:
        files = SPLIT_FILES[args.split]

    print(f"Downloading {len(files)} file(s) from {REPO_ID} -> {args.out}\n")
    for name in files:
        print(f"  fetching {name} ...")
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=name,
            repo_type="dataset",
            local_dir=args.out,
        )
        print(f"    -> {path}")

    print("\nDone. Point the ClimSim data loader at:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
