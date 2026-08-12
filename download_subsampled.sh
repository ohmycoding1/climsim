#!/usr/bin/env bash
# Download the ClimSim subsampled low-resolution dataset.
#
# Follows the ClimSim quickstart:
#   https://leap-stc.github.io/ClimSim/quickstart.html
#
# Subsampled + prenormalized low-res real-geography data (train / val / scoring),
# hosted on the Hugging Face Hub:
#   https://huggingface.co/datasets/LEAP/subsampled_low_res
#
# Usage:
#   ./download_subsampled.sh            # -> ./data  (all .npy files)
#   OUT=/path/to/data ./download_subsampled.sh
#
# Requires the huggingface CLI:  pip install -U "huggingface_hub[cli]"
set -euo pipefail

REPO_ID="LEAP/subsampled_low_res"
OUT="${OUT:-./data}"

FILES=(
  train_input.npy   train_target.npy
  val_input.npy     val_target.npy
  scoring_input.npy scoring_target.npy
)

mkdir -p "$OUT"

# Prefer the modern `hf` CLI; fall back to the legacy `huggingface-cli`.
if command -v hf >/dev/null 2>&1; then
  DL=(hf download "$REPO_ID" --repo-type dataset --local-dir "$OUT")
elif command -v huggingface-cli >/dev/null 2>&1; then
  DL=(huggingface-cli download "$REPO_ID" --repo-type dataset --local-dir "$OUT")
else
  echo "Need the Hugging Face CLI. Install with:  pip install -U 'huggingface_hub[cli]'" >&2
  exit 1
fi

echo "Downloading ${#FILES[@]} file(s) from $REPO_ID -> $OUT"
"${DL[@]}" "${FILES[@]}"

echo "Done. Data in: $OUT"
