# ClimSim — Subsampled dataset download

Helpers to download the **subsampled low-resolution** ClimSim dataset described in the
[ClimSim quickstart](https://leap-stc.github.io/ClimSim/quickstart.html).

## What this is

The subsampled + prenormalized low-res real-geography data used for training, validation,
and scoring in the ClimSim paper, hosted on the Hugging Face Hub:

- Dataset: <https://huggingface.co/datasets/LEAP/subsampled_low_res>

| Split | Files | Approx. size |
|-------|-------|--------------|
| train   | `train_input.npy`, `train_target.npy`     | ~10.2 GB |
| val     | `val_input.npy`, `val_target.npy`         | ~1.5 GB  |
| scoring (test) | `scoring_input.npy`, `scoring_target.npy` | ~1.7 GB  |

`scoring_*` can be treated as the test set. Full dataset ≈ **13 GB**, so make sure you have
enough disk space and bandwidth before downloading everything.

## Download

Python (recommended):

```bash
pip install huggingface_hub
python download_subsampled.py              # all splits -> ./data
python download_subsampled.py --split train   # just the training arrays
python download_subsampled.py --out /path/to/data
```

Shell (Hugging Face CLI):

```bash
pip install -U "huggingface_hub[cli]"
./download_subsampled.sh                   # all splits -> ./data
OUT=/path/to/data ./download_subsampled.sh
```

## Load

```python
import numpy as np
train_input  = np.load("data/train_input.npy")
train_target = np.load("data/train_target.npy")
```

## Notes

- If your machine is behind a proxy/firewall that blocks `huggingface.co`, run these on a
  network that allows it — the download must reach the Hugging Face Hub.
- The subsampled arrays can also be regenerated from the full dataset via the upstream
  `preprocessing/create_npy_data_splits.ipynb` notebook in the
  [ClimSim repo](https://github.com/leap-stc/ClimSim).
