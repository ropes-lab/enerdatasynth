# Quick Start

## If a model is already trained (the common case)

Run inside Python (a `.py` file, `python3` interactive shell, or a notebook) — not pasted directly into your terminal's shell prompt:

```python
import enerdatasynth

load = enerdatasynth.model("pjm_load", zone="PS")
load.set(month=7, Tmax=86, Tmin=70, weekday=True, holiday=False)
samples = load.sample(n=100, seed=0)   # (100, 24) ndarray, MW
```

Use `zone="PS"` (the default) for PSE&G in New Jersey, or any other PJM zone (`"AE"`, `"AP"`, `"BC"`, `"CE"`, `"DAY"`, `"DEOK"`, `"DOM"`, `"DPL"`, `"DUQ"`, `"EKPC"`, `"JC"`, `"ME"`, `"PE"`, `"PL"`, `"PN"`, `"RECO"`); each has its own trained model. The call is otherwise identical.

That's it — no tensors, checkpoints, or architecture choices to think about. See [`docs/api_reference.md`](api_reference.md) for the full `.set()`/`.sample()` contract, and [`MODEL_CARD.md`](../MODEL_CARD.md) for what this model was trained on and its limitations.

The trained checkpoints and the scaler files (`data/processed/pjm_load/metadata.json` for PS, and `zones/<ZONE>/metadata.json` beside it for the other zones) are committed here and also bundled in the PyPI wheel. There is no real data, so the above works immediately after `pip install enerdatasynth` (see [`docs/installation.md`](installation.md)). The rest of this page covers training a model from scratch, which needs your own copy of the data.

## 1. Get raw data in place

Put PJM `hrl_load_metered_*.csv` files and `pjm_hourly_temps_wide.csv`
in `data/raw/pjm_load/`. This repository ships no real data, so you download it
yourself: [`data/README.md`](../data/README.md#data-sources) lists the sources,
`data/raw/pjm_load/README.md` the expected filenames, and
`src/enerdatasynth/resources/pjm_load/data/temperature.py` rebuilds the temperature
file from NOAA.

## 2. Preprocess

(For a zone other than PS, add `--zone DOM` to this and to the training and calibration commands below; that zone's files go in a `zones/DOM/` subfolder, so nothing for PS is touched.)

```bash
enerdatasynth preprocess
```

Builds `data/processed/pjm_load/{X_daily_profiles,Z_conditions}[_val].npy`,
`metadata.json`, and a backup at `data/processed/pjm_load_backup/`. All of these except `metadata.json` are git-ignored.

## 3. Train

```bash
enerdatasynth train                             # transformer, validated defaults
enerdatasynth train --config data/configs/pjm_load/mlp.yaml   # legacy MLP baseline instead
```

Writes `data/checkpoints/pjm_load/<run_id>/{best.pt,latest.pt,loss_curve.png,result.json}`
and `data/outputs/pjm_load/logs/<run_id>_training_log.csv`. See [`docs/training.md`](training.md) for the full training workflow (sweeps, configs, reproducibility).

## 4. Calibrate (recommended)

```bash
enerdatasynth calibrate --run-id transformer_T1000_H64_L2
```

Fits the scalar bias correction (see [`docs/evaluation.md`](evaluation.md)) and saves
`data/checkpoints/pjm_load/<run_id>/calibration.json`. Both `enerdatasynth.model(...)` and
`LoadDiffusionModel` apply it automatically once it exists.

## 5. Generate

Via the fluent SDK (see above), or the lower-level API if you already have raw conditioning vectors:

```python
from enerdatasynth import LoadDiffusionModel
import numpy as np

model = LoadDiffusionModel.from_run_id("transformer_T1000_H64_L2")

# z: (N, cond_dim) conditioning vectors. This example reuses the validation
# set's conditions, which `enerdatasynth preprocess` builds from your own data.
z = np.load("data/processed/pjm_load/Z_conditions_val.npy")[:10]
samples = model.sample(z, seed=0)   # (10, 24), calibration applied automatically
```

## 6. Evaluate

```bash
enerdatasynth evaluate --run-id transformer_T1000_H64_L2
enerdatasynth report
```

Writes the three validation plots, `data/reports/pjm_load/failure_analysis.md`, and
`data/reports/pjm_load/results_summary.md`.

## All at once

```bash
enerdatasynth pipeline   # preprocess -> sweep -> generate -> calibrate -> validate -> report
```

See `examples/pjm_load/run_all.py` for the equivalent as a plain script, [`docs/examples.md`](examples.md) for runnable example scripts, and [`docs/training.md`](training.md) for the full training reference.
