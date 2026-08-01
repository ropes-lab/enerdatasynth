# Quick Start

## 1. Get raw data in place

Put PJM `hrl_load_metered_*.csv` files and `pjm_hourly_temps_wide.csv`
in `data/raw/` (already populated in this repository; see
`data/raw/PUT_RAW_CSVS_HERE.txt` for the expected filenames if starting
from scratch, and `src/pjm_diffusion/data/temperature.py` to rebuild
the temperature file from NOAA).

## 2. Preprocess

```bash
pjm-diffusion preprocess
```

Builds `data/processed/{X_daily_profiles,Z_conditions}[_val].npy`,
`metadata.json`, and a frozen backup at `data/processed_backup/`.

## 3. Train

```bash
pjm-diffusion train                      # transformer, validated defaults
pjm-diffusion train --config configs/mlp.yaml   # legacy MLP baseline instead
```

Writes `checkpoints/<run_id>/{best.pt,latest.pt,loss_curve.png,result.json}`
and `outputs/logs/<run_id>_training_log.csv`.

## 4. Calibrate (recommended)

```bash
pjm-diffusion calibrate --run-id transformer_T1000_H64_L2
```

Fits the scalar bias correction (see `docs/evaluation.md`) and saves
`checkpoints/<run_id>/calibration.json`. `LoadDiffusionModel` applies
it automatically once it exists.

## 5. Generate

```python
from pjm_diffusion import LoadDiffusionModel
import numpy as np

model = LoadDiffusionModel.from_run_id("transformer_T1000_H64_L2")

# z: (N, cond_dim) conditioning vectors — here, reusing the real
# validation set's conditions as an example.
z = np.load("data/processed/Z_conditions_val.npy")[:10]
samples = model.sample(z, seed=0)   # (10, 24), calibration applied automatically
```

## 6. Evaluate

```bash
pjm-diffusion evaluate --run-id transformer_T1000_H64_L2
pjm-diffusion report
```

Writes the three validation plots, `reports/failure_analysis.md`, and
`reports/results_summary.md`.

## All at once

```bash
pjm-diffusion pipeline   # preprocess -> sweep -> generate -> calibrate -> validate -> report
```

See `scripts/run_all.py` for the equivalent as a plain script, and
`examples/` for small standalone demonstrations of each piece.
