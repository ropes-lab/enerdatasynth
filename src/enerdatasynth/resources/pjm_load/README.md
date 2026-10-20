# `enerdatasynth.resources.pjm_load`

The generative pipeline for PJM load. It was written for this project; wind, by contrast, is vendored from elsewhere. `enerdatasynth.generators.LoadDiffusionModel` wraps it for `enerdatasynth.model()`, and the package also has its own standalone `api.py` that doesn't depend on the rest of `enerdatasynth`.

For the science, see [`docs/models/pjm_load.md`](../../../../docs/models/pjm_load.md). This file is about the code. [`../wind/README.md`](../wind/README.md) covers the parallel wind structure and what differs on purpose: load has preprocessing, training, sweeping and calibration written from scratch, while wind ships pre-trained checkpoints and has none of those.

## Layout

The folders match `resources/wind/` wherever that makes sense (`data/`, `models/`, `diffusion/`, `training/`, `evaluation/`, `sampling/`, `visualization/`). Load adds `api.py`, `config.py`, `exceptions.py` and `manifest.json` at the package root.

| File/folder | Purpose |
|---|---|
| `api.py` | Public entry points: `generate_load_profile()`, `list_models()`, `describe_model()`. Added to match `resources/wind/api.py` |
| `config.py` | Thin re-export of `enerdatasynth.utils.config` (`RunConfig`, path constants, defaults). `PROJECT_ROOT`, `DATA_ROOT` and `ensure_dirs()` stay in the shared module; the docstring says why |
| `exceptions.py` | `PjmLoadError`, `CheckpointNotFoundError`, `InvalidConditionError`. Mirrors `resources/wind/exceptions.py`. They subclass the builtin `FileNotFoundError` and `ValueError`, so existing `except` blocks keep working |
| `manifest.json` | Metadata for the bundled checkpoints, collected from `data/checkpoints/pjm_load/*/result.json` (architecture, params, best validation loss, epochs, calibrated). Same shape as `resources/wind/manifest.json` |
| `data/preprocess.py` | Raw-CSV loading, DST and missing-data cleaning, conditioning-vector construction |
| `data/temperature.py` | NOAA ISD fetch and zone-weighted temperature averaging |
| `models/transformer.py` | Transformer denoiser, the recommended architecture and the shipped default |
| `models/denoiser.py` | MLP denoiser, the legacy baseline (see `docs/models/pjm_load.md` §5) |
| `diffusion/ddpm.py` | Forward and reverse diffusion, with the corrected ancestral sampler |
| `diffusion/schedules.py` | Beta schedules: linear by default, cosine and quadratic as alternatives |
| `training/train.py` | Single-config training |
| `training/sweep.py` | Hyperparameter sweep over architectures and configs |
| `sampling/generate.py` | `generate_for_run`, which `enerdatasynth generate` calls for this resource |
| `evaluation/calibration.py` | Scalar bias correction, fit on the training set |
| `evaluation/metrics.py` | The 7-metric suite: Hourly, FirstDiff, ACF, RMSE, Peak, Energy, Ramp |
| `evaluation/baselines.py` | Classical ARIMA/ARMA baselines to compare the diffusion model against |
| `evaluation/failure_analysis.py` | Flags weak conditional slices in evaluation results |
| `evaluation/results_report.py` | Rebuilds `data/reports/pjm_load/results_summary.md` |
| `visualization/plots.py` | Validation plots: real vs. generated, per-metric breakdowns |

Training, sweeping, calibration, evaluation and plotting need the processed arrays, which this repository doesn't ship. Build them with `enerdatasynth preprocess` on your own copy of the PJM and NOAA data. Generating from the shipped checkpoints does not need them.

## Two ways in

`api.py` is the standalone entry point. It doesn't depend on `enerdatasynth` and returns a `LoadProfile` dataclass instead of a bare array. The bridge (`generators/load.py` and `generators/load_conditioning.py`) calls `data.preprocess`, `sampling.generate` and `evaluation.calibration` directly. Both work; `api.py` is simpler if you don't need the conditioning helpers from `enerdatasynth`.

## What doesn't belong here

Anything specific to `enerdatasynth` itself. This package should stay usable on its own, the same rule `resources/wind/` follows. Glue code goes in `generators/load*.py`.
