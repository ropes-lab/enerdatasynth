# `enerdatasynth.resources.pjm_load`

PJM load's generative pipeline, built for this project (wind, by contrast, is vendored from elsewhere). `enerdatasynth.generators.LoadDiffusionModel` wraps it for the `enerdatasynth.model()` bridge. This package also has its own standalone `api.py` entry point that doesn't depend on the rest of `enerdatasynth`.

See [`docs/models/pjm_load.md`](../../../../docs/models/pjm_load.md) for the science; this file is about the code. [`../wind/README.md`](../wind/README.md) covers the parallel wind structure this mirrors, and what's different on purpose — this resource has full from-scratch preprocessing, training, sweeping, and calibration, none of which wind has, since wind ships pre-trained checkpoints instead.

## Layout

Same category taxonomy as `resources/wind/` wherever it applies (`data/`, `models/`, `diffusion/`, `training/`, `evaluation/`, `sampling/`, `visualization/`), plus load-only infrastructure at this package's root: `api.py`, `config.py`, `exceptions.py`, `manifest.json`.

| File/folder | Purpose |
|---|---|
| `api.py` | Public entry points: `generate_load_profile()`, `list_models()`, `describe_model()`. Added for parity with `resources/wind/api.py` |
| `config.py` | Thin re-export of `enerdatasynth.utils.config` (`RunConfig`, path constants, defaults). Not a migration — the module docstring explains why `PROJECT_ROOT`/`DATA_ROOT`/`ensure_dirs()` stay in the shared module |
| `exceptions.py` | `PjmLoadError` base, `CheckpointNotFoundError`, `InvalidConditionError`. Mirrors `resources/wind/exceptions.py`; both subclass the builtin `FileNotFoundError`/`ValueError` they replace, so existing `except` sites keep working |
| `manifest.json` | Bundled-checkpoint metadata, aggregated from the real `data/checkpoints/pjm_load/*/result.json` files (architecture, params, best validation loss, epochs trained, calibrated). Shaped to match `resources/wind/manifest.json` |
| `data/preprocess.py` | Raw-CSV loading, DST/missing-data cleaning, conditioning-vector construction |
| `data/temperature.py` | NOAA ISD fetch and zone-weighted temperature aggregation |
| `models/transformer.py` | Transformer denoiser, the recommended architecture |
| `models/denoiser.py` | MLP denoiser — the legacy baseline, and still the current package default (see `docs/models/pjm_load.md` §5) |
| `diffusion/ddpm.py` | Forward/reverse diffusion process, corrected ancestral sampling |
| `diffusion/schedules.py` | Beta schedules — linear by default, with cosine/quadratic alternatives |
| `training/train.py` | Single-config training entry point |
| `training/sweep.py` | Hyperparameter sweep over architectures and configs |
| `sampling/generate.py` | Generates synthetic profiles from a trained checkpoint (`generate_for_run`) — what `enerdatasynth generate` calls for this resource |
| `evaluation/calibration.py` | Scalar bias correction, fit on the training set |
| `evaluation/metrics.py` | The 7-metric evaluation suite: Hourly, FirstDiff, ACF, RMSE, Peak, Energy, Ramp |
| `evaluation/baselines.py` | Classical ARIMA/ARMA baselines the diffusion model gets compared against |
| `evaluation/failure_analysis.py` | Flags weak conditional slices from real evaluation results |
| `evaluation/results_report.py` | Regenerates `data/reports/pjm_load/results_summary.md` |
| `visualization/plots.py` | Validation plots — real vs. generated, per-metric breakdowns |

## Public API vs. internal

`api.py` is the standalone entry point, independent of `enerdatasynth`, added here second for structural parity (wind's came first). The bridge (`generators/load.py`, `generators/load_conditioning.py`) calls `data.preprocess`, `sampling.generate`, and `evaluation.calibration` directly instead. Both are valid paths — `api.py` is simpler if you don't need `enerdatasynth`'s conditioning conveniences, and it returns a `LoadProfile` dataclass rather than a bare array.

## What should not be placed here

Anything `enerdatasynth`-specific. This package is meant to stay meaningful on its own, same convention as `resources/wind/README.md`. Resource-specific glue belongs in `generators/load*.py` instead.
