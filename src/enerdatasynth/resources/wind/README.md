# `enerdatasynth.resources.wind`

The wind generation pipeline. It was copied (vendored) from `wind_diffusion`, a separate
Data-For-Good project, and lives here instead of as its own top-level package.
`enerdatasynth.generators.WindDiffusionModel` wraps it. The pipeline has its own API,
config and checkpoint lookup, and doesn't depend on the rest of `enerdatasynth`.

This file covers the code. For the model itself, see
[`docs/models/wind.md`](../../../../docs/models/wind.md). The package map in
[`../../README.md`](../../README.md) covers the matching PJM load layout.

## Architecture

A conditional DDPM (Ho et al., 2020), built upstream rather than here. All three channel
presets (`1ch`, `2ch`, `3ch`) use it; only the number of input and output channels
changes.

| Property | Value |
|---|---|
| Denoiser | Transformer, `hidden_dim=128`, `num_heads=4` |
| Diffusion steps | `T=200` |
| Beta schedule | Cosine (`beta_start=1e-4`, `beta_end=0.02`); load uses linear |
| Sampler | DDIM with classifier-free guidance and dynamic thresholding |
| Sequence length | 288 steps (24 hours at 5-minute resolution) |
| Output | Always `(N, 288, 2)`: `[..., 0]` is windspeed (m/s), `[..., 1]` is direction (degrees), whatever the preset's internal channel count |

Training recipes for each preset, including `2ch`'s extra joint-consistency loss, are in
[`docs/models/wind.md`](../../../../docs/models/wind.md) §5–6.

## Results

From `manifest.json`, measured against each preset's validation split:

| Preset | `marginal_wasserstein` range | `regime_accuracy` range | Status |
|---|---|---|---|
| `1ch` | 2.00–2.25 | 0.41 (mid) to 0.69 (low) | Best tuned |
| `3ch` | 1.16–2.67 | 0.41 (low) to 0.82 (high) | Good |
| `2ch` | n/a | n/a | Improved, but not equivalent to `1ch`/`3ch`; needs `eta=0.0` and no guidance to stay stable |

For `1ch` and `3ch`, `mid` regime accuracy has never gone above about 41–43% under any
generation setting tried, barely better than chance with three regimes.

Full per-regime tables: [`docs/models/wind.md`](../../../../docs/models/wind.md) §7–8 and
[`docs/report_cards/wind_presets.md`](../../../../docs/report_cards/wind_presets.md).

## Layout

Subfolders follow `resources/pjm_load/` where they apply (`data/`, `models/`,
`diffusion/`, `training/`, `evaluation/`, `sampling/`, `visualization/`). Files that
only wind needs (`api.py`, `config.py`, `registry.py`, `exceptions.py`, `utils.py`,
`manifest.json`) sit at the package root.

| File/folder | Purpose |
|---|---|
| `api.py` | Public entry points: `list_models()`, `describe_model()`, `generate_wind_profile()` |
| `config.py` | `WindDiffusionConfig`, the hyperparameters for each preset |
| `models/models.py` | The transformer denoiser |
| `diffusion/diffusion.py` | Forward and reverse diffusion (`DiffusionSchedule`) |
| `diffusion/samplers.py` | DDIM sampling with classifier-free guidance and dynamic thresholding |
| `data/data.py` | Raw CSV loading, channel derivation and windowing. Needs data this repository doesn't include (`docs/models/wind.md` §2) |
| `data/transforms.py` | Wind-regime bucketing and ordinal/one-hot encoding |
| `evaluation/evaluate.py` | `generate_multiple_for_condition` and `build_condition_vector`, which the `enerdatasynth` wrapper calls |
| `evaluation/metrics.py` | Evaluation metrics (`marginal_wasserstein`, `acf_rmse`, `regime_accuracy` and others), described in `docs/models/wind.md` §7 |
| `evaluation/failure_analysis.py` | Flags weak spots, such as near-chance `mid` accuracy, in the numbers already in `manifest.json`. It doesn't run a new comparison, because there's no real data here. Matches `resources/pjm_load/evaluation/failure_analysis.py` |
| `evaluation/results_report.py` | Regenerates `data/reports/wind/results_summary.md` for all three presets. Matches `resources/pjm_load/evaluation/results_report.py` |
| `sampling/generate.py` | `generate_for_regimes()`, which generates a batch for each regime (low, mid, high) in one call |
| `registry.py` | `resolve_weights()`, the checkpoint lookup. `data/checkpoints/wind/` is checked first (see `generators/wind_conditioning.py`) |
| `manifest.json` | Preset metadata, metrics and recommended generation settings. Its `status: "pending"` refers only to an unused download location |
| `exceptions.py` | `CheckpointNotFoundError` and related errors |
| `training/train.py` | The upstream training entry point. It isn't installed as a command here |
| `visualization/plotting.py` | Real-vs-synthetic plots. Needs real data this repository doesn't include |
| `utils.py` | Seeding and a few small helpers |

## Public API vs. internal

The three functions in `api.py` can be used on their own, without `enerdatasynth`. The
wrapper (`generators/wind.py`) calls `evaluation.evaluate`, `registry` and `config`
directly instead. Either route works; `api.py` is simpler if you don't need
`enerdatasynth`'s conditioning helpers.

## What should not be changed casually

The `recommended_generation` settings for each preset in `manifest.json` were chosen
from measurements. For example, `2ch` needs `guidance_scale=1.0`, and any higher value
breaks it. See `docs/models/wind.md` §8.

## What should not be placed here

Anything specific to `enerdatasynth`. This package should keep working on its own; glue
code belongs in `generators/wind*.py`.
