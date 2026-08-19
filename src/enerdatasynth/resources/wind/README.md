# `enerdatasynth.resources.wind`

Wind's generative pipeline, vendored from a separate, previously-published Data-For-Good project (`wind_diffusion`) and nested here instead of kept as its own top-level package. `enerdatasynth.generators.WindDiffusionModel` wraps it. It has its own API, config, and checkpoint-resolution mechanism, and doesn't depend on the rest of `enerdatasynth`.

See [`docs/models/wind.md`](../../../../docs/models/wind.md) for the science; this file is about the code. [`../../README.md`](../../README.md)'s package map covers the parallel PJM-load structure this mirrors.

## Architecture

A conditional DDPM (Ho et al., 2020), not built here — vendored from a separate, previously-published project. Shared across all three channel presets (`1ch`/`2ch`/`3ch`); only the input/output channel count differs per preset.

| Property | Value |
|---|---|
| Denoiser | Transformer-style, `hidden_dim=128`, `num_heads=4` |
| Diffusion steps | `T=200` |
| Beta schedule | Cosine (`beta_start=1e-4`, `beta_end=0.02`) — chosen independently, not matched to load's linear default |
| Sampler | DDIM, with classifier-free guidance and dynamic thresholding |
| Sequence length | 288 steps (24h at 5-minute resolution) |
| Output | `(N, 288, 2)` always — `[..., 0]` = windspeed (m/s), `[..., 1]` = direction (degrees), a fixed resource-level contract independent of the preset's internal channel count |

Full detail, including per-preset training-recipe differences (`2ch`'s non-default joint-consistency loss): [`docs/models/wind.md`](../../../../docs/models/wind.md) §5–6.

## Results

Real numbers from `manifest.json`, measured against each preset's own validation split:

| Preset | `marginal_wasserstein` range | `regime_accuracy` range | Status |
|---|---|---|---|
| `1ch` | 2.00 – 2.25 | 0.41 (mid) – 0.69 (low) | Best-tuned |
| `3ch` | 1.16 – 2.67 | 0.41 (low) – 0.82 (high) | Healthy |
| `2ch` | — | — | Improved, but not equivalent to `1ch`/`3ch` — needs different generation settings (`eta=0.0`, no guidance) to stay stable |

**The one problem nothing fixed:** `mid`-regime steering never exceeds ~41–43% across every preset and every generation setting tried — barely above chance for a 3-way bucket.

Full per-regime tables and methodology: [`docs/models/wind.md`](../../../../docs/models/wind.md) §7–8, [`docs/report_cards/wind_presets.md`](../../../../docs/report_cards/wind_presets.md).

## Layout

Same category taxonomy as `resources/pjm_load/` wherever it applies (`data/`, `models/`, `diffusion/`, `training/`, `evaluation/`, `sampling/`, `visualization/`), plus wind-only infrastructure at this package's root (`api.py`, `config.py`, `registry.py`, `exceptions.py`, `utils.py`, `manifest.json`) rather than forcing it into a shared folder that wouldn't really fit.

| File/folder | Purpose |
|---|---|
| `api.py` | Public entry points: `list_models()`, `describe_model()`, `generate_wind_profile()` |
| `config.py` | `WindDiffusionConfig` — hyperparameters per preset (`1ch`/`2ch`/`3ch`) |
| `models/models.py` | The Transformer-style denoiser |
| `diffusion/diffusion.py` | Forward/reverse diffusion process (`DiffusionSchedule`) |
| `diffusion/samplers.py` | DDIM sampling with classifier-free guidance and dynamic thresholding |
| `data/data.py` | Raw-CSV loading, channel derivation, windowing. Needs data this repo doesn't ship — see `docs/models/wind.md` §2 |
| `data/transforms.py` | Feature engineering: wind-regime bucketing, ordinal/onehot encoding |
| `evaluation/evaluate.py` | `generate_multiple_for_condition`, `build_condition_vector` — what the `enerdatasynth` bridge calls |
| `evaluation/metrics.py` | Fidelity-gate metrics (`marginal_wasserstein`, `acf_rmse`, `regime_accuracy`, ...), documented in `docs/models/wind.md` §7 |
| `evaluation/failure_analysis.py` | Flags weak spots (near-chance `mid`-regime accuracy, for instance) already sitting in `manifest.json`'s real, measured numbers. Mirrors `resources/pjm_load/evaluation/failure_analysis.py`. It doesn't run a new comparison, since there's no real data here to compare against |
| `evaluation/results_report.py` | Regenerates `data/reports/wind/results_summary.md` across all three presets. Mirrors `resources/pjm_load/evaluation/results_report.py` |
| `sampling/generate.py` | `generate_for_regimes()` — a thin wrapper that generates one profile per regime (low/mid/high) in a single call, for parity with `resources/pjm_load/sampling/generate.py` |
| `registry.py` | `resolve_weights()`, the checkpoint resolution logic. `data/checkpoints/wind/` gets checked first — see `generators/wind_conditioning.py` |
| `manifest.json` | Preset metadata, metrics, recommended generation settings. Its `status: "pending"` only describes its own unused download path |
| `exceptions.py` | `CheckpointNotFoundError` and friends |
| `training/train.py` | The upstream project's training entry point, not exposed as a console script here |
| `visualization/plotting.py` | Real-vs-synthetic plots. Needs data this repo doesn't have |
| `utils.py` | Seeding and a few small shared helpers |

## Public API vs. internal

`api.py`'s three functions are the standalone entry point, independent of `enerdatasynth`. The bridge (`generators/wind.py`) calls `evaluation.evaluate`, `registry`, and `config` directly instead. Both work — `api.py` is just simpler if you don't need `enerdatasynth`'s conditioning conveniences.

## What should not be changed casually

`manifest.json`'s per-preset `recommended_generation` settings are measured-best values, not defaults picked for convenience — `2ch` needs `guidance_scale=1.0`, and anything nonzero measurably breaks it. See `docs/models/wind.md` §8 for why.

## What should not be placed here

Anything `enerdatasynth`-specific. This package is meant to stay meaningful on its own. Resource-specific glue belongs in `generators/wind*.py` instead.
