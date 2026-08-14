# `enerdatasynth.resources.wind`

Wind's generative pipeline — vendored from a separate, previously-published
Data-For-Good project (`wind_diffusion`), nested here instead of kept as a separate
top-level package. `enerdatasynth.generators.WindDiffusionModel` wraps it; it has its
own independent API, config, and checkpoint-resolution mechanism and doesn't depend on
the rest of `enerdatasynth`.

See [`docs/models/wind.md`](../../../../docs/models/wind.md) for the science; this file
is about the code. See [`../../README.md`](../../README.md)'s package map for the
parallel PJM-load structure this mirrors.

## Layout

Same category taxonomy as `resources/pjm_load/` wherever a category genuinely applies
(`data/`, `models/`, `diffusion/`, `training/`, `evaluation/`, `sampling/`,
`visualization/`), plus wind-only infrastructure at this package's root (`api.py`,
`config.py`, `registry.py`, `exceptions.py`, `utils.py`, `manifest.json`) rather than
forced into a fake shared folder.

| File/folder | Purpose |
|---|---|
| `api.py` | Public entry points: `list_models()`, `describe_model()`, `generate_wind_profile()` |
| `config.py` | `WindDiffusionConfig` — hyperparameters per preset (`1ch`/`2ch`/`3ch`) |
| `models/models.py` | The Transformer-style denoiser |
| `diffusion/diffusion.py` | Forward/reverse diffusion process (`DiffusionSchedule`) |
| `diffusion/samplers.py` | DDIM sampling with classifier-free guidance and dynamic thresholding |
| `data/data.py` | Raw-CSV loading, channel derivation, windowing — needs data this repo doesn't ship (`docs/models/wind.md` §2) |
| `data/transforms.py` | Feature engineering (wind-regime bucketing, ordinal/onehot encoding) |
| `evaluation/evaluate.py` | `generate_multiple_for_condition`, `build_condition_vector` — what the `enerdatasynth` bridge calls |
| `evaluation/metrics.py` | Fidelity-gate metrics (`marginal_wasserstein`, `acf_rmse`, `regime_accuracy`, ...) — `docs/models/wind.md` §7 |
| `evaluation/failure_analysis.py` | Flags weak spots (e.g. near-chance `mid`-regime accuracy) already present in `manifest.json`'s real, measured numbers — mirrors `resources/pjm_load/evaluation/failure_analysis.py`. Does not run a new comparison; there's no real data to compare against (`docs/models/wind.md` §2) |
| `evaluation/results_report.py` | Regenerates `data/reports/wind/results_summary.md` across all three presets — mirrors `resources/pjm_load/evaluation/results_report.py` |
| `sampling/generate.py` | `generate_for_regimes()` — thin wrapper generating one profile per regime (low/mid/high) in one call, for structural parity with `resources/pjm_load/sampling/generate.py` |
| `registry.py` | `resolve_weights()` checkpoint resolution. **`data/checkpoints/wind/` is checked first** — see `generators/wind_conditioning.py` |
| `manifest.json` | Preset metadata/metrics/recommended generation settings. Its `status: "pending"` describes only its own (unused here) download path |
| `exceptions.py` | `CheckpointNotFoundError` and friends |
| `training/train.py` | Upstream project's training entry point — not exposed as a console script here |
| `visualization/plotting.py` | Real-vs-synthetic plots — needs data this repo doesn't have |
| `utils.py` | Seeding and small shared helpers |

## Public API vs. internal

`api.py`'s three functions are the standalone entry point (independent of
`enerdatasynth`). The bridge (`generators/wind.py`) calls `evaluation.evaluate`/
`registry`/`config` directly instead — both valid; `api.py` is simpler without
`enerdatasynth`'s conditioning conveniences.

## What should not be changed casually

`manifest.json`'s per-preset `recommended_generation` settings — measured-best values
(e.g. `2ch` requires `guidance_scale=1.0`; nonzero measurably breaks it), not defaults
picked for convenience. See `docs/models/wind.md` §8 for why.

## What should not be placed here

Anything `enerdatasynth`-specific — this package stays meaningful on its own.
Resource-specific glue belongs in `generators/wind*.py` instead.
