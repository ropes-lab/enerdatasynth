# `src/wind_diffusion/`

A self-contained conditional-diffusion wind-profile generator, vendored from a separate
Data-For-Good research project — not written for or by this repository. `enerdatasynth`
wraps it (`enerdatasynth.generators.WindDiffusionModel`,
`enerdatasynth.model("wind", ...)`); it does not depend on `enerdatasynth` at all and
has its own independent public API, versioning (`__version__` in `__init__.py`), and
checkpoint-resolution mechanism.

See [`docs/models/wind.md`](../../docs/models/wind.md) for the full scientific report
on what this actually produces and how well-validated it is — this file is about the
code, not the science.

## Files

| File | Purpose |
|---|---|
| `api.py` | Public entry points: `list_models()`, `describe_model()`, `generate_wind_profile()` |
| `config.py` | `WindDiffusionConfig` — architecture/diffusion/conditioning hyperparameters per preset (`1ch`/`2ch`/`3ch`) |
| `models.py` | The Transformer-style denoiser |
| `diffusion.py` | Forward/reverse diffusion process (`DiffusionSchedule`) |
| `samplers.py` | DDIM sampling with classifier-free guidance and dynamic thresholding |
| `data.py` | Raw-CSV loading, channel derivation, daily windowing — needs real data this repository doesn't ship (see `docs/models/wind.md` §2) |
| `transforms.py` | Feature engineering (wind-regime bucketing, ordinal/onehot regime encoding) |
| `evaluate.py` | `generate_multiple_for_condition`, `build_condition_vector` — what `enerdatasynth`'s wind bridge actually calls |
| `metrics.py` | The fidelity-gate metric suite (`marginal_wasserstein`, `acf_rmse`, `regime_accuracy`, etc.) — see `docs/models/wind.md` §7 |
| `registry.py` | `resolve_weights()` — checkpoint resolution (explicit path → env dir → manifest download → error). **`enerdatasynth`'s own bundled-checkpoint convention (`data/checkpoints/wind/`) is checked before this ever runs** — see `enerdatasynth/generators/wind_conditioning.py` |
| `manifest.json` | Bundled index of preset metadata/metrics/recommended generation settings. Its `status: "pending"` fields describe *this package's own* (unused, in this repo) download mechanism, not whether wind works here |
| `exceptions.py` | `CheckpointNotFoundError` and friends |
| `train.py` | Training entry point for the *upstream* project — not exposed as a console script by this repository's `pyproject.toml` (see `docs/models/wind.md` §14) |
| `plotting.py` | Real-vs-synthetic comparison plots — needs real data this repository doesn't have |
| `utils.py` | Seeding and small shared helpers |

## Public API vs. internal

`api.py`'s three functions are the intended entry point for using this package
standalone. `enerdatasynth` instead calls `evaluate.py`/`registry.py`/`config.py`
directly at a lower level (`enerdatasynth/generators/wind.py`) — both are valid, `api.py`
is simpler if you don't need `enerdatasynth`'s conditioning-vector conveniences.

## What should not be changed casually

Any of the per-preset `recommended_generation` settings in `manifest.json` — these are
measured-best values (e.g. `2ch` requires `guidance_scale=1.0`; a nonzero value
measurably breaks it even with dynamic thresholding on), not defaults chosen for
convenience. See `docs/models/wind.md` §8 for why each one is what it is.

## What should not be placed here

Anything `enerdatasynth`-specific — this package is meant to remain buildable and
meaningful independent of `enerdatasynth`. Resource-specific glue belongs in
`src/enerdatasynth/generators/wind*.py` instead.
