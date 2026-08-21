# `data/processed/wind/`

Empty. Unlike [`../pjm_load/`](../pjm_load/README.md) — which holds real `X_daily_profiles.npy`/`Z_conditions.npy` tensors and a `metadata.json`, produced by `enerdatasynth preprocess` — wind has no from-scratch preprocessing pipeline in this package at all. It ships pre-trained checkpoints (`data/checkpoints/wind/`) instead of a raw-CSV-to-tensor path, so there's nothing here to produce or hold.

The one hint of what upstream preprocessing looked like: `resources/wind/registry.py`'s own error messages reference a `wind-diffusion-train --preset <preset> --data-path <csv> --output-dir <dir>` command — but that command is **not registered as an installable console script** by this package's `pyproject.toml`. It describes the *original* `wind_diffusion` project's training invocation, not something you can actually run from this repository. See [`docs/models/wind.md`](../../../docs/models/wind.md) §14 for the full explanation.

Would hold wind's cleaned/derived tensors if a from-scratch preprocessing pipeline is ever built here.
