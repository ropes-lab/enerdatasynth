# `data/outputs/pjm_load/`

Artifacts written by PJM load's own CLI and training code:

- **`generated/X_generated_<run_id>.npy`** — one synthetic sample per validation day, shape `(600, 24)`, in normalized units and *uncalibrated*. Written by `enerdatasynth generate`; it lines up one-to-one with `data/processed/pjm_load/X_daily_profiles_val.npy`, which you rebuild from your own data with `enerdatasynth preprocess` and which the in-sample validation plots and metrics compare against. Two are bundled: `transformer_T1000_H64_L2` (the shipped default) and `T100_H256_L4` (the legacy MLP baseline).
- **`logs/<run_id>_training_log.csv`** — per-epoch training logs, one per bundled checkpoint (eight in total).

Both folders are regenerable working output; see the [category README](../README.md). Wind writes no equivalent files — see [`../wind/README.md`](../wind/README.md).
