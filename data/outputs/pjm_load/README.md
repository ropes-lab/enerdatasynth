# `data/outputs/pjm_load/`

Artifacts written by PJM load's own CLI and training code:

- **`generated/X_generated_T100_H256_L4.npy`** — one synthetic sample per validation day for the best bundled run, shape `(600, 24)`, in normalized units. Written by `enerdatasynth generate`; it lines up one-to-one with `data/processed/pjm_load/X_daily_profiles_val.npy`, which is what the in-sample validation plots and metrics compare against.
- **`logs/<run_id>_training_log.csv`** — per-epoch training logs, one per bundled checkpoint (seven in total).

Both folders are regenerable working output; see the [category README](../README.md). Wind writes no equivalent files — see [`../wind/README.md`](../wind/README.md).
