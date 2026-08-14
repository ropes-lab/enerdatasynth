# `data/outputs/`

Where the package writes its own generated artifacts — as opposed to `data/checkpoints/pjm_load/`
(trained models) or `data/processed/pjm_load/` (cleaned training data).

- **`generated/`** — synthetic samples written by `enerdatasynth.resources.pjm_load.sampling.generate.generate_for_run`
  (backs `enerdatasynth generate`).
- **`logs/`** — per-epoch training CSV logs, one per run (`<run_id>_training_log.csv`),
  written by `enerdatasynth.resources.pjm_load.training.train.train_one_config`.

Both are created on demand by `ensure_dirs()` (`src/enerdatasynth/utils/config.py`) —
empty here is the normal starting state, not a missing-file problem.

## What should not be placed here

Anything you want to keep across a `preprocess`/`train`/`sweep` re-run without checking
first — these folders are treated as regenerable working output, not permanent storage.

Wind writes no equivalent output files — see [`wind/README.md`](wind/README.md) for why.
