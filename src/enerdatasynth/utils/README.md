# `enerdatasynth.utils`

Shared infrastructure that the rest of the package depends on: paths, config, seeding and device selection.

## Files

**`config.py`** holds:

- The path constants (`PROJECT_ROOT`, `DATA_ROOT`, `RAW_DIR`, `PROCESSED_DIR`, `CHECKPOINT_DIR` and so on). `PROJECT_ROOT` comes from `$ENERDATASYNTH_PROJECT_ROOT` or the current working directory. It deliberately does not use `__file__`, because that broke non-editable installs (see `MIGRATION.md`).
- `ensure_dirs()`, which creates the `data/` folders. Whatever function writes into them calls it explicitly; it is never run on import, since a library shouldn't touch the filesystem just by being imported.
- `RunConfig`, the training and generation hyperparameters as a dataclass. `RunConfig.from_yaml(...)` loads it from `data/configs/pjm_load/*.yaml`.
- `processed_path()` and `load_processed()`, which load files from `data/processed/pjm_load/`. If the file is missing they raise an error that says the repository ships no real data and points to where to get it.
- `resolve_device(device=None)` and `set_seed(seed)`, used throughout training and sampling.
- `COND_DIM`, `N_MONTHS`, `N_DOW`, `DEFAULT_ARCHITECTURE` and the other package-wide constants that `docs/research_status.md` refers to.

## Public API

`RunConfig`, `set_seed` and `DEFAULT_ARCHITECTURE` are re-exported at the top level. The path constants like `CHECKPOINT_DIR` are used all over the package but aren't re-exported; import `enerdatasynth.utils.config` if you need them.

## Be careful with

`COND_DIM` (currently 23) is tied by a runtime assertion in `generators/load_conditioning.py` to the width implied by `LOAD_CONDITION_SPEC`. Change one without the other and generation breaks with a shape mismatch. The order `PROJECT_ROOT` is resolved in (env var, then cwd, never `__file__`) fixes a real packaging bug, so read `MIGRATION.md` before changing it.
