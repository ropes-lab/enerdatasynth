# `enerdatasynth.utils`

Shared, cross-cutting infrastructure every other subpackage depends on: paths, config,
seeding, device resolution.

## Files

- **`config.py`**:
  - Path constants (`PROJECT_ROOT`, `DATA_ROOT`, `RAW_DIR`, `PROCESSED_DIR`,
    `CHECKPOINT_DIR`, etc.) — `PROJECT_ROOT` resolves from `$ENERDATASYNTH_PROJECT_ROOT`
    or `cwd`, deliberately **not** from `__file__` (that broke non-editable installs —
    see `MIGRATION.md`).
  - `ensure_dirs()` — creates the `data/` scaffold, called explicitly by whichever
    function actually writes into it, not at import time (a library shouldn't mutate
    the filesystem just from being imported).
  - `RunConfig` — the training/generation hyperparameter dataclass, loadable from
    `data/configs/pjm_load/*.yaml` via `RunConfig.from_yaml(...)`.
  - `resolve_device(device=None)`, `set_seed(seed)` — device/RNG helpers used
    throughout training and sampling.
  - `COND_DIM`, `N_MONTHS`, `N_DOW`, `DEFAULT_ARCHITECTURE`, and the other
    package-wide constants referenced throughout `docs/research_status.md`.

## Public API vs. internal

`RunConfig`, `set_seed`, `DEFAULT_ARCHITECTURE` are re-exported at the top level.
Path constants (`CHECKPOINT_DIR`, etc.) are used internally throughout the package but
not re-exported — import `enerdatasynth.utils.config` directly if you need them.

## What should not be changed casually

`COND_DIM` (currently 23) is tied by a runtime assertion in
`generators/load_conditioning.py` to `LOAD_CONDITION_SPEC`'s implied width — changing
one without the other breaks silently at generation time. `PROJECT_ROOT`'s resolution
order (env var, then cwd — never `__file__`) is a deliberate fix for a real packaging
bug; see `MIGRATION.md` before changing it.
