# `data/processed/`

Cleaned, model-ready tensors, one folder per resource:

- [`pjm_load/`](pjm_load/README.md) — bundled, small (~1.3MB), required for `enerdatasynth.model("pjm_load", ...)` to construct.
- `pjm_load_backup/` — a frozen backup copy of `pjm_load/`, same contents. Not resource-nested further since it's PJM-load-specific by nature (a backup of a preprocessing run); wind has no equivalent since it has no preprocessing pipeline to back up.
- [`wind/`](wind/README.md) — placeholder; wind has no preprocessing pipeline in this package.
