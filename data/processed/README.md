# `data/processed/`

Cleaned, model-ready arrays, built by `enerdatasynth preprocess` from your own copy of the source files. They are not part of this repository and git ignores them.

- [`pjm_load/`](pjm_load/README.md) — what `preprocess` produces. The one tracked file is `metadata.json`, the scalers and condition spec the shipped checkpoint needs to build model inputs and convert outputs back to MW.
- [`wind/`](wind/README.md) — empty; wind has no preprocessing pipeline in this package.

If you run `enerdatasynth preprocess`, it also keeps a backup copy of its output in `pjm_load_backup/`, which is ignored the same way.
