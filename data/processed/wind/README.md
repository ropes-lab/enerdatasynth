# `data/processed/wind/`

Empty. [`../pjm_load/`](../pjm_load/README.md) is where `enerdatasynth preprocess` writes
its arrays and `metadata.json`. Wind has no preprocessing step in this package; it ships
trained checkpoints (`data/checkpoints/wind/`) instead.

The upstream training command appears in error messages from
`resources/wind/registry.py`: `wind-diffusion-train --preset <preset> --data-path <csv> --output-dir <dir>`.
This package doesn't install that command, so you can't run it here
([`docs/models/wind.md`](../../../docs/models/wind.md) §14).

If a wind preprocessing pipeline is ever added, its output tensors go here.
