# `data/configs/wind/`

Empty. Wind's per-preset configuration lives in code
(`enerdatasynth.resources.wind.config.WindDiffusionConfig` / `manifest.json`'s
`recommended_generation`), not as standalone YAML files the way PJM load's
[`../pjm_load/`](../README.md) configs are. Exists as the resource-nested
sibling for structural consistency — see
[`docs/models/wind.md`](../../../docs/models/wind.md) §5/§8 for wind's actual
configuration.
