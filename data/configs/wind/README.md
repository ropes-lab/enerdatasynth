# `data/configs/wind/`

Empty. Unlike [`../pjm_load/`](../README.md), which has `mlp.yaml` and
`transformer.yaml`, wind has no YAML config files. Its settings are in code:

- **Model and training settings:** the `WindDiffusionConfig` dataclass in
  `enerdatasynth.resources.wind.config`. Defaults: transformer denoiser,
  `hidden_dim=128`, `num_heads=4`, `T=200` diffusion steps, cosine beta schedule
  (`beta_start=1e-4, beta_end=0.02`).
- **Generation settings for each preset** (sampler, steps, `eta`, guidance): the
  `recommended_generation` block for each preset in `resources/wind/manifest.json`.
  These differ by preset (`2ch` needs `eta=0.0`; `1ch` and `3ch` use `eta=1.0`), so one
  shared config file wouldn't work.

`enerdatasynth train` only supports `pjm_load`
([`docs/models/wind.md`](../../../docs/models/wind.md) §14), so nothing would read a
config file from this folder anyway.
