# `data/configs/wind/`

Empty, and not the wrong place to look — wind genuinely has no standalone YAML config files the way [`../pjm_load/`](../README.md)'s `mlp.yaml`/`transformer.yaml` are. Its configuration lives in two places instead, both in code:

- **Model architecture and training hyperparameters**: `enerdatasynth.resources.wind.config.WindDiffusionConfig`, a dataclass, not a YAML-loaded config. Defaults: Transformer denoiser, `hidden_dim=128`, `num_heads=4`, `T=200` diffusion steps, cosine beta schedule (`beta_start=1e-4, beta_end=0.02`) — a different schedule from load's linear default, chosen independently by the upstream project.
- **Per-preset generation-time settings** (sampler, steps, `eta`, guidance scale/scope): `resources/wind/manifest.json`'s `recommended_generation` block, one entry per preset (`1ch`/`2ch`/`3ch`). These genuinely differ by preset — `2ch` needs `eta=0.0`, `1ch`/`3ch` want `eta=1.0` — so a single shared config wouldn't even be correct if one existed.

There's also no CLI training entry point for wind (`enerdatasynth train` is `pjm_load`-only — see [`docs/models/wind.md`](../../../docs/models/wind.md) §14), so there's no config-loading code path that would ever read a YAML file from this folder in the first place. Exists as the resource-nested sibling of `../pjm_load/`, for structural consistency.
