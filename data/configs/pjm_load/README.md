# `data/configs/pjm_load/`

YAML mirrors of `RunConfig`'s hyperparameters for PJM load:

- **`transformer.yaml`** — the recommended configuration: `architecture: transformer`, `diffusion_steps: 1000`, `hidden_dim: 64`, `n_layers: 2`, `n_heads: 4`, `lr: 3e-4`, `batch_size: 64`, up to 400 epochs with patience 40, `seed: 42`.
- **`mlp.yaml`** — the legacy MLP sweep configuration: `diffusion_steps: 100`, `hidden_dim: 128`, `n_layers: 4`, up to 200 epochs with patience 10.

Load either with `RunConfig.from_yaml("data/configs/pjm_load/<file>.yaml")` or `enerdatasynth train --config data/configs/pjm_load/<file>.yaml`.

One caveat on reproducing the cited Transformer results from `transformer.yaml` is in the [category README](../README.md). Wind's equivalent lives in code, not YAML — see [`../wind/README.md`](../wind/README.md).
