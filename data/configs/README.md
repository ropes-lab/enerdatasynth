# `data/configs/`

Diffable YAML mirrors of `RunConfig`'s hyperparameters, loadable via `RunConfig.from_yaml(path)`.

- **`transformer.yaml`** — the current recommended configuration (`architecture: transformer`, `T=1000`, `hidden_dim=64`, `n_layers=2`, `n_heads=4`). This is the config the shipped default (`transformer_T1000_H64_L2`) was trained from. One nuance the file's own comment doesn't mention: these are the same *hyperparameters* used to produce the Week 7 Transformer results, but running `enerdatasynth train --config data/configs/pjm_load/transformer.yaml` today trains against the package's current 23-dimensional one-hot conditioning, not the 6-dimensional research conditioning those results actually used (see `../../docs/research_status.md` §C/§D). Matching hyperparameters isn't the same as reproducing the cited results — a fresh run from this config would be a new experiment, judged on its own merits against `docs/results.md`, not an automatic reproduction of the old one.
- **`mlp.yaml`** — the legacy MLP sweep configuration (`T=100`, `hidden_dim=128`, `n_layers=4`). Kept for reproducibility and as a comparison point; not recommended for new work.

## How to use

```python
from enerdatasynth.utils.config import RunConfig
cfg = RunConfig.from_yaml("data/configs/pjm_load/transformer.yaml")
```

or via the CLI: `enerdatasynth train --config data/configs/pjm_load/transformer.yaml`.

Wind has no YAML configs of its own — see [`wind/README.md`](wind/README.md) for why.
