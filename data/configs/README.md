# `data/configs/`

YAML copies of `RunConfig`'s hyperparameters that diff cleanly in git. Load one with `RunConfig.from_yaml(path)`.

- **`transformer.yaml`**: the recommended configuration (`architecture: transformer`, `T=1000`, `hidden_dim=64`, `n_layers=2`, `n_heads=4`), and the one the shipped default (`transformer_T1000_H64_L2`) was trained from. One thing the file's own comment doesn't say: these are the same *hyperparameters* as the Week 7 Transformer results, but running `enerdatasynth train --config data/configs/pjm_load/transformer.yaml` today trains on the package's current 23-dimensional one-hot conditioning, not the 6-dimensional conditioning those results used (`../../docs/research_status.md` §C and §D). Matching hyperparameters doesn't reproduce the cited results. A fresh run from this config is a new experiment, to be judged against `docs/results.md` on its own.
- **`mlp.yaml`**: the legacy MLP sweep configuration (`T=100`, `hidden_dim=128`, `n_layers=4`). Kept for reproducibility and as a comparison point, not for new work.

## Using one

```python
from enerdatasynth.utils.config import RunConfig
cfg = RunConfig.from_yaml("data/configs/pjm_load/transformer.yaml")
```

Or from the CLI: `enerdatasynth train --config data/configs/pjm_load/transformer.yaml`. Training needs your own processed data (`enerdatasynth preprocess`).

Wind has no YAML configs; [`wind/README.md`](wind/README.md) explains why.
