# Training

**This page covers training new PJM load checkpoints specifically** — wind ships
pre-trained, bundled checkpoints and has no from-scratch training workflow in this
package (see [`docs/models/wind.md`](models/wind.md) §6/§14). For generating samples
from an already-trained model (either resource), see [`docs/quickstart.md`](quickstart.md)
and [`docs/api_reference.md`](api_reference.md) instead.

## Prerequisites

```bash
enerdatasynth preprocess
```

Builds `data/processed/pjm_load/{X_daily_profiles,Z_conditions}[_val].npy` and `metadata.json` from raw CSVs (`data/raw/pjm_load/`). Only needs to be re-run when raw data changes.

## Training a single configuration

```bash
enerdatasynth train                                    # transformer, validated defaults
enerdatasynth train --config data/configs/pjm_load/transformer.yaml   # equivalent, explicit
enerdatasynth train --config data/configs/pjm_load/mlp.yaml            # legacy MLP baseline
enerdatasynth train --architecture transformer --diffusion-steps 500 --hidden-dim 128 --n-layers 3
```

Writes `data/checkpoints/pjm_load/<run_id>/{best.pt,latest.pt,loss_curve.png,result.json}` and `data/outputs/pjm_load/logs/<run_id>_training_log.csv`. `run_id` is `{architecture}_T{diffusion_steps}_H{hidden_dim}_L{n_layers}`.

`result.json` records everything needed to reload the checkpoint later (`architecture`, `x_dim`, `cond_dim`, `hidden_dim`, `n_layers`, `n_heads`, `diffusion_steps`) — this is what `LoadDiffusionModel.from_run_id()` and the fluent SDK's checkpoint auto-discovery read.

## Configs

`RunConfig.from_yaml(path)` / `.to_yaml(path)` (see `src/enerdatasynth/utils/config.py`) round-trip a training configuration to/from YAML. Unknown keys raise rather than being silently ignored. Two are shipped:

- `data/configs/pjm_load/transformer.yaml` — the validated defaults (mirrors `RunConfig()`'s built-in defaults).
- `data/configs/pjm_load/mlp.yaml` — the legacy MLP baseline's original hyperparameters.

Write your own for a new configuration you want to reproduce exactly; `cfg.to_yaml(path)` after constructing a `RunConfig` snapshots whatever you used.

## Sweeping

```bash
enerdatasynth sweep                    # one-factor-at-a-time, 9 runs
enerdatasynth sweep --full-grid        # full 3x3x3 = 27-run grid
enerdatasynth sweep --epochs 100       # override epochs per config
```

**Caveat**: the sweep grid (`SWEEP_DIFFUSION_STEPS`, `SWEEP_HIDDEN_DIM`, `SWEEP_N_LAYERS` in `utils/config.py`) was originally tuned while comparing MLP hyperparameters (`data/research_history/pjm_load/`) and has not itself been re-swept for the transformer architecture. Running it against the current default architecture is a reasonable, currently-unstarted next step — see the README's Roadmap.

## Calibration

```bash
enerdatasynth calibrate --run-id transformer_T1000_H64_L2
```

Fits the scalar bias correction (see [`docs/evaluation.md`](evaluation.md)) and saves `data/checkpoints/pjm_load/<run_id>/calibration.json`. Applied automatically by `LoadDiffusionModel`/the fluent SDK once it exists — no code changes needed to start using it.

## Generating an in-sample validation dataset

```bash
enerdatasynth generate --diffusion-steps 1000 --hidden-dim 64 --n-layers 2
```

One synthetic sample per real validation-set condition vector — used by `enerdatasynth evaluate` and `examples/run_benchmark.py`.

## Reproducibility

Every training run seeds `random`/`numpy`/`torch` via `RunConfig.seed` (default 42, `set_seed()`). `LoadDiffusionModel.sample()` / `ResourceModel.sample()` accept an optional `seed` for reproducible generation — a gap in the original pipeline (`generate_for_run` didn't reseed) that's now fixed as an opt-in parameter.

## Everything at once

```bash
enerdatasynth pipeline   # preprocess -> sweep -> generate -> calibrate -> validate -> report
```

Equivalent to `examples/run_all.py`.
