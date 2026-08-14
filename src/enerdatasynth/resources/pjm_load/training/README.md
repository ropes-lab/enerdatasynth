# `enerdatasynth.resources.pjm_load.training` and `enerdatasynth.resources.pjm_load.sampling`

**PJM load's from-scratch training/generation workflow specifically.** Wind ships
pre-trained, bundled checkpoints and has no training code in this package at all — its
original training code lives in the upstream `wind_diffusion` project, not here (see
[`docs/models/wind.md`](../../../../../docs/models/wind.md) §6/§14). Both resources
*generate* the same way regardless (`enerdatasynth.model(...).sample(...)`) — only
training is load-specific.

Grouped together here because they're used together: train a checkpoint, then generate
from it.

## Files

- **`training/train.py`** — `train_one_config(config)`: single-configuration training
  loop. AdamW, standard diffusion MSE loss only, best+latest checkpoint saving, per-epoch
  CSV logs, loss-curve plot, optional early stopping (patience=10).
- **`training/sweep.py`** — `run_sweep()`: a 27-configuration grid (diffusion steps ×
  hidden dim × layers) for whichever architecture `RunConfig.architecture` is set to.
  **Read the module docstring's own caveat**: this grid was tuned against MLP
  hyperparameters and has not been re-swept for the Transformer — the current
  Transformer defaults are the validated-in-research values, not the winner of a
  from-scratch sweep at this architecture.
- **`sampling/generate.py`** — `generate_for_run(...)`: generates one synthetic sample
  per real condition vector in a dataset split, required for the in-sample validation
  methodology this project uses throughout (`docs/results.md` notes this explicitly).

## How to run it

```bash
enerdatasynth train --architecture transformer   # single config
enerdatasynth sweep                              # full grid
enerdatasynth generate --diffusion-steps 1000 --hidden-dim 64 --n-layers 2
```

## What a researcher should look at first

If the goal is closing the gap flagged in `docs/research_status.md` §D — no Transformer
checkpoint compatible with the current 23-dim conditioning exists — this is where that
training run would start: `enerdatasynth train` (defaults to `architecture="transformer"`
already), then `enerdatasynth calibrate --run-id <new-run-id>`.

## Assumptions this makes

`generate_for_run`'s historical version didn't reseed RNGs before sampling (a gap noted
in `generators/load.py`'s own docstring, fixed at the `LoadDiffusionModel.sample()`
layer, not here) — pass `seed=` explicitly at that layer for reproducible generation.
