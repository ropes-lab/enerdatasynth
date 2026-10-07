# `enerdatasynth.resources.pjm_load.training` and `enerdatasynth.resources.pjm_load.sampling`

PJM load's from-scratch training and generation workflow, specifically. Wind ships pre-trained, bundled checkpoints and has no training code in this package at all — its original training code lives upstream, in the `wind_diffusion` project (see [`docs/models/wind.md`](../../../../../docs/models/wind.md) §6/§14). Both resources generate the same way regardless (`enerdatasynth.model(...).sample(...)`); only training is load-specific.

These two folders are grouped together here because they're used together: train a checkpoint, then generate from it.

## Files

- **`training/train.py`** — `train_one_config(config)`, a single-configuration training loop. AdamW, standard diffusion MSE loss, best-and-latest checkpoint saving, per-epoch CSV logs, a loss-curve plot, and optional early stopping (patience 10).
- **`training/sweep.py`** — `run_sweep()`, a 27-configuration grid over diffusion steps, hidden dim, and layers, for whichever architecture `RunConfig.architecture` is set to. Worth reading the module docstring's own caveat here: this grid was tuned against MLP hyperparameters and hasn't been re-swept for the Transformer, so the current Transformer defaults are the validated-in-research values, not the winner of a from-scratch sweep at this architecture.
- **`sampling/generate.py`** — `generate_for_run(...)` generates one synthetic sample per real condition vector in a dataset split, which the in-sample validation methodology used throughout this project requires (`docs/results.md` spells this out).

## How to run it

```bash
enerdatasynth train --architecture transformer   # single config
enerdatasynth sweep                              # full grid
enerdatasynth generate --diffusion-steps 1000 --hidden-dim 64 --n-layers 2
```

## What a researcher should look at first

The shipped default was produced with exactly this workflow: `enerdatasynth train --config data/configs/pjm_load/transformer.yaml`, then a calibration fit (`enerdatasynth calibrate --run-id <new-run-id> --seed 42 ...` with the same architecture flags). `docs/research_status.md` §D has the history. The open item is a from-scratch hyperparameter sweep for the Transformer; the current values are the research-validated ones.

## Assumptions this makes

The historical version of `generate_for_run` didn't reseed RNGs before sampling — a gap noted in `generators/load.py`'s own docstring, and fixed at the `LoadDiffusionModel.sample()` layer rather than here. Pass `seed=` explicitly at that layer if you want reproducible generation.
