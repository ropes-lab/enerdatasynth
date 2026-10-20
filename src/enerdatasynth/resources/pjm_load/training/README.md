# `enerdatasynth.resources.pjm_load.training` and `sampling`

The from-scratch training and generation workflow for PJM load. Wind ships pre-trained checkpoints and has no training code in this package; its training code lives upstream in the `wind_diffusion` project ([`docs/models/wind.md`](../../../../../docs/models/wind.md) §6 and §14). Both resources generate the same way (`enerdatasynth.model(...).sample(...)`). Only training is specific to load.

The two folders are covered together because you use them together: train a checkpoint, then generate from it.

## Files

- **`training/train.py`**: `train_one_config(config)`, a single-configuration training loop. AdamW, plain diffusion MSE loss, best and latest checkpoints, per-epoch CSV logs, a loss-curve plot, and optional early stopping (patience 10).
- **`training/sweep.py`**: `run_sweep()`, a 27-configuration grid over diffusion steps, hidden size and layers for whatever `RunConfig.architecture` is set to. The module docstring has a caveat worth reading: the grid was tuned for the MLP and hasn't been re-run for the Transformer, so the Transformer defaults are the values validated in research, not the winner of a fresh sweep.
- **`sampling/generate.py`**: `generate_for_run(...)` makes one synthetic sample for every real condition vector in a dataset split. The in-sample validation used throughout this project needs that one-to-one match (`docs/results.md` explains).

## Running it

```bash
enerdatasynth train --architecture transformer   # single config
enerdatasynth sweep                              # full grid
enerdatasynth generate --diffusion-steps 1000 --hidden-dim 64 --n-layers 2
```

All three read the processed arrays, which the repository doesn't include. Run `enerdatasynth preprocess` on your own copy of the PJM and NOAA data first.

## Where to start

The shipped default came from exactly this workflow: `enerdatasynth train --config data/configs/pjm_load/transformer.yaml`, then a calibration fit (`enerdatasynth calibrate --run-id <new-run-id> --seed 42` with the same architecture flags). `docs/research_status.md` §D has the history. The open item is a from-scratch sweep for the Transformer.

## Reproducibility

The original `generate_for_run` didn't reseed the random generators before sampling. `generators/load.py` notes this in its docstring, and the fix went into `LoadDiffusionModel.sample()` rather than here. Pass `seed=` at that level if you want reproducible output.
