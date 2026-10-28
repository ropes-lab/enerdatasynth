"""
examples/pjm_load/generate_samples.py
==============================
Minimal demonstration: load an already-trained checkpoint and
generate synthetic load profiles using the public `LoadDiffusionModel`
API, which handles architecture selection, checkpoint loading, and
calibration (auto-applied if `calibration.json` exists next to the
checkpoint) behind one object.

Requires a trained checkpoint at checkpoints/<run_id>/best.pt with a
result.json next to it, produced by `enerdatasynth train` /
`enerdatasynth sweep` / `enerdatasynth.resources.pjm_load.training.train.train_one_config`.

Reproduce:
    1. From a clone of this repository: `pip install -e ".[research]"`.
    2. Get the PJM and NOAA source files yourself (sources: data/README.md) and put
       them in data/raw/pjm_load/. The repository ships no real data.
    3. `enerdatasynth preprocess`
    4. `python examples/pjm_load/generate_samples.py`
    Output: printed (shape of the generated profiles). The trained checkpoint ships with the package; the real
    validation condition vectors it uses as input come from step 3.
"""

from __future__ import annotations
import numpy as np

from enerdatasynth import LoadDiffusionModel, RunConfig
from enerdatasynth.utils.config import load_processed

if __name__ == "__main__":
    cfg = RunConfig()  # default: transformer, T=1000, model_dim=64, layers=2, heads=4
    model = LoadDiffusionModel.from_run_id(cfg.run_id)

    # Use the validation set's real condition vectors as an example —
    # any (N, cond_dim) array built by enerdatasynth.resources.pjm_load.data.preprocess
    # works here.
    Z_val = load_processed("Z_conditions_val.npy")
    samples = model.sample(Z_val[:10], seed=0)
    print(f"Generated {samples.shape[0]} synthetic profiles, shape={samples.shape}")

    # Equivalent lower-level call, if you don't want the checkpoint
    # metadata auto-discovery LoadDiffusionModel does:
    #
    # from enerdatasynth.resources.pjm_load.sampling.generate import generate_for_run
    # generate_for_run(cfg.run_id, cfg.diffusion_steps, cfg.hidden_dim,
    #                   cfg.n_layers, architecture=cfg.architecture, n_heads=cfg.n_heads)
