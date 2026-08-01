"""
examples/generate_samples.py
==============================
Minimal demonstration: load an already-trained checkpoint and
generate synthetic load profiles using the public `LoadDiffusionModel`
API, which handles architecture selection, checkpoint loading, and
calibration (auto-applied if `calibration.json` exists next to the
checkpoint) behind one object.

Requires a trained checkpoint at checkpoints/<run_id>/best.pt with a
result.json next to it, produced by `pjm_diffusion train` /
`pjm_diffusion sweep` / `pjm_diffusion.training.train.train_one_config`.
"""

from __future__ import annotations
import numpy as np

from pjm_diffusion import LoadDiffusionModel, RunConfig
from pjm_diffusion.utils.config import PROCESSED_DIR

if __name__ == "__main__":
    cfg = RunConfig()  # default: transformer, T=1000, model_dim=64, layers=2, heads=4
    model = LoadDiffusionModel.from_run_id(cfg.run_id)

    # Use the validation set's real condition vectors as an example —
    # any (N, cond_dim) array built by pjm_diffusion.data.preprocess
    # works here.
    Z_val = np.load(PROCESSED_DIR / "Z_conditions_val.npy")
    samples = model.sample(Z_val[:10], seed=0)
    print(f"Generated {samples.shape[0]} synthetic profiles, shape={samples.shape}")

    # Equivalent lower-level call, if you don't want the checkpoint
    # metadata auto-discovery LoadDiffusionModel does:
    #
    # from pjm_diffusion.sampling.generate import generate_for_run
    # generate_for_run(cfg.run_id, cfg.diffusion_steps, cfg.hidden_dim,
    #                   cfg.n_layers, architecture=cfg.architecture, n_heads=cfg.n_heads)
