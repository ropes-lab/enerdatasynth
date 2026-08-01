"""
examples/generate_samples.py
==============================
Minimal demonstration: load an already-trained checkpoint and
generate synthetic load profiles for the validation set's condition
vectors.

This calls the same lower-level function `scripts/run_all.py` uses
internally (`powerdiffusion.sampling.generate.generate_for_run`). A
friendlier, higher-level API is planned for a later phase — see the
project roadmap — but isn't built yet, so this example shows the
pipeline as it actually works today rather than a future interface.

Requires a trained checkpoint at checkpoints/<run_id>/best.pt,
produced by `scripts/run_all.py` or `powerdiffusion.training.train`.
"""

from __future__ import annotations

from powerdiffusion.utils.config import RunConfig
from powerdiffusion.sampling.generate import generate_for_run

if __name__ == "__main__":
    cfg = RunConfig()  # T=100, hidden=128, layers=4 — the pipeline defaults
    out_path = generate_for_run(
        run_id=cfg.run_id,
        diffusion_steps=cfg.diffusion_steps,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
    )
    print(f"Synthetic profiles written to: {out_path}")
