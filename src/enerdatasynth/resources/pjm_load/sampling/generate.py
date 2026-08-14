"""
src/enerdatasynth/sampling/generate.py
=========================================
After training: generate one synthetic sample for every real
condition vector in the validation (or training) set.

    for each z_i:
        x_hat_i = generate(z_i)

The generated dataset contains exactly the same number of samples
as the real dataset it was conditioned on, which is required for
in-sample validation.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from enerdatasynth.utils.config import (
    PROCESSED_DIR, GENERATED_DIR, CHECKPOINT_DIR,
    DEFAULT_ARCHITECTURE, DEFAULT_N_HEADS, ensure_dirs, resolve_device,
)
from enerdatasynth.resources.pjm_load.models import build_denoiser
from enerdatasynth.resources.pjm_load.diffusion.ddpm import DDPM


def generate_for_run(run_id: str, diffusion_steps: int, hidden_dim: int,
                      n_layers: int, architecture: str = DEFAULT_ARCHITECTURE,
                      n_heads: int = DEFAULT_N_HEADS, use_val: bool = True,
                      seed: int | None = None, device: str | None = None) -> Path:
    """
    Load the best checkpoint for `run_id` and generate one synthetic
    profile per real condition vector (using the validation set by
    default, since that's what we report on).

    `seed`, if given, reseeds every RNG immediately before sampling so
    the result is reproducible; if omitted, sampling uses whatever
    global RNG state the process is already in (matches prior
    behavior — this was a known reproducibility gap, now opt-in fixed
    rather than silently changed for existing callers).
    """
    ensure_dirs()
    device = resolve_device(device)
    if seed is not None:
        from enerdatasynth.utils.config import set_seed
        set_seed(seed)

    Z = np.load(PROCESSED_DIR / ("Z_conditions_val.npy" if use_val
                                  else "Z_conditions.npy"))
    X_real = np.load(PROCESSED_DIR / ("X_daily_profiles_val.npy" if use_val
                                       else "X_daily_profiles.npy"))

    x_dim    = X_real.shape[1]
    cond_dim = Z.shape[1]

    model = build_denoiser(
        architecture, x_dim=x_dim, cond_dim=cond_dim,
        hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
    ).to(device)
    ckpt_path = CHECKPOINT_DIR / run_id / "best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    ddpm = DDPM(T=diffusion_steps, device=device)
    Z_t  = torch.tensor(Z, dtype=torch.float32, device=device)

    print(f"Generating {len(Z)} synthetic samples for {run_id} "
          f"(T={diffusion_steps})...")
    X_gen = ddpm.generate_batch(model, Z_t, x_dim, batch_size=128)
    X_gen_np = X_gen.cpu().numpy()

    out_path = GENERATED_DIR / f"X_generated_{run_id}.npy"
    np.save(out_path, X_gen_np)
    print(f"Saved: {out_path}  shape={X_gen_np.shape}")

    # Sanity check: same sample count as real dataset
    assert X_gen_np.shape[0] == X_real.shape[0], (
        f"Generated dataset has {X_gen_np.shape[0]} samples, "
        f"expected {X_real.shape[0]} to match the real dataset."
    )
    return out_path


if __name__ == "__main__":
    from enerdatasynth.utils.config import RunConfig

    cfg = RunConfig()
    generate_for_run(
        cfg.run_id, cfg.diffusion_steps, cfg.hidden_dim, cfg.n_layers,
        architecture=cfg.architecture, n_heads=cfg.n_heads,
    )
