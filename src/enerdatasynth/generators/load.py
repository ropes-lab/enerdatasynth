"""
src/enerdatasynth/generators/load.py
======================================
`LoadDiffusionModel` — the package's public, production-quality
interface for electric load generation: load a trained checkpoint and
generate synthetic PJM load profiles without needing to know which
architecture, hyperparameters, or research script produced it.

    from enerdatasynth import LoadDiffusionModel

    model = LoadDiffusionModel.from_run_id("transformer_T1000_H64_L2")
    samples = model.sample(z, seed=0)

Internally this wraps checkpoint loading, architecture selection
(`enerdatasynth.models.build_denoiser`), the DDPM sampler
(`enerdatasynth.diffusion.ddpm.DDPM`), and calibration
(`enerdatasynth.evaluation.calibration`, auto-discovered next to the
checkpoint and applied by default) behind one object. It implements
`generators.base.BaseGenerator`, the common contract any other
resource-type generator in this package (e.g. a future wind model)
would also implement.

Not yet built (candidate follow-up, not claimed as done here): a
friendlier `generate(month=7, weekday=True, Tmax=32, ...)` convenience
method that builds the conditioning vector `z` from human-readable
arguments instead of requiring a caller to already have a `(N,
cond_dim)` array. `sample()` below takes `z` directly.
"""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch

from enerdatasynth.utils.config import (
    CHECKPOINT_DIR, X_DIM, COND_DIM,
    DEFAULT_ARCHITECTURE, DEFAULT_T, DEFAULT_HIDDEN_DIM, DEFAULT_N_LAYERS, DEFAULT_N_HEADS,
    set_seed, resolve_device,
)
from enerdatasynth.models import build_denoiser
from enerdatasynth.diffusion.ddpm import DDPM
from enerdatasynth.evaluation.calibration import load_calibration, apply_calibration as _apply_calibration_correction
from enerdatasynth.generators.base import BaseGenerator


class LoadDiffusionModel(BaseGenerator):
    """
    High-level interface for generating synthetic PJM load profiles
    from a trained checkpoint.
    """

    domain = "load"

    def __init__(self, checkpoint_path: str | Path,
                 architecture: str = DEFAULT_ARCHITECTURE,
                 x_dim: int = X_DIM, cond_dim: int = COND_DIM,
                 diffusion_steps: int = DEFAULT_T,
                 hidden_dim: int = DEFAULT_HIDDEN_DIM,
                 n_layers: int = DEFAULT_N_LAYERS,
                 n_heads: int = DEFAULT_N_HEADS,
                 calibration_path: str | Path | None = None,
                 use_calibration: bool = True,
                 device: str | None = None):
        device = resolve_device(device)
        self.device = device
        self.x_dim = x_dim
        self.cond_dim = cond_dim
        self.architecture = architecture

        self.model = build_denoiser(
            architecture, x_dim=x_dim, cond_dim=cond_dim,
            hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
        ).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.ddpm = DDPM(T=diffusion_steps, device=device)

        if calibration_path is None:
            calibration_path = Path(checkpoint_path).parent / "calibration.json"
        self.calibration = load_calibration(calibration_path) if use_calibration else None

    @classmethod
    def from_run_id(cls, run_id: str, checkpoint_root: str | Path = CHECKPOINT_DIR,
                     device: str | None = None, use_calibration: bool = True) -> "LoadDiffusionModel":
        """
        Construct from a run_id produced by `enerdatasynth train` /
        `enerdatasynth sweep`. Reads architecture and hyperparameters
        back out of `result.json` (saved next to the checkpoint by
        `train_one_config`) so callers don't need to know them.

        Checkpoints trained before this field existed (the original
        MLP hyperparameter sweep — see MIGRATION.md) have no
        `architecture` key in their result.json at all. Defaulting a
        missing key to the *current* `DEFAULT_ARCHITECTURE`
        ("transformer") would be wrong for those: every checkpoint
        without this key was trained as an MLP, since that was the
        only architecture that existed at the time. So the fallback
        here is "mlp", not `DEFAULT_ARCHITECTURE` — a deliberate
        historical default, not the current recommended one.
        """
        ckpt_dir = Path(checkpoint_root) / run_id
        result_path = ckpt_dir / "result.json"
        if not result_path.exists():
            raise FileNotFoundError(
                f"No result.json found at {result_path}. Use the main "
                f"constructor directly (with explicit architecture/"
                f"dimensions/hyperparameters) if this checkpoint wasn't "
                f"produced by enerdatasynth.training.train.train_one_config."
            )
        with open(result_path) as f:
            result = json.load(f)
        return cls(
            checkpoint_path=ckpt_dir / "best.pt",
            architecture=result.get("architecture", "mlp"),
            x_dim=result.get("x_dim", X_DIM),
            cond_dim=result.get("cond_dim", COND_DIM),
            diffusion_steps=result["diffusion_steps"],
            hidden_dim=result["hidden_dim"],
            n_layers=result["n_layers"],
            n_heads=result.get("n_heads", DEFAULT_N_HEADS),
            device=device,
            use_calibration=use_calibration,
        )

    def sample(self, z: np.ndarray, seed: int | None = None,
               apply_calibration: bool | None = None,
               batch_size: int = 256) -> np.ndarray:
        """
        Generate one synthetic profile per row of `z`.

        z : (cond_dim,) or (N, cond_dim) conditioning vector(s)

        `seed`, if given, reseeds every RNG immediately before
        sampling for reproducibility — `enerdatasynth.sampling.
        generate.generate_for_run` historically didn't do this (a
        known gap flagged in the Phase 1 audit); this entry point
        fixes it by making it opt-in explicit.

        `apply_calibration` overrides the instance default (applied
        automatically whenever a `calibration.json` was found next to
        the checkpoint); pass False to see raw, uncalibrated output.
        """
        if seed is not None:
            set_seed(seed)

        z = self._ensure_batch(z)
        z_t = torch.tensor(z, device=self.device)

        with torch.no_grad():
            generated = self.ddpm.generate_batch(self.model, z_t, self.x_dim, batch_size=batch_size)
        generated_np = generated.cpu().numpy()

        use_calibration = (self.calibration is not None) if apply_calibration is None else apply_calibration
        if use_calibration and self.calibration is not None:
            mode = self.calibration.get("mode_recommended", "scalar")
            generated_np = _apply_calibration_correction(generated_np, self.calibration, mode=mode)
        return generated_np
