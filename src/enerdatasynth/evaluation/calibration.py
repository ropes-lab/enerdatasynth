"""
src/enerdatasynth/evaluation/calibration.py
===============================================
Post-processing bias correction for generated profiles.

The trained denoiser reproduces daily SHAPE well but carries a small,
systematic per-hour level offset; summed over 24 hours that offset
compounds into a large daily-energy error even though every shape
metric (first-difference, ramp, ACF) is unaffected by a constant
shift. This was diagnosed and fixed in
research_history/scripts/week7_calibrate.py: measuring the offset on
the TRAINING set (never validation) and adding it back at generation
time cut energy error from 0.1644 to 0.0563 (66%) at zero cost to
shape metrics — see research_history/reports/week7/calibration_report.md.

Two correction modes were compared there: a single scalar offset, and
a 24-value per-hour vector. The per-hour variant is the more elaborate
of the two but was NOT the better one — it degraded ACF (0.0317 ->
0.0501) and RMSE (0.0210 -> 0.0249) relative to no calibration at all,
while the scalar variant matched or improved every metric. The
original research script's own output explicitly recorded
`"mode_recommended": "scalar"`. This module defaults to scalar for
that reason; per-hour is provided for completeness, not because it's
the validated choice.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch

CalibrationMode = Literal["scalar", "per_hour"]


def compute_bias(real: np.ndarray, generated: np.ndarray) -> dict:
    """
    Per-hour and scalar bias (real - generated), averaged over all
    rows. Positive values mean the generator sits low and this amount
    should be ADDED back to correct it.
    """
    per_hour = (real.mean(axis=0) - generated.mean(axis=0)).astype(float)
    scalar = float(real.mean() - generated.mean())
    return {"scalar": scalar, "per_hour": per_hour.tolist()}


def calibrate_from_samples(model: torch.nn.Module, ddpm, X_train: np.ndarray,
                            Z_train: np.ndarray, n_fit: int = 1200,
                            seed: int | None = None, device: str = "cpu") -> dict:
    """
    Reproduce the validated calibration procedure: sample a subset of
    the TRAINING set (never validation — that would fit to the test
    data) from the given model/ddpm, and measure the resulting bias.
    """
    if seed is not None:
        from enerdatasynth.utils.config import set_seed
        set_seed(seed)

    n_fit = min(n_fit, len(X_train))
    idx = np.random.choice(len(X_train), n_fit, replace=False)
    Z_t = torch.tensor(Z_train[idx], dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        generated = ddpm.generate_batch(model, Z_t, X_train.shape[1]).cpu().numpy()

    bias = compute_bias(X_train[idx], generated)
    bias.update({
        "mode_recommended": "scalar",
        "fitted_on": "training set",
        "n_days_used": n_fit,
    })
    return bias


def apply_calibration(generated: np.ndarray, calibration: dict,
                       mode: CalibrationMode = "scalar") -> np.ndarray:
    """Add the stored bias correction back onto generated profiles."""
    if mode == "scalar":
        return generated + calibration["scalar"]
    if mode == "per_hour":
        return generated + np.asarray(calibration["per_hour"])[None, :]
    raise ValueError(f"Unknown calibration mode: {mode!r} (expected 'scalar' or 'per_hour')")


def save_calibration(path: Path, calibration: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(calibration, f, indent=2)


def load_calibration(path: Path) -> dict | None:
    """
    Returns None (rather than raising) if no calibration file exists
    at `path` — callers should treat calibration as optional: a
    checkpoint that hasn't been calibrated yet is still usable.
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def calibrate_checkpoint(run_id: str, architecture: str, diffusion_steps: int,
                          hidden_dim: int, n_layers: int, n_heads: int = 4,
                          n_fit: int = 1200, seed: int | None = None) -> dict:
    """
    Load a trained checkpoint, fit the scalar calibration against the
    training set, and save it as `calibration.json` next to the
    checkpoint — matching the convention from the original research
    ("stored in calibration.json next to the checkpoint... applied at
    generation time"). `LoadDiffusionModel` auto-discovers this file.
    """
    from enerdatasynth.utils.config import PROCESSED_DIR, CHECKPOINT_DIR, DEVICE, ensure_dirs
    from enerdatasynth.models import build_denoiser
    from enerdatasynth.diffusion.ddpm import DDPM

    ensure_dirs()
    X_train = np.load(PROCESSED_DIR / "X_daily_profiles.npy")
    Z_train = np.load(PROCESSED_DIR / "Z_conditions.npy")

    model = build_denoiser(
        architecture, x_dim=X_train.shape[1], cond_dim=Z_train.shape[1],
        hidden_dim=hidden_dim, n_layers=n_layers, n_heads=n_heads,
    ).to(DEVICE)
    ckpt_path = CHECKPOINT_DIR / run_id / "best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    ddpm = DDPM(T=diffusion_steps, device=DEVICE)
    calibration = calibrate_from_samples(
        model, ddpm, X_train, Z_train, n_fit=n_fit, seed=seed, device=DEVICE,
    )
    calibration["checkpoint"] = str(ckpt_path)

    out_path = CHECKPOINT_DIR / run_id / "calibration.json"
    save_calibration(out_path, calibration)
    calibration["path"] = str(out_path)
    return calibration
