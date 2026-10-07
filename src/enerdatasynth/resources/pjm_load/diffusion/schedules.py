"""
src/enerdatasynth/resources/pjm_load/diffusion/schedules.py
===========================================================
Beta-schedule constructors for the DDPM forward process.

Extracted from week6.py's `make_betas` (see research_history/week6.py),
the one piece of the architecture/noise-schedule sweep that added real
capability beyond what `ddpm.py` supports on its own:
`diffusion/ddpm.py`'s `DDPM.__init__` only builds a linear schedule.
This module makes the cosine and quadratic variants available to any
caller without re-deriving the schedule math again.
"""

from __future__ import annotations
import math
import torch


def make_betas(schedule: str, T: int = 1000, beta_start: float = 1e-4,
                beta_end: float = 0.02) -> torch.Tensor:
    """
    Build a beta schedule for the DDPM forward process.

    schedule : "linear", "quadratic", or "cosine"
    T        : number of diffusion steps
    """
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T)
    if schedule == "quadratic":
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, T) ** 2
    if schedule == "cosine":
        s = 0.008
        steps = torch.arange(T + 1) / T
        alpha_bar = torch.cos(((steps + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas.clamp(1e-4, 0.999)
    raise ValueError(f"Unknown schedule: {schedule!r}")
