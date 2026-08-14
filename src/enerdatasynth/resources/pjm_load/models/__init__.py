"""
src/enerdatasynth/models
===========================
Denoiser architectures, selected through `build_denoiser()` rather
than importing a specific class directly, so callers (training,
sampling, the public API) don't need to special-case each
architecture's constructor signature.

    "transformer" (default) -> TransformerDenoiser  — see transformer.py
    "mlp"                   -> SimpleMLPDenoiser     — legacy/research baseline
"""

from __future__ import annotations
import torch.nn as nn

from enerdatasynth.resources.pjm_load.models.denoiser import SimpleMLPDenoiser, SinusoidalTimeEmbedding
from enerdatasynth.resources.pjm_load.models.transformer import TransformerDenoiser
from enerdatasynth.utils.config import DEFAULT_ARCHITECTURE


def build_denoiser(architecture: str, x_dim: int, cond_dim: int,
                    hidden_dim: int = 128, n_layers: int = 4,
                    n_heads: int = 4, t_embed_dim: int | None = None) -> nn.Module:
    """
    Construct a denoiser by name.

    `hidden_dim`/`n_layers` map onto each architecture's natural
    equivalent (MLP: hidden width / block count; transformer: model
    dimension / encoder-layer count), so the same `RunConfig` sweeps
    either architecture without architecture-specific fields, aside
    from `n_heads` (transformer-only, ignored otherwise).
    """
    if architecture == "transformer":
        return TransformerDenoiser(
            x_dim=x_dim, cond_dim=cond_dim,
            model_dim=hidden_dim, num_heads=n_heads, num_layers=n_layers,
            t_embed_dim=t_embed_dim or 64,
        )
    if architecture == "mlp":
        return SimpleMLPDenoiser(
            x_dim=x_dim, cond_dim=cond_dim,
            hidden_dim=hidden_dim, n_layers=n_layers,
            t_embed_dim=t_embed_dim or 32,
        )
    raise ValueError(f"Unknown architecture: {architecture!r} (expected 'transformer' or 'mlp')")


__all__ = [
    "build_denoiser",
    "DEFAULT_ARCHITECTURE",
    "TransformerDenoiser",
    "SimpleMLPDenoiser",
    "SinusoidalTimeEmbedding",
]
