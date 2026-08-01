"""
src/pjm_diffusion/models/denoiser.py
=======================================
Simple feedforward MLP denoiser for the conditional diffusion model.

This was the original baseline architecture. Research later found a
transformer denoiser (`pjm_diffusion.models.transformer.TransformerDenoiser`)
performs better on generation quality at a fraction of the parameters
— see research_history/reports/week6/arch/summary_report.md — and
`architecture="transformer"` is now the package default. This module
is kept as the `architecture="mlp"` option: a legacy/research baseline,
still fully supported, useful as a simpler point of comparison.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Encode the integer diffusion timestep t as a fixed sinusoidal
    embedding, the same scheme used in the original DDPM paper.
    This is the standard, simplest way to give the network a sense
    of "how noisy" the current input is — not a custom/fancy trick.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class SimpleMLPDenoiser(nn.Module):
    """
    Simple feedforward MLP denoiser.

    Inputs
    ------
    x_t : (B, x_dim)   noisy load profile at diffusion step t
    t   : (B,)         integer diffusion timestep
    z   : (B, cond_dim) conditioning vector

    Output
    ------
    eps_hat : (B, x_dim)  predicted noise

    Architecture
    ------------
    All three inputs (x_t, time embedding, z) are concatenated and
    passed through `n_layers` fully-connected layers of width
    `hidden_dim`, each followed by a SiLU activation. A final linear
    layer maps back to x_dim. No residual connections, no attention,
    no normalization tricks — this is the simplest version that works.
    """
    def __init__(self, x_dim: int, cond_dim: int,
                 hidden_dim: int = 128, n_layers: int = 4,
                 t_embed_dim: int = 32):
        super().__init__()
        self.x_dim    = x_dim
        self.cond_dim = cond_dim

        self.t_embed = SinusoidalTimeEmbedding(t_embed_dim)

        in_dim = x_dim + t_embed_dim + cond_dim
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, x_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_embed(t)                       # (B, t_embed_dim)
        h     = torch.cat([x_t, t_emb, z], dim=-1)     # (B, x_dim+t_embed_dim+cond_dim)
        return self.net(h)                             # (B, x_dim)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
