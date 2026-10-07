"""
src/enerdatasynth/resources/pjm_load/models/transformer.py
==========================================================
Transformer denoiser — the Week-6 architecture-comparison winner,
promoted to the primary/default architecture in Week 7. It beats the
MLP baseline on 6 of 7 generation metrics at ~6x fewer parameters
(110K vs 688K) and beats every classical baseline (AR/ARMA/ARIMA/
SARIMA) on 7 of 7 — see
research_history/reports/week6/arch/summary_report.md and
data/research_history/pjm_load/reports/week7/summary_report.md.

Extracted from research_history/scripts/week7.py's `TransformerDenoiser`.
Submodule attribute names (`cond`, `val`, `pos`, `enc`, `out`) are kept
identical to the original so trained checkpoints from that lineage
remain loadable via `load_state_dict` given matching dimensions. The
one behavioral fix applied here: the original hardcoded the sequence
length used for the positional embedding to a module-level global
(`H = 24`) instead of the constructor's own `x_dim` argument — harmless
in practice (this project only ever uses 24-hour profiles) but not a
correct pattern for a reusable library class, so `self.pos` is now
sized from `x_dim` directly. `t_embed` reuses
`SinusoidalTimeEmbedding` from `models/denoiser.py` instead of a
second duplicate implementation — that class holds no learnable
parameters, so this has no effect on checkpoint compatibility.

Caveat (see the migration report): the checkpoint this architecture
was actually validated against (`transformer_baseline_z5.pt`) was
trained with `cond_dim=6` (the research lineage's raw "Z5" conditioning
vector), not the 23-dimensional one-hot conditioning
`enerdatasynth.resources.pjm_load.data.preprocess` currently produces. No pretrained
checkpoint compatible with the current preprocessing pipeline exists
yet — training a fresh one is required (`enerdatasynth train`, which
now defaults to this architecture).
"""

from __future__ import annotations
import torch
import torch.nn as nn

from enerdatasynth.resources.pjm_load.models.denoiser import SinusoidalTimeEmbedding


class TransformerDenoiser(nn.Module):
    """
    Transformer-encoder denoiser.

    Inputs
    ------
    x_t : (B, x_dim)    noisy load profile at diffusion step t
    t   : (B,)          integer diffusion timestep
    z   : (B, cond_dim) conditioning vector

    Output
    ------
    eps_hat : (B, x_dim)  predicted noise

    Architecture
    ------------
    Each hour of the profile is treated as one token: a learned
    per-hour positional embedding plus a shared projection of
    (timestep embedding, conditioning vector) is added to a linear
    embedding of that hour's value, then run through a standard
    `nn.TransformerEncoder`. A final linear layer maps each token
    back to a scalar noise prediction.
    """

    def __init__(self, x_dim: int, cond_dim: int,
                 model_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, t_embed_dim: int = 64):
        super().__init__()
        self.x_dim = x_dim
        self.cond_dim = cond_dim

        self.t_embed = SinusoidalTimeEmbedding(t_embed_dim)
        self.cond = nn.Sequential(
            nn.Linear(cond_dim + t_embed_dim, model_dim), nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.val = nn.Linear(1, model_dim)
        self.pos = nn.Parameter(torch.randn(1, x_dim, model_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, dim_feedforward=model_dim * 4,
            batch_first=True, dropout=0.0, activation="gelu",
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(model_dim, 1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                z: torch.Tensor) -> torch.Tensor:
        cond = self.cond(torch.cat([z, self.t_embed(t)], dim=-1))[:, None, :]  # (B, 1, model_dim)
        h = self.val(x_t[:, :, None]) + self.pos + cond                       # (B, x_dim, model_dim)
        h = self.enc(h)
        return self.out(h)[:, :, 0]                                            # (B, x_dim)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
