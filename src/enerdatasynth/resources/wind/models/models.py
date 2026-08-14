"""Model architecture: sinusoidal time embedding, condition MLP, dilated
residual Conv1D blocks, and the hybrid Conv1D + Transformer noise-prediction
denoiser.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding for the diffusion timestep (like in Transformers)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer timesteps
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device, dtype=torch.float32) / half_dim
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        if self.dim % 2 == 1:  # pad if odd
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb


class CondMLP(nn.Module):
    """Small MLP that embeds the raw condition vector z into a learned embedding."""

    def __init__(self, cond_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ResidualConvBlock(nn.Module):
    """Residual temporal Conv1D block.

    Conditioning is injected as an additive bias (broadcast across the time
    dimension) before the first convolution. Structure per block:

        x -> (+ cond_bias) -> Conv1d -> GroupNorm -> GELU -> Dropout
          -> Conv1d -> GroupNorm -> (+ x, residual skip)
    """

    def __init__(self, hidden_dim: int, kernel_size: int, dropout: float, dilation: int = 1, norm_groups: int = 8):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(norm_groups, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(norm_groups, hidden_dim)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x:    (B, hidden_dim, seq_len)
        # cond: (B, hidden_dim) -> broadcast over the time dimension
        cond_bias = cond.unsqueeze(-1)  # (B, hidden_dim, 1)

        h = x + cond_bias
        h = self.conv1(h)
        h = self.norm1(h)
        h = self.act(h)
        h = self.dropout(h)

        h = self.conv2(h)
        h = self.norm2(h)

        return x + h  # residual skip connection


class TransformerDenoiser(nn.Module):
    """Hybrid Conv1D + Transformer noise-prediction network for (B, n_channels, seq_len) sequences."""

    def __init__(
        self,
        n_channels: int,
        cond_dim: int,
        hidden_dim: int,
        num_conv_blocks: int,
        num_transformer_layers: int,
        num_heads: int,
        kernel_size: int,
        dropout: float,
        time_emb_dim: int,
        cond_emb_dim: int,
        seq_len: int = 288,
        norm_first: bool = True,
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(n_channels, hidden_dim, kernel_size=3, padding=1)

        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.cond_mlp = CondMLP(cond_dim, hidden_dim=128, out_dim=cond_emb_dim)

        self.cond_proj = nn.Sequential(
            nn.Linear(time_emb_dim + cond_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        norm_groups = 8 if hidden_dim % 8 == 0 else 1
        self.conv_blocks = nn.ModuleList(
            [
                ResidualConvBlock(hidden_dim, kernel_size, dropout, dilation=2**i, norm_groups=norm_groups)
                for i in range(num_conv_blocks)
            ]
        )

        # Global temporal modeling
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.output_head = nn.Conv1d(hidden_dim, n_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: (B, n_channels, seq_len)
        h = self.input_proj(x)  # (B, hidden_dim, seq_len)

        t_emb = self.time_mlp(self.time_embedding(t))  # (B, time_emb_dim)
        z_emb = self.cond_mlp(z)  # (B, cond_emb_dim)
        cond = self.cond_proj(torch.cat([t_emb, z_emb], dim=-1))  # (B, hidden_dim)

        # Apply local residual convolutions
        for block in self.conv_blocks:
            h = block(h, cond)

        # Transpose to (B, seq_len, hidden_dim)
        h = h.permute(0, 2, 1)

        # Add positional embedding
        h = h + self.pos_embed

        # Prepend conditioning token
        cond_token = cond.unsqueeze(1)  # (B, 1, hidden_dim)
        h = torch.cat([cond_token, h], dim=1)  # (B, seq_len + 1, hidden_dim)

        # Transformer encoder
        h = self.transformer(h)  # (B, seq_len + 1, hidden_dim)

        # Slice off condition token and transpose back
        h = h[:, 1:, :]  # (B, seq_len, hidden_dim)
        h = h.permute(0, 2, 1)  # (B, hidden_dim, seq_len)

        return self.output_head(h)  # (B, n_channels, seq_len) predicted noise
