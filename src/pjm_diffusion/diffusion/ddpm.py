"""
src/pjm_diffusion/diffusion/ddpm.py
======================================
Denoising Diffusion Probabilistic Model (DDPM) — forward noising
process, training loss, and reverse sampling.

Standard DDPM only: no custom losses, no Fourier/spectral losses,
no moment-matching losses.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class DDPM:
    """
    Forward process:
        q(x_t | x_0) = N( sqrt(alpha_bar_t) * x_0,  (1 - alpha_bar_t) * I )
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

    Training loss:
        L = MSE( eps_hat, eps )      <- standard diffusion MSE loss ONLY

    Reverse process (sampling):
        Standard ancestral sampling, following the original DDPM
        sampling equations (Ho et al., 2020). A DDIM sampler was tried
        earlier in research and found buggy (see research_history/
        scripts/sampler_bugfix.py) — this ancestral sampler is the
        corrected, validated replacement and is the only one used
        anywhere in this package.
    """

    def __init__(self, T: int, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: str = "cpu"):
        self.T      = T
        self.device = device

        betas     = torch.linspace(beta_start, beta_end, T)
        alphas    = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.betas     = betas.to(device)
        self.alphas    = alphas.to(device)
        self.alpha_bar = alpha_bar.to(device)

    # ── Forward process ──────────────────────────────────────────────
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor | None = None):
        """Add noise to x0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        ab  = self.alpha_bar[t].unsqueeze(1)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        return x_t, noise

    # ── Training loss (standard MSE, no variants) ───────────────────
    def training_loss(self, model: nn.Module, x0: torch.Tensor,
                       z: torch.Tensor) -> torch.Tensor:
        """
        Sample a random timestep for each item in the batch, add
        noise, ask the model to predict it. Return plain MSE.
        No custom loss terms are added here, by design.
        """
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=self.device)
        x_t, noise = self.q_sample(x0, t)
        eps_hat = model(x_t, t, z)
        return nn.functional.mse_loss(eps_hat, noise)

    # ── Reverse process (sampling) ───────────────────────────────────
    @torch.no_grad()
    def p_sample_step(self, model: nn.Module, x_t: torch.Tensor,
                       t_scalar: int, z: torch.Tensor) -> torch.Tensor:
        """One step of ancestral sampling: x_t -> x_{t-1}."""
        B = x_t.shape[0]
        t_vec = torch.full((B,), t_scalar, dtype=torch.long, device=self.device)

        eps_hat = model(x_t, t_vec, z)

        beta      = self.betas[t_scalar]
        alpha     = self.alphas[t_scalar]
        alpha_bar = self.alpha_bar[t_scalar]

        # Predicted mean of p(x_{t-1} | x_t)
        coef = beta / torch.sqrt(1 - alpha_bar)
        mean = (1 / torch.sqrt(alpha)) * (x_t - coef * eps_hat)

        if t_scalar > 0:
            noise = torch.randn_like(x_t)
            sigma = beta.sqrt()
            return mean + sigma * noise
        return mean

    @torch.no_grad()
    def sample(self, model: nn.Module, z: torch.Tensor,
               x_dim: int) -> torch.Tensor:
        """
        Full reverse process: generate one sample per row of z,
        starting from pure Gaussian noise.
        """
        model.eval()
        B = z.shape[0]
        x = torch.randn(B, x_dim, device=self.device)
        for t in reversed(range(self.T)):
            x = self.p_sample_step(model, x, t, z)
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
            x = x.clamp(-3, 3)
        return x

    @torch.no_grad()
    def generate_batch(self, model: nn.Module, z: torch.Tensor,
                        x_dim: int, batch_size: int = 256) -> torch.Tensor:
        """Generate samples in batches to keep memory bounded."""
        outputs = []
        for i in range(0, z.shape[0], batch_size):
            z_batch = z[i:i + batch_size]
            x_batch = self.sample(model, z_batch, x_dim)
            outputs.append(x_batch)
        return torch.cat(outputs, dim=0)
