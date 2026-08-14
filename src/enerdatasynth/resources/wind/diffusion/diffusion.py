"""DDPM forward/reverse diffusion process: beta schedule, forward noising
(q_sample), and ancestral reverse sampling (ddpm_sample).

Deviation from the original scripts: `make_beta_schedule` is unchanged, but
`q_sample`/`ddpm_sample` no longer close over module-level globals
(`alphas_cumprod`, `betas`, `DEVICE`, `SAMPLE_CLIP_MIN/MAX`, ...). Instead
those precomputed tensors are bundled into a `DiffusionSchedule` object
(built once via `DiffusionSchedule.build(...)`) that gets passed in
explicitly. This makes the functions testable/importable without any
training-script side effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


def make_beta_schedule(T: int, beta_start: float, beta_end: float, schedule: str = "linear") -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T)
    elif schedule == "cosine":
        # Simple cosine schedule (Nichol & Dhariwal style, simplified)
        steps = torch.arange(T + 1, dtype=torch.float32)
        s = 0.008
        f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 1e-5, 0.999)
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")


@dataclass
class DiffusionSchedule:
    T: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor
    alphas_cumprod_prev: torch.Tensor
    sqrt_alphas_cumprod: torch.Tensor
    sqrt_one_minus_alphas_cumprod: torch.Tensor
    posterior_variance: torch.Tensor

    @classmethod
    def build(
        cls,
        T: int,
        beta_start: float,
        beta_end: float,
        beta_schedule: str = "linear",
        device: str | torch.device = "cpu",
    ) -> "DiffusionSchedule":
        betas = make_beta_schedule(T, beta_start, beta_end, beta_schedule).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), alphas_cumprod[:-1]])

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        return cls(
            T=T,
            betas=betas,
            alphas=alphas,
            alphas_cumprod=alphas_cumprod,
            alphas_cumprod_prev=alphas_cumprod_prev,
            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
            posterior_variance=posterior_variance,
        )


def q_sample(schedule: DiffusionSchedule, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
    """Forward diffusion: sample x_t given x_0 and timestep t (per-sample).

    x0: (B, channels, seq_len)
    t:  (B,) integer timesteps
    """
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ac = schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1)  # (B, 1, 1)
    sqrt_1m_ac = schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)  # (B, 1, 1)
    return sqrt_ac * x0 + sqrt_1m_ac * noise, noise


def clamp_x0(
    x0_pred: torch.Tensor,
    clip_min: torch.Tensor,
    clip_max: torch.Tensor,
    dynamic_threshold: bool = False,
    dynamic_threshold_percentile: float = 99.0,
) -> torch.Tensor:
    """Constrain predicted x0 to a physically-plausible standardized-space
    range at each reverse-diffusion step.

    dynamic_threshold=False (default): static clamp to [clip_min, clip_max].
    Measured to cause a self-reinforcing lock-in, not just an occasional
    correction: once a position's raw x0_pred exceeds the bound, hard-
    clamping pins it exactly at the boundary, and ~94% of the time it is
    STILL pinned there many reverse-diffusion steps later -- the denoiser
    was only ever trained on real (never-clamped) x0, so it has no learned
    way to "recover" from a clamped state once it enters one. Traced runs
    showed single positions pinned at the exact boundary for up to 5+ hours
    of simulated time, driving ~2x-inflated variance and a much-too-fast
    ACF decay relative to real data (see the realization/ACF/moments
    analysis in presentation_plots/).

    dynamic_threshold=True ("dynamic thresholding", per Imagen/Saharia et al.
    2022): per-sample, compute s = the dynamic_threshold_percentile of
    |x0_pred| across the whole window, floored at the static bound (so it
    only ever tightens, never loosens, the static clamp), clip to [-s, s],
    then rescale by clip_max/s. This pulls outlier values in proportionally
    relative to the whole prediction, rather than truncating individual
    positions against a hard wall they can get stuck against repeatedly.
    Measured: cuts the fraction of values sitting at the boundary from
    ~15.6% to ~1.1%, and standardized-space std from 2.42 to 1.56 (real
    data's std is ~1.0) -- a purely generation-time change, no retraining
    needed since it only touches how sampling clips x0, not the model.
    """
    if not dynamic_threshold:
        return torch.clamp(x0_pred, clip_min, clip_max)

    static_bound = float(clip_max.reshape(-1)[0].item())
    b = x0_pred.shape[0]
    abs_flat = x0_pred.reshape(b, -1).abs()
    s = torch.quantile(abs_flat, dynamic_threshold_percentile / 100.0, dim=1)
    s = torch.clamp(s, min=static_bound)
    s = s.view(b, *([1] * (x0_pred.dim() - 1)))
    x0_pred = torch.clamp(x0_pred, -s, s)
    return x0_pred / s * static_bound


def _cfg_predict(
    model,
    x: torch.Tensor,
    t: torch.Tensor,
    z: torch.Tensor,
    guidance_scale: float,
    guidance_slice: slice | None = None,
) -> torch.Tensor:
    """Classifier-free guidance noise prediction.

    guidance_scale=1.0 (default) is a pure pass-through -- exactly one model
    call, identical to every checkpoint's original behavior, regardless of
    whether that checkpoint was trained with cond_dropout_prob>0.
    guidance_scale != 1.0 makes a second call with `guidance_slice` of z
    zeroed (default None = the whole vector) and extrapolates away from it:
    `noise = uncond + guidance_scale * (cond - uncond)`. Meaningless on a
    checkpoint trained with cond_dropout_prob=0 in the matching scope --
    the zeroed slice is out-of-distribution for it and the extrapolation has
    nothing real to amplify.

    `guidance_slice` (regime-scoped guidance): measured to matter in
    practice, not just in theory -- whole-vector guidance (guidance_slice=
    None) amplifies the entire 25-dim condition equally, dominated by
    month+day_of_week, and made regime steering *worse* (see
    reports/BASELINE_1CH_CFG_GS*.md). Passing
    `transforms.wind_regime_slice(condition_on)` here instead zeroes only
    the 3-dim wind-regime one-hot for the "uncond" branch, leaving
    month/day_of_week/temp identical in both branches, so the extrapolation
    isolates and amplifies specifically the regime signal's marginal
    contribution.
    """
    pred_cond = model(x, t, z)
    if guidance_scale == 1.0:
        return pred_cond
    z_uncond = z.clone()
    target_slice = guidance_slice if guidance_slice is not None else slice(None)
    z_uncond[:, target_slice] = 0.0
    pred_uncond = model(x, t, z_uncond)
    return pred_uncond + guidance_scale * (pred_cond - pred_uncond)


@torch.no_grad()
def ddpm_sample(
    model,
    schedule: DiffusionSchedule,
    cond_matrix: np.ndarray,
    n_channels: int,
    seq_len: int,
    clip_min: torch.Tensor,
    clip_max: torch.Tensor,
    device: str | torch.device = "cpu",
    n_repeats: int = 1,
    guidance_scale: float = 1.0,
    guidance_slice: slice | None = None,
    dynamic_threshold: bool = False,
    dynamic_threshold_percentile: float = 99.0,
) -> np.ndarray:
    """Generate samples for each condition vector in cond_matrix via ancestral
    DDPM sampling, with predicted-x0 clamping to `[clip_min, clip_max]`
    (standardized space) at each reverse step.

    cond_matrix: np.ndarray (N, cond_dim)
    n_repeats: how many independent samples to draw per condition vector
    clip_min/clip_max: tensors broadcastable to (1, n_channels, 1)
    guidance_scale/guidance_slice: classifier-free guidance scale and
        target slice, see `_cfg_predict`. guidance_scale=1.0 (default) = no
        guidance, one model call per step.
    dynamic_threshold/dynamic_threshold_percentile: see `clamp_x0`.
        dynamic_threshold=False (default) = identical to every checkpoint's
        original behavior.

    Returns: np.ndarray (N * n_repeats, n_channels, seq_len) in *standardized* space.
    """
    model.eval()
    z = torch.tensor(cond_matrix, dtype=torch.float32, device=device)
    z = z.repeat_interleave(n_repeats, dim=0)
    n = z.shape[0]

    x = torch.randn(n, n_channels, seq_len, device=device)

    T = schedule.T
    betas = schedule.betas
    alphas = schedule.alphas
    alphas_cumprod = schedule.alphas_cumprod
    alphas_cumprod_prev = schedule.alphas_cumprod_prev
    posterior_variance = schedule.posterior_variance

    for t_step in reversed(range(T)):
        t = torch.full((n,), t_step, device=device, dtype=torch.long)
        pred_noise = _cfg_predict(model, x, t, z, guidance_scale, guidance_slice)

        alpha_t = alphas[t_step]
        alpha_cumprod_t = alphas_cumprod[t_step]
        beta_t = betas[t_step]

        # Predict x0 and clamp it in standardized space
        sqrt_ac = torch.sqrt(alpha_cumprod_t)
        sqrt_1m_ac = torch.sqrt(1.0 - alpha_cumprod_t)
        x0_pred = (x - sqrt_1m_ac * pred_noise) / sqrt_ac
        x0_pred = clamp_x0(x0_pred, clip_min, clip_max, dynamic_threshold, dynamic_threshold_percentile)

        # Compute mean of p(x_{t-1} | x_t) using clamped x0_pred
        alpha_cumprod_prev_t = alphas_cumprod_prev[t_step]
        coef_x0 = (torch.sqrt(alpha_cumprod_prev_t) * beta_t) / (1.0 - alpha_cumprod_t)
        coef_xt = (torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev_t)) / (1.0 - alpha_cumprod_t)
        mean = coef_x0 * x0_pred + coef_xt * x

        if t_step > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(posterior_variance[t_step])
            x = mean + sigma * noise
        else:
            x = mean

    return x.cpu().numpy()
