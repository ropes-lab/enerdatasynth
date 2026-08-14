"""Sampler abstraction for turning a trained denoiser + DiffusionSchedule
into generated samples.

Two strategies are exposed behind one entry point, `sample()`:

- ``sampler="ancestral"``: the original full-length DDPM reverse process
  (``diffusion.ddpm_sample``), one model call per timestep in
  ``schedule.T``. Slow but exact to how the models were trained/evaluated
  originally.
- ``sampler="ddim"`` (default): Denoising Diffusion Implicit Models
  sampling over a strided subsequence of ``steps`` timesteps. Same
  predicted-x0-and-clamp step as ancestral sampling at each selected
  timestep, but far fewer model calls (e.g. 50 instead of 200), which is
  what makes "instant" interactive generation feasible.

Both paths share the same output contract: a ``(N * n_repeats, n_channels,
seq_len)`` ndarray in standardized space.
"""

from __future__ import annotations

import numpy as np
import torch

from enerdatasynth.resources.wind.diffusion.diffusion import DiffusionSchedule, _cfg_predict, clamp_x0, ddpm_sample
from enerdatasynth.resources.wind.utils import resolve_device

# ---------------------------------------------------------------------------
# CPU latency heuristic
# ---------------------------------------------------------------------------
#
# Threshold is expressed in "total model forward-pass calls" =
# n_repeats * n_conditions * steps (or schedule.T for ancestral), since each
# reverse-process step is exactly one model.forward() call and that
# dominates wall-clock time on CPU (no GPU parallelism across the batch to
# hide it). 5000 is a rough empirical line: for the transformer denoiser
# sizes used in this project (hidden_dim ~64, seq_len=288, a few conv/
# transformer layers), a single CPU forward pass over a modest batch takes
# on the order of a few milliseconds to a couple tens of milliseconds, so
# 5000 calls lands in the "tens of seconds and up" range - past the point
# where a call feels interactive. It's intentionally a coarse heuristic, not
# a measured guarantee, since actual latency depends heavily on batch size,
# CPU, and model width.
_CPU_CALL_COUNT_WARN_THRESHOLD = 5000
_ASSUMED_MS_PER_CPU_CALL = 8.0  # rough order-of-magnitude estimate, see above


def _maybe_warn_cpu_latency(device: torch.device, n_conditions: int, n_repeats: int, steps: int, sampler: str) -> None:
    if device.type != "cpu":
        return
    total_calls = n_conditions * n_repeats * steps
    if total_calls <= _CPU_CALL_COUNT_WARN_THRESHOLD:
        return

    est_seconds = total_calls * _ASSUMED_MS_PER_CPU_CALL / 1000.0
    suggestion = (
        "reduce `steps`, reduce `n_repeats`/number of conditions"
        if sampler != "ancestral"
        else "reduce `n_repeats`/number of conditions, or pass sampler=\"ddim\" (with a small `steps`) instead of \"ancestral\""
    )
    print(
        f"[enerdatasynth.resources.wind.diffusion.samplers] WARNING: about to make ~{total_calls} model forward calls "
        f"on CPU (sampler={sampler!r}), estimated ~{est_seconds:.1f}s. To speed this up: {suggestion}."
    )


# ---------------------------------------------------------------------------
# DDIM
# ---------------------------------------------------------------------------


@torch.no_grad()
def _ddim_sample(
    model,
    schedule: DiffusionSchedule,
    cond_matrix: np.ndarray,
    n_channels: int,
    seq_len: int,
    clip_min: torch.Tensor,
    clip_max: torch.Tensor,
    device: torch.device,
    n_repeats: int = 1,
    steps: int = 50,
    eta: float = 0.0,
    guidance_scale: float = 1.0,
    guidance_slice: slice | None = None,
    dynamic_threshold: bool = False,
    dynamic_threshold_percentile: float = 99.0,
) -> np.ndarray:
    """DDIM reverse sampling over a strided subsequence of `steps` timesteps.

    Mirrors `diffusion.ddpm_sample`'s predicted-x0-and-clamp math at each
    selected timestep, then applies the DDIM update rule (deterministic
    when eta=0, matching DDPM-style stochasticity when eta=1).

    guidance_scale/guidance_slice: classifier-free guidance scale and target
        slice, see `diffusion._cfg_predict`. guidance_scale=1.0 (default) =
        no guidance, one model call per step.
    dynamic_threshold/dynamic_threshold_percentile: see `diffusion.clamp_x0`.
        dynamic_threshold=False (default) = identical to prior behavior.
    """
    model.eval()
    z = torch.tensor(cond_matrix, dtype=torch.float32, device=device)
    z = z.repeat_interleave(n_repeats, dim=0)
    n = z.shape[0]

    x = torch.randn(n, n_channels, seq_len, device=device)

    T = schedule.T
    alphas_cumprod = schedule.alphas_cumprod

    # Strided subsequence of timesteps, e.g. steps=50 out of T=200 ->
    # roughly every 4th timestep. np.unique both sorts ascending and
    # deduplicates (steps could exceed T after rounding collisions).
    raw_steps = min(steps, T)
    selected = np.unique(np.round(np.linspace(0, T - 1, raw_steps)).astype(int))
    selected = selected[::-1]  # descending: start at the noisiest timestep

    for i, t_cur in enumerate(selected):
        t_cur = int(t_cur)
        t_prev = int(selected[i + 1]) if i + 1 < len(selected) else -1

        t_batch = torch.full((n,), t_cur, device=device, dtype=torch.long)
        pred_noise = _cfg_predict(model, x, t_batch, z, guidance_scale, guidance_slice)

        alpha_cumprod_t = alphas_cumprod[t_cur]
        sqrt_ac_t = torch.sqrt(alpha_cumprod_t)
        sqrt_1m_ac_t = torch.sqrt(1.0 - alpha_cumprod_t)

        # Predict x0 and clamp it in standardized space (same as ddpm_sample).
        x0_pred = (x - sqrt_1m_ac_t * pred_noise) / sqrt_ac_t
        x0_pred = clamp_x0(x0_pred, clip_min, clip_max, dynamic_threshold, dynamic_threshold_percentile)

        # Recompute the noise direction consistent with the clamped x0, so
        # the DDIM update below doesn't mix an unclamped epsilon with a
        # clamped x0 prediction.
        eps_hat = (x - sqrt_ac_t * x0_pred) / sqrt_1m_ac_t

        if t_prev >= 0:
            alpha_cumprod_prev = alphas_cumprod[t_prev]
        else:
            alpha_cumprod_prev = torch.ones_like(alpha_cumprod_t)

        if eta > 0 and t_prev >= 0:
            sigma = eta * torch.sqrt(
                (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t) * (1.0 - alpha_cumprod_t / alpha_cumprod_prev)
            )
        else:
            sigma = torch.zeros_like(alpha_cumprod_t)

        sqrt_ac_prev = torch.sqrt(alpha_cumprod_prev)
        dir_coeff = torch.sqrt(torch.clamp(1.0 - alpha_cumprod_prev - sigma**2, min=0.0))

        mean = sqrt_ac_prev * x0_pred + dir_coeff * eps_hat

        if sigma.item() > 0:
            noise = torch.randn_like(x)
            x = mean + sigma * noise
        else:
            x = mean

    return x.cpu().numpy()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def sample(
    model,
    schedule: DiffusionSchedule,
    cond_matrix: np.ndarray,
    n_channels: int,
    seq_len: int,
    clip_min: torch.Tensor,
    clip_max: torch.Tensor,
    device: str | torch.device = "cpu",
    n_repeats: int = 1,
    sampler: str = "ddim",
    steps: int = 50,
    eta: float = 0.0,
    guidance_scale: float = 1.0,
    guidance_slice: slice | None = None,
    dynamic_threshold: bool = False,
    dynamic_threshold_percentile: float = 99.0,
) -> np.ndarray:
    """Generate samples from `model` via the requested sampler.

    sampler="ancestral": delegates to `diffusion.ddpm_sample` (full T-step
        reverse process); `steps`/`eta` are ignored.
    sampler="ddim" (default): strided DDIM sampling over `steps` timesteps.
    guidance_scale/guidance_slice: classifier-free guidance scale and target
        slice (see `diffusion._cfg_predict`), passed through to whichever
        sampler is selected. guidance_scale=1.0 (default) = no guidance, one
        model call per step -- behavior is completely unchanged from before
        this parameter existed. Guidance is only meaningful against a
        checkpoint trained with `cond_dropout_prob > 0` in the matching
        scope; doubles model calls per step when guidance_scale != 1.0.
        guidance_slice=None (default) guides the whole condition vector;
        pass `transforms.wind_regime_slice(condition_on)` to guide only the
        wind-regime component (see `_cfg_predict` for why that matters).
    dynamic_threshold/dynamic_threshold_percentile: see `diffusion.clamp_x0`.
        dynamic_threshold=False (default) = identical to every prior release's
        behavior. Unlike guidance, this needs no special training -- it only
        changes how x0 is clamped during sampling, so it's safe to enable
        against any existing checkpoint.

    Returns: np.ndarray (N * n_repeats, n_channels, seq_len) in standardized
    space, same contract as `diffusion.ddpm_sample`.
    """
    device_obj = device if isinstance(device, torch.device) else resolve_device(device)
    n_conditions = cond_matrix.shape[0]

    if sampler == "ancestral":
        _maybe_warn_cpu_latency(device_obj, n_conditions, n_repeats, schedule.T, sampler)
        return ddpm_sample(
            model,
            schedule,
            cond_matrix,
            n_channels,
            seq_len,
            clip_min,
            clip_max,
            device=device_obj,
            n_repeats=n_repeats,
            guidance_scale=guidance_scale,
            guidance_slice=guidance_slice,
            dynamic_threshold=dynamic_threshold,
            dynamic_threshold_percentile=dynamic_threshold_percentile,
        )
    elif sampler == "ddim":
        _maybe_warn_cpu_latency(device_obj, n_conditions, n_repeats, steps, sampler)
        return _ddim_sample(
            model,
            schedule,
            cond_matrix,
            n_channels,
            seq_len,
            clip_min,
            clip_max,
            device=device_obj,
            n_repeats=n_repeats,
            steps=steps,
            eta=eta,
            guidance_scale=guidance_scale,
            guidance_slice=guidance_slice,
            dynamic_threshold=dynamic_threshold,
            dynamic_threshold_percentile=dynamic_threshold_percentile,
        )
    else:
        raise ValueError(f"Unknown sampler {sampler!r}; expected 'ancestral' or 'ddim'.")
