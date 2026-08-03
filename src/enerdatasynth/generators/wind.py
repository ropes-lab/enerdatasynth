"""
src/enerdatasynth/generators/wind.py
=======================================
`WindDiffusionModel` -- the wind-domain implementation of `BaseGenerator`,
wrapping the vendored `wind_diffusion` package (`src/wind_diffusion/`,
copied from Data-For-Good) rather than reimplementing diffusion sampling
here. The wind counterpart to `generators/load.py`'s `LoadDiffusionModel`.

Structural/API skeleton only: no trained wind checkpoint has been
published yet (`wind_diffusion/manifest.json`'s presets are all
`"pending"`), so constructing this against the bundled manifest raises
`wind_diffusion`'s own `CheckpointNotFoundError` until one exists --
expected, not a wiring bug. Pass an explicit `checkpoint=`/`norm_stats=`
pair (e.g. from a local `wind_diffusion` training run) to exercise this
against real weights today.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from wind_diffusion.config import PresetName
from wind_diffusion.data import NormStats
from wind_diffusion.diffusion import DiffusionSchedule
from wind_diffusion.evaluate import generate_multiple_for_condition, get_cached_model
from wind_diffusion.registry import resolve_weights
from wind_diffusion.utils import set_seed as _wind_set_seed

from enerdatasynth.generators.base import BaseGenerator
from enerdatasynth.utils.config import resolve_device


class WindDiffusionModel(BaseGenerator):
    """
    Tier-2 (research/development) interface for generating synthetic wind
    profiles from a trained `wind_diffusion` checkpoint. Delegates
    checkpoint resolution (`wind_diffusion.registry.resolve_weights` --
    already sha256-verified explicit-path/env-dir/cached-download
    resolution, strictly better than `LoadDiffusionModel`'s local-
    checkpoint-directory-only discovery) and sampling entirely to the
    vendored package rather than reimplementing either.
    """

    domain = "wind"

    def __init__(
        self,
        preset: PresetName = "3ch",
        checkpoint: str | Path | None = None,
        norm_stats: str | Path | None = None,
        device: str | None = None,
    ):
        self.preset = preset
        self.device = resolve_device(device)

        ckpt_path, norm_path = resolve_weights(preset, checkpoint=checkpoint, norm_stats=norm_stats)
        self.model, self.config, self._checkpoint_dict = get_cached_model(
            str(ckpt_path), device=self.device, preset=preset,
        )
        with open(norm_path) as f:
            self.norm_stats = NormStats.from_dict(json.load(f))

        self.schedule = DiffusionSchedule.build(
            T=self.config.diffusion_timesteps,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            beta_schedule=self.config.beta_schedule,
            device=self.device,
        )

        n_channels = self.config.n_model_channels
        self._clip_min = torch.full(
            (1, n_channels, 1), -self.config.sample_clip_std, dtype=torch.float32, device=self.device,
        )
        self._clip_max = torch.full(
            (1, n_channels, 1), self.config.sample_clip_std, dtype=torch.float32, device=self.device,
        )

    @classmethod
    def from_run_id(cls, run_id: str, **kwargs) -> "WindDiffusionModel":
        """
        `wind_diffusion` doesn't use enerdatasynth's `run_id`/`result.json`
        convention -- checkpoints resolve via `wind_diffusion.registry`
        instead, keyed by preset. `run_id` here IS the preset
        (`"1ch"`/`"2ch"`/`"3ch"`), kept only for `BaseGenerator`-contract
        symmetry with `LoadDiffusionModel.from_run_id(run_id)`.
        """
        return cls(preset=run_id, **kwargs)

    def sample(
        self,
        z: np.ndarray,
        seed: int | None = None,
        sampler: str = "ddim",
        steps: int = 50,
        eta: float = 1.0,
        guidance_scale: float = 1.0,
        guidance_slice=None,
        dynamic_threshold: bool = True,
        dynamic_threshold_percentile: float = 99.0,
    ) -> np.ndarray:
        """
        Generate one synthetic wind day per row of `z` -- a raw
        `(N, cond_dim)` conditioning array in `wind_diffusion`'s exact
        format (see `wind_diffusion.evaluate.build_condition_vector`).
        Build it via `enerdatasynth.model("wind", ...).set()` (Tier 1),
        not by hand, unless you already have conditioning vectors in
        wind_diffusion's own format.

        Returns `(N, seq_len, 2)`: `[..., 0]` = windspeed (m/s), `[..., 1]`
        = direction (degrees). Unlike `LoadDiffusionModel.sample()` (a
        single scalar per hour), a wind profile always carries both --
        collapsing to windspeed-only would silently drop direction for
        `2ch`/`3ch`.

        `sampler`/`steps`/`eta`/`dynamic_threshold`/`guidance_scale`
        default to `wind_diffusion`'s own measured-best recipe (see
        `MODEL_CARD.md`/`API_REVIEW.md` in the original repo), not the
        library's own conservative "off" defaults -- deliberately, since
        this Tier-2 entry point IS where research/development callers who
        want that recipe land.
        """
        if seed is not None:
            _wind_set_seed(seed)

        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:
            z = z[None, :]
        n = z.shape[0]

        df_gen = generate_multiple_for_condition(
            self.model, self.schedule, z, n,
            self.config.n_model_channels, self.config.seq_len,
            self.norm_stats.target_mean, self.norm_stats.target_std, self.preset,
            self._clip_min, self._clip_max, device=self.device,
            sampler=sampler, steps=steps, eta=eta,
            guidance_scale=guidance_scale, guidance_slice=guidance_slice,
            dynamic_threshold=dynamic_threshold, dynamic_threshold_percentile=dynamic_threshold_percentile,
        )
        windspeed = df_gen["windspeed_100m"].to_numpy().reshape(n, self.config.seq_len)
        direction = df_gen["winddirection_100m"].to_numpy().reshape(n, self.config.seq_len)
        return np.stack([windspeed, direction], axis=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
