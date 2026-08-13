"""
src/enerdatasynth/generators/wind.py
=======================================
`WindDiffusionModel` -- the wind-domain implementation of `BaseGenerator`,
wrapping the vendored `wind_diffusion` package (`src/wind_diffusion/`,
copied from Data-For-Good) rather than reimplementing diffusion sampling
here. The wind counterpart to `generators/load.py`'s `LoadDiffusionModel`.

Real trained checkpoints (1ch/2ch/3ch) are bundled under
`data/checkpoints/wind/<preset>/` and used automatically -- see
`generators/wind_conditioning.py`'s `wind_construct`, which is what
`enerdatasynth.model("wind", ...)` actually calls. Constructing
`WindDiffusionModel` directly (bypassing that) falls through to
`wind_diffusion.registry.resolve_weights`'s own chain (explicit path ->
`WIND_DIFFUSION_CHECKPOINTS` env dir -> manifest download), which *does*
raise `CheckpointNotFoundError` for a preset with no local checkpoint
and no manifest-published weights -- not a wiring bug, just a different,
lower-level entry point than the SDK's.
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
from wind_diffusion.registry import describe_model, resolve_weights
from wind_diffusion.transforms import wind_regime_slice
from wind_diffusion.utils import set_seed as _wind_set_seed

# Fallback if a preset's manifest entry has no recommended_generation for
# some reason (shouldn't happen against the bundled manifest.json, but a
# caller could point checkpoint=/norm_stats= at something manifest.json
# knows nothing about). Deliberately the "safe against any checkpoint"
# values (guidance_scale=1.0 disables guidance; eta=1.0/dynamic_threshold
# have no training dependency), NOT any preset's specific recommended
# recipe -- those differ per preset (see __init__'s docstring: eta=1.0 is
# actually wrong for 2ch's own recommendation, eta=0.0) and this fallback
# only runs when we have no manifest data to know which preset we'd even
# be recommending for.
_GENERATION_DEFAULTS_FALLBACK = {
    "sampler": "ddim", "steps": 50, "eta": 1.0,
    "guidance_scale": 1.0, "guidance_scope": "full",
    "dynamic_threshold": True, "dynamic_threshold_percentile": 99.0,
}

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
        # "1ch" to match enerdatasynth.model("wind")'s own default (the
        # first entry in wind_conditioning.py's registered zones list) --
        # these two entry points previously disagreed (this defaulted to
        # "3ch"), so the same "give me wind, no preference" request
        # resolved to a different preset depending on which one you called.
        preset: PresetName = "1ch",
        checkpoint: str | Path | None = None,
        norm_stats: str | Path | None = None,
        device: str | None = None,
    ):
        self.preset = preset
        self.device = resolve_device(device)

        self._recommended_generation = describe_model(preset).get("recommended_generation") or dict(
            _GENERATION_DEFAULTS_FALLBACK
        )

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
        sampler: str | None = None,
        steps: int | None = None,
        eta: float | None = None,
        guidance_scale: float | None = None,
        guidance_scope: str | None = None,
        dynamic_threshold: bool | None = None,
        dynamic_threshold_percentile: float | None = None,
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

        `sampler`/`steps`/`eta`/`guidance_scale`/`guidance_scope`/
        `dynamic_threshold`/`dynamic_threshold_percentile`: `None`
        (default, for any of these) resolves to *this preset's own*
        `recommended_generation` from `wind_diffusion`'s bundled
        `manifest.json` (fetched once in `__init__`), not one fixed set of
        defaults shared across presets -- these genuinely differ per
        preset. Concretely: `1ch`/`3ch` recommend `eta=1.0,
        guidance_scale=40.0, guidance_scope="wind_regime"`; `2ch` requires
        `eta=0.0, guidance_scale=1.0` instead (guidance measurably breaks
        `2ch` even with dynamic thresholding on, and `eta=1.0` -- a clear
        win for `1ch`/`3ch` -- was measured *mixed* for `2ch` and not
        adopted as its default). Passing an explicit value for any of
        these overrides that preset's recommendation, same as always.
        """
        if seed is not None:
            _wind_set_seed(seed)

        rec = self._recommended_generation
        sampler = rec["sampler"] if sampler is None else sampler
        steps = rec["steps"] if steps is None else steps
        eta = rec["eta"] if eta is None else eta
        guidance_scale = rec["guidance_scale"] if guidance_scale is None else guidance_scale
        guidance_scope = rec["guidance_scope"] if guidance_scope is None else guidance_scope
        dynamic_threshold = rec["dynamic_threshold"] if dynamic_threshold is None else dynamic_threshold
        dynamic_threshold_percentile = (
            rec["dynamic_threshold_percentile"] if dynamic_threshold_percentile is None
            else dynamic_threshold_percentile
        )
        guidance_slice = (
            wind_regime_slice(self.config.condition_on, wind_regime_encoding=self.config.wind_regime_encoding)
            if guidance_scope == "wind_regime" else None
        )

        z = self._ensure_batch(z)
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

