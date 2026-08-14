"""Public v2 API: `generate_wind_profile` + `WindProfile`, plus the model
registry lookups (`list_models`/`describe_model`) and a `preload` convenience
wrapper. This is the intended entry point for library consumers -- it wires
together `enerdatasynth.resources.wind.registry` (weights resolution),
`enerdatasynth.resources.wind.evaluation.evaluate` (cached model loading + generation), and
`enerdatasynth.resources.wind.diffusion.diffusion`/`samplers` (the actual sampling call chain)
behind a single ergonomic function.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from enerdatasynth.resources.wind.config import PresetName
from enerdatasynth.resources.wind.data.data import NormStats
from enerdatasynth.resources.wind.diffusion.diffusion import DiffusionSchedule
from enerdatasynth.resources.wind.evaluation.evaluate import (
    build_condition_vector,
    generate_multiple_for_condition,
    get_cached_model,
)
from enerdatasynth.resources.wind.evaluation.evaluate import preload as _evaluate_preload
from enerdatasynth.resources.wind.exceptions import InvalidConditionError
from enerdatasynth.resources.wind.registry import describe_model as _registry_describe_model
from enerdatasynth.resources.wind.registry import list_models as _registry_list_models
from enerdatasynth.resources.wind.data.transforms import assign_wind_bucket, wind_persistence_slice, wind_regime_slice
from enerdatasynth.resources.wind.registry import resolve_weights
from enerdatasynth.resources.wind.utils import resolve_device, set_seed

_DEFAULT_WIND_REGIME_FALLBACK = "mid"


# ---------------------------------------------------------------------------
# WindProfile
# ---------------------------------------------------------------------------


@dataclass
class WindProfile:
    """A batch of generated synthetic wind-profile realizations for one
    condition (preset + wind regime + anchor date).

    Attributes:
        time: DatetimeIndex, freq="5min", length == seq_len, anchored at
            `date` (or a fixed default date if none was given).
        windspeed: (n_realizations, seq_len) array, m/s. May have fewer than
            the requested n_realizations rows if generate_wind_profile was
            called with rejection_sampling=True and ran out of attempts
            before finding enough matches (see metadata["rejection_stats"]).
        direction: (n_realizations, seq_len) array, degrees (0-360).
        condition: the resolved condition (wind regime, bucket edges, ...).
        metadata: generation provenance (preset, sampler, steps, checkpoint
            path, device, wall-clock seconds, ...).
    """

    time: pd.DatetimeIndex
    windspeed: np.ndarray
    direction: np.ndarray
    condition: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Tidy dataframe, DatetimeIndex-based, one row per (realization, timestep)."""
        n_realizations, seq_len = self.windspeed.shape
        frames = []
        for i in range(n_realizations):
            frames.append(
                pd.DataFrame(
                    {
                        "time": self.time,
                        "realization": i,
                        "windspeed_100m": self.windspeed[i],
                        "winddirection_100m": self.direction[i],
                    }
                )
            )
        df = pd.concat(frames, ignore_index=True)
        return df.set_index("time")

    def mean(self) -> np.ndarray:
        """Windspeed mean across realizations, shape (seq_len,)."""
        return self.windspeed.mean(axis=0)

    def percentile_envelope(self, percentiles: tuple[float, float] = (2.5, 97.5)) -> tuple[np.ndarray, np.ndarray]:
        """Windspeed (lower, upper) percentile envelope across realizations,
        each shape (seq_len,)."""
        lo, hi = percentiles
        lower = np.percentile(self.windspeed, lo, axis=0)
        upper = np.percentile(self.windspeed, hi, axis=0)
        return lower, upper


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _build_cond_matrix_for_batch(
    norm_stats_obj: NormStats,
    config,
    wind_regime: str,
    previous_day_windspeed: float | None,
    batch_n: int,
    month: int | None = None,
    day_of_week: int | None = None,
    temp_min: float | None = None,
    temp_max: float | None = None,
) -> tuple[np.ndarray, str, float | None]:
    """Build a (batch_n, cond_dim) condition matrix for one generation batch.

    Factored out of generate_wind_profile so the rejection-sampling loop can
    rebuild a right-sized batch on each retry (see generate_wind_profile's
    rejection_sampling docstring) using the exact same wind-persistence
    default-sampling logic as a single-shot call.

    `month`/`day_of_week`/`temp_min`/`temp_max`: None (default) means "use
    build_condition_vector's own default for this field" -- only overridden
    when the caller passes an explicit value, so existing callers that don't
    pass these keep today's behavior byte-for-byte.

    Returns (cond_matrix, previous_day_windspeed_mode, resolved_previous_day_windspeed)
    -- see generate_wind_profile's metadata fields of the same names.
    """
    persistence_enabled = bool(config.condition_on.get("wind_persistence"))

    condition_kwargs: dict[str, Any] = {"wind_regime": wind_regime}
    if month is not None:
        condition_kwargs["month"] = month
    if day_of_week is not None:
        condition_kwargs["day_of_week"] = day_of_week
    if temp_min is not None:
        condition_kwargs["temp_min"] = temp_min
    if temp_max is not None:
        condition_kwargs["temp_max"] = temp_max

    if not persistence_enabled:
        vec = build_condition_vector(norm_stats_obj, config.condition_on, **condition_kwargs)
        return np.tile(vec[None, :], (batch_n, 1)).astype(np.float32), "disabled", None

    if previous_day_windspeed is not None:
        vec = build_condition_vector(
            norm_stats_obj, config.condition_on,
            previous_day_windspeed=previous_day_windspeed, **condition_kwargs,
        )
        return np.tile(vec[None, :], (batch_n, 1)).astype(np.float32), "explicit", previous_day_windspeed

    regime_history = norm_stats_obj.persistence_by_regime.get(wind_regime) or []
    if not regime_history:
        vec = build_condition_vector(norm_stats_obj, config.condition_on, **condition_kwargs)
        return np.tile(vec[None, :], (batch_n, 1)).astype(np.float32), "disabled", None

    # Sample a DIFFERENT plausible "yesterday" per realization from the real,
    # regime-matched empirical distribution rather than one shared value for
    # the whole batch -- otherwise every synthetic day would be conditioned
    # on the exact same "yesterday," an artificial homogeneity real data
    # doesn't have (real days in a regime each have their own predecessor).
    sampled = np.random.choice(regime_history, size=batch_n)
    base_vector = build_condition_vector(norm_stats_obj, config.condition_on, **condition_kwargs)
    cond_matrix = np.tile(base_vector[None, :], (batch_n, 1)).astype(np.float32)
    p_slice = wind_persistence_slice(config.condition_on, wind_regime_encoding=config.wind_regime_encoding)
    standardized = (sampled - norm_stats_obj.wind_persistence_mean) / norm_stats_obj.wind_persistence_std
    cond_matrix[:, p_slice] = standardized.reshape(-1, 1).astype(np.float32)
    return cond_matrix, "sampled_per_realization", None


def _generate_batch(
    model, schedule, norm_stats_obj, config, wind_regime, previous_day_windspeed, batch_n,
    n_channels, seq_len, clip_min, clip_max, device_obj,
    sampler, steps, eta, guidance_scale, guidance_slice, dynamic_threshold, dynamic_threshold_percentile,
    month=None, day_of_week=None, temp_min=None, temp_max=None,
) -> tuple[np.ndarray, np.ndarray, str, float | None]:
    """Build a condition matrix and generate one batch of `batch_n` synthetic
    days. Returns (windspeed, direction, previous_day_windspeed_mode, resolved_previous_day_windspeed)."""
    cond_matrix, mode, resolved = _build_cond_matrix_for_batch(
        norm_stats_obj, config, wind_regime, previous_day_windspeed, batch_n,
        month=month, day_of_week=day_of_week, temp_min=temp_min, temp_max=temp_max,
    )
    df_gen = generate_multiple_for_condition(
        model, schedule, cond_matrix, batch_n, n_channels, seq_len,
        norm_stats_obj.target_mean, norm_stats_obj.target_std, config.preset,
        clip_min, clip_max, device=device_obj, sampler=sampler, steps=steps, eta=eta,
        guidance_scale=guidance_scale, guidance_slice=guidance_slice,
        dynamic_threshold=dynamic_threshold, dynamic_threshold_percentile=dynamic_threshold_percentile,
    )
    windspeed = df_gen["windspeed_100m"].to_numpy().reshape(batch_n, seq_len)
    direction = df_gen["winddirection_100m"].to_numpy().reshape(batch_n, seq_len)
    return windspeed, direction, mode, resolved


def _generate_with_rejection_sampling(
    model, schedule, norm_stats_obj, config, wind_regime, previous_day_windspeed, n_realizations,
    n_channels, seq_len, clip_min, clip_max, device_obj,
    sampler, steps, eta, guidance_scale, guidance_slice, dynamic_threshold, dynamic_threshold_percentile,
    oversample_factor, max_batches, rank_by_diurnal_shape,
    month=None, day_of_week=None, temp_min=None, temp_max=None,
) -> tuple[np.ndarray, np.ndarray, str, float | None, dict]:
    """Generate more days than requested, keep only the ones whose OWN
    realized daily-mean windspeed actually falls in `wind_regime`'s bucket,
    and repeat until `n_realizations` matches are collected or
    `max_batches` batches have been tried.

    This sidesteps steering the network's conditioning altogether: instead
    of relying on the model to *produce* the requested regime from a weak
    conditioning signal (measured to fail specifically for "mid" --
    reports/MID_REGIME_ANALYSIS.md), it filters after the fact on the one
    thing that actually matters -- whether the generated day's own windspeed
    lands in the right bucket -- so regime match is guaranteed by
    construction rather than hoped for. Cost: extra generate calls (cheap on
    GPU) and, if the model's hit rate for this regime is low and
    max_batches is exhausted first, possibly fewer than n_realizations
    returned (see the returned stats dict's "fully_satisfied").

    `rank_by_diurnal_shape` (opt-in, default off): bucket-matching alone
    says nothing about time-of-day shape, and was measured to sometimes
    badly regress diurnal_rmse (reports/REGIME_REJECTION_SAMPLING_INVESTIGATION.md
    -- "low" got ~4.7x worse). Enabling this ranks bucket-matched candidates
    by RMSE distance to the regime's real reference diurnal profile
    (`NormStats.diurnal_profile_by_regime`, precomputed from TRAIN data) and
    keeps only the closest-shaped `n_realizations`. Measured: this DOES fix
    diurnal_rmse substantially, but is NOT a free win -- selecting for
    closest-to-average shape reduces the selected population's diversity,
    which measurably regresses spectral fidelity for mid/high (worse than
    even plain rejection sampling, not just worse than baseline) -- see
    reports/REJECTION_SAMPLING_DIURNAL_RANKING_ADDENDUM.md. Off by default
    so `rejection_sampling=True` alone keeps its already-reported behavior;
    this is a separate, deliberate tradeoff a caller opts into.
    """
    matched_windspeed: list[np.ndarray] = []
    matched_direction: list[np.ndarray] = []
    n_generated_total = 0
    n_matched_total = 0
    previous_day_windspeed_mode = "disabled"
    resolved_previous_day_windspeed = previous_day_windspeed

    for _ in range(max_batches):
        n_have = sum(w.shape[0] for w in matched_windspeed)
        n_needed = n_realizations - n_have
        if n_needed <= 0:
            break
        batch_n = max(int(np.ceil(n_needed * oversample_factor)), n_needed)

        windspeed, direction, previous_day_windspeed_mode, resolved_previous_day_windspeed = _generate_batch(
            model, schedule, norm_stats_obj, config, wind_regime, previous_day_windspeed, batch_n,
            n_channels, seq_len, clip_min, clip_max, device_obj,
            sampler, steps, eta, guidance_scale, guidance_slice, dynamic_threshold, dynamic_threshold_percentile,
            month=month, day_of_week=day_of_week, temp_min=temp_min, temp_max=temp_max,
        )
        daily_means = windspeed.mean(axis=1)
        buckets = np.array([assign_wind_bucket(float(m), *norm_stats_obj.wind_bucket_edges) for m in daily_means])
        mask = buckets == wind_regime

        n_generated_total += batch_n
        n_matched_total += int(mask.sum())
        matched_windspeed.append(windspeed[mask])
        matched_direction.append(direction[mask])

    windspeed_all = np.concatenate(matched_windspeed, axis=0) if matched_windspeed else np.empty((0, seq_len))
    direction_all = np.concatenate(matched_direction, axis=0) if matched_direction else np.empty((0, seq_len))
    fully_satisfied = windspeed_all.shape[0] >= n_realizations

    reference_profile = norm_stats_obj.diurnal_profile_by_regime.get(wind_regime) if rank_by_diurnal_shape else None
    diurnal_shape_ranking_applied = bool(reference_profile) and windspeed_all.shape[0] > n_realizations
    if diurnal_shape_ranking_applied:
        reference = np.asarray(reference_profile)
        distances = np.sqrt(((windspeed_all - reference[None, :]) ** 2).mean(axis=1))
        order = np.argsort(distances)
        windspeed_all = windspeed_all[order]
        direction_all = direction_all[order]

    windspeed_all = windspeed_all[:n_realizations]
    direction_all = direction_all[:n_realizations]

    stats = {
        "n_generated_total": n_generated_total,
        "n_matched_total": n_matched_total,
        "diurnal_shape_ranking_applied": diurnal_shape_ranking_applied,
        "hit_rate": n_matched_total / n_generated_total if n_generated_total else 0.0,
        "n_returned": windspeed_all.shape[0],
        "fully_satisfied": fully_satisfied,
    }
    return windspeed_all, direction_all, previous_day_windspeed_mode, resolved_previous_day_windspeed, stats


def generate_wind_profile(
    channel_method: PresetName = "3ch",
    wind_regime: str | None = None,
    *,
    n_realizations: int = 100,
    date: str | None = None,
    sampler: str = "ddim",
    steps: int = 50,
    checkpoint: str | Path | None = None,
    norm_stats: str | Path | None = None,
    seed: int | None = None,
    device: str = "auto",
    eta: float = 0.0,
    guidance_scale: float = 1.0,
    guidance_scope: str = "full",
    dynamic_threshold: bool = False,
    dynamic_threshold_percentile: float = 99.0,
    previous_day_windspeed: float | None = None,
    rejection_sampling: bool = False,
    rejection_oversample_factor: float = 6.0,
    rejection_max_batches: int = 10,
    rejection_rank_by_diurnal_shape: bool = False,
    month: int | None = None,
    day_of_week: int | None = None,
    temp_min: float | None = None,
    temp_max: float | None = None,
) -> WindProfile:
    """Generate `n_realizations` synthetic daily wind profiles for one
    condition, from a trained wind_diffusion checkpoint.

    month/day_of_week/temp_min/temp_max: optional conditioning overrides,
        passed straight through to `evaluate.build_condition_vector` (which
        already accepts them). None (default, for any of the four) means
        "use build_condition_vector's own default for that field" --
        `month=6, day_of_week=0, temp_min=10.0, temp_max=22.0` -- so a call
        that doesn't set these keeps generating exactly what it always did.
        `month` must be one of the checkpoint's `NormStats.months` and
        `day_of_week` one of `NormStats.days_of_week`, or `build_condition_vector`
        raises (via `list.index`) same as an out-of-range value always has.

    dynamic_threshold: replaces the static per-channel x0 clip with
        Imagen-style dynamic thresholding (see `diffusion.clamp_x0`).
        False (default) = identical to every prior release's behavior.
        Measured to fix a real problem present in every preset: hard
        clamping creates a self-reinforcing lock-in where a position, once
        clamped, has a ~94% chance of still being clamped many steps later
        -- some positions stay pinned at the exact clip boundary for hours
        of simulated time, inflating variance ~2x and making the
        autocorrelation decay far faster than real data's. Unlike guidance,
        this is a pure generation-time change with no training dependency,
        so it's safe to enable against any existing checkpoint.

    guidance_scale/guidance_scope: classifier-free guidance scale and scope
        (see `diffusion._cfg_predict`). guidance_scale=1.0 (default) = no
        guidance, unchanged behavior. guidance_scope="full" (default) guides
        the entire condition vector; guidance_scope="wind_regime" guides
        only the 3-dim wind-regime one-hot slice, leaving month/day-of-week/
        temp untouched in the extrapolation -- measured to matter: "full"
        amplifies whatever the whole vector's net effect already is
        (dominated by month+day-of-week) rather than the regime signal
        specifically, and made regime steering worse in practice (see
        reports/BASELINE_1CH_CFG_GS*.md). Guidance is only meaningful
        against a checkpoint trained with `cond_dropout_prob > 0` in the
        matching scope (see `enerdatasynth.resources.wind.training.train`); otherwise the zeroed
        slice is out-of-distribution for the model and the result is not
        meaningful.

    previous_day_windspeed: optional raw daily-mean windspeed (m/s) for
        "yesterday," only meaningful against a checkpoint trained with
        `condition_on["wind_persistence"]` enabled (see config.py). Purely
        optional, by design: if the checkpoint doesn't use this feature,
        it's ignored; if it does and no value is supplied, one is sampled
        from the real, regime-matched empirical distribution of previous-day
        values seen in training (`NormStats.persistence_by_regime`) rather
        than a fixed placeholder -- so the common call
        `generate_wind_profile(preset, wind_regime=...)` keeps working
        unchanged and still gets a realistic (not neutral/zeroed) previous-
        day value under the hood.

    rejection_sampling: instead of relying solely on the model's own
        conditioning to produce the requested regime, generate more days
        than asked for and keep only the ones whose OWN realized daily-mean
        windspeed actually lands in `wind_regime`'s bucket -- regime match
        becomes guaranteed by construction rather than dependent on how well
        the network's conditioning steers, which is measured to be weak
        specifically for "mid" (see reports/MID_REGIME_ANALYSIS.md). False
        (default) = unchanged prior behavior. When True, may return FEWER
        than `n_realizations` days if `rejection_max_batches` is exhausted
        before enough matches are found (rare in practice -- check
        `metadata["rejection_stats"]["fully_satisfied"]`). Costs extra
        generate calls (cheap on GPU) proportional to
        `rejection_oversample_factor` divided by the regime's actual hit
        rate under the given generation settings.

    rejection_rank_by_diurnal_shape: only meaningful when rejection_sampling=True.
        False (default) keeps rejection_sampling's originally-measured
        behavior (bucket-matched candidates kept in generation order). True
        ranks bucket-matched candidates by closeness to the regime's real
        reference diurnal (time-of-day) shape and keeps only the
        closest-shaped ones -- this substantially fixes diurnal_rmse but is
        NOT a free win: it reduces the selected population's diversity,
        which measurably regresses spectral fidelity for mid/high. See
        reports/REJECTION_SAMPLING_DIURNAL_RANKING_ADDENDUM.md before
        enabling this by default for a given use case.

    Weights are resolved via `enerdatasynth.resources.wind.registry.resolve_weights`
    (explicit checkpoint=/norm_stats= args > WIND_DIFFUSION_CHECKPOINTS env
    dir > cached download from the manifest > CheckpointNotFoundError).
    `wind_regime` is validated against the checkpoint's own
    `NormStats.wind_buckets` (typically ["low", "mid", "high"]) before any
    model loading happens; an unrecognized regime raises
    `InvalidConditionError`.

    Raises:
        CheckpointNotFoundError: no usable checkpoint/norm_stats found.
        InvalidConditionError: `wind_regime` not in the checkpoint's wind_buckets.
    """
    start = time.time()

    ckpt_path, norm_path = resolve_weights(channel_method, checkpoint=checkpoint, norm_stats=norm_stats)

    with open(norm_path) as f:
        norm_stats_obj = NormStats.from_dict(json.load(f))

    if wind_regime is None:
        wind_regime = (
            _DEFAULT_WIND_REGIME_FALLBACK
            if _DEFAULT_WIND_REGIME_FALLBACK in norm_stats_obj.wind_buckets
            else norm_stats_obj.wind_buckets[0]
        )
    if wind_regime not in norm_stats_obj.wind_buckets:
        raise InvalidConditionError(
            f"wind_regime={wind_regime!r} is not recognized for preset {channel_method!r}; "
            f"expected one of {norm_stats_obj.wind_buckets}."
        )

    if seed is not None:
        set_seed(seed)

    device_obj = resolve_device(device)
    # Pass channel_method through as an explicit preset override: the
    # caller has already told us which preset they want (and resolve_weights
    # above resolved checkpoint/norm_stats specifically for it), so the
    # checkpoint's own saved config should always be interpreted under that
    # preset rather than whatever get_cached_model/load_checkpoint_and_model
    # would otherwise default to for a legacy checkpoint with no "preset"
    # key in its saved config (see evaluate.load_checkpoint_and_model).
    model, config, checkpoint_dict = get_cached_model(str(ckpt_path), device=device_obj, preset=channel_method)

    schedule = DiffusionSchedule.build(
        T=config.diffusion_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        device=device_obj,
    )

    guidance_slice = (
        wind_regime_slice(config.condition_on, wind_regime_encoding=config.wind_regime_encoding)
        if guidance_scope == "wind_regime"
        else None
    )

    n_channels = config.n_model_channels
    seq_len = config.seq_len
    clip_min = torch.full((1, n_channels, 1), -config.sample_clip_std, dtype=torch.float32, device=device_obj)
    clip_max = torch.full((1, n_channels, 1), config.sample_clip_std, dtype=torch.float32, device=device_obj)

    rejection_stats = None
    if rejection_sampling:
        windspeed, direction, previous_day_windspeed_mode, resolved_previous_day_windspeed, rejection_stats = (
            _generate_with_rejection_sampling(
                model, schedule, norm_stats_obj, config, wind_regime, previous_day_windspeed, n_realizations,
                n_channels, seq_len, clip_min, clip_max, device_obj,
                sampler, steps, eta, guidance_scale, guidance_slice, dynamic_threshold, dynamic_threshold_percentile,
                rejection_oversample_factor, rejection_max_batches, rejection_rank_by_diurnal_shape,
                month=month, day_of_week=day_of_week, temp_min=temp_min, temp_max=temp_max,
            )
        )
    else:
        windspeed, direction, previous_day_windspeed_mode, resolved_previous_day_windspeed = _generate_batch(
            model, schedule, norm_stats_obj, config, wind_regime, previous_day_windspeed, n_realizations,
            n_channels, seq_len, clip_min, clip_max, device_obj,
            sampler, steps, eta, guidance_scale, guidance_slice, dynamic_threshold, dynamic_threshold_percentile,
            month=month, day_of_week=day_of_week, temp_min=temp_min, temp_max=temp_max,
        )

    anchor = pd.Timestamp(date) if date else pd.Timestamp.today()
    anchor = anchor.normalize()
    time_index = pd.date_range(anchor, periods=seq_len, freq="5min")

    wall_time = time.time() - start

    condition = {
        "wind_regime": wind_regime,
        "wind_bucket_edges": norm_stats_obj.wind_bucket_edges,
        "month": month,
        "day_of_week": day_of_week,
        "temp_min": temp_min,
        "temp_max": temp_max,
    }
    metadata = {
        "preset": channel_method,
        "sampler": sampler,
        "steps": steps,
        "checkpoint_path": str(ckpt_path),
        "norm_stats_path": str(norm_path),
        "checkpoint_epoch": checkpoint_dict.get("epoch"),
        "checkpoint_val_loss": checkpoint_dict.get("val_loss"),
        "device": str(device_obj),
        "wall_time_seconds": wall_time,
        "n_realizations": n_realizations,
        "n_realizations_returned": windspeed.shape[0],
        "seed": seed,
        "eta": eta,
        "guidance_scale": guidance_scale,
        "guidance_scope": guidance_scope,
        "dynamic_threshold": dynamic_threshold,
        "dynamic_threshold_percentile": dynamic_threshold_percentile,
        "previous_day_windspeed": resolved_previous_day_windspeed,
        "previous_day_windspeed_mode": previous_day_windspeed_mode,
        "rejection_sampling": rejection_sampling,
        "rejection_rank_by_diurnal_shape": rejection_rank_by_diurnal_shape,
        "rejection_stats": rejection_stats,
    }

    return WindProfile(
        time=time_index,
        windspeed=windspeed,
        direction=direction,
        condition=condition,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Registry passthroughs
# ---------------------------------------------------------------------------


def list_models() -> dict:
    """List all presets known to the bundled manifest with their status/metrics."""
    return _registry_list_models()


def describe_model(preset: PresetName) -> dict:
    """Describe a single preset's manifest entry plus local cache-state info."""
    return _registry_describe_model(preset)


def preload(
    preset: PresetName,
    checkpoint: str | Path | None = None,
    norm_stats: str | Path | None = None,
    device: str = "auto",
) -> None:
    """Resolve weights for `preset` and warm the process-level model cache."""
    ckpt_path, _norm_path = resolve_weights(preset, checkpoint=checkpoint, norm_stats=norm_stats)
    _evaluate_preload(str(ckpt_path), device=device, preset=preset)


# ---------------------------------------------------------------------------
# Console entry point: wind-diffusion-generate
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic wind profiles from a trained wind_diffusion checkpoint.")
    parser.add_argument("--channel-method", choices=["1ch", "2ch", "3ch"], default="3ch")
    parser.add_argument("--wind-regime", default=None)
    parser.add_argument("--n-realizations", type=int, default=100)
    parser.add_argument("--date", default=None)
    parser.add_argument("--sampler", choices=["ancestral", "ddim"], default="ddim")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--norm-stats", dest="norm_stats", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument(
        "--guidance-scale", type=float, default=1.0,
        help="Classifier-free guidance scale. 1.0 (default) = no guidance. Only meaningful against a checkpoint trained with cond_dropout_prob > 0.",
    )
    parser.add_argument(
        "--guidance-scope", choices=["full", "wind_regime"], default="full",
        help='Which part of the condition vector guidance_scale amplifies: "full" (default) or "wind_regime" (only the regime one-hot slice).',
    )
    parser.add_argument(
        "--dynamic-threshold", action="store_true",
        help="Use dynamic (percentile-based) thresholding instead of a static x0 clip. Fixes a measured clip-boundary lock-in; safe against any checkpoint.",
    )
    parser.add_argument("--dynamic-threshold-percentile", type=float, default=99.0)
    parser.add_argument(
        "--previous-day-windspeed", type=float, default=None,
        help="Yesterday's actual daily-mean windspeed (m/s), only meaningful against a checkpoint trained with wind_persistence conditioning. Default: sampled from the regime-matched empirical distribution if the checkpoint uses this feature, else ignored.",
    )
    parser.add_argument(
        "--rejection-sampling", action="store_true",
        help="Generate more days than requested and keep only the ones whose own realized windspeed lands in wind_regime's bucket, guaranteeing regime match by construction instead of relying on the model's conditioning alone.",
    )
    parser.add_argument("--rejection-oversample-factor", type=float, default=6.0)
    parser.add_argument("--rejection-max-batches", type=int, default=10)
    parser.add_argument(
        "--rejection-rank-by-diurnal-shape", action="store_true",
        help="Only meaningful with --rejection-sampling. Ranks bucket-matched candidates by closeness to the regime's real diurnal shape and keeps the closest -- fixes diurnal_rmse but regresses spectral fidelity for mid/high (a real tradeoff, not a free win; see reports/REJECTION_SAMPLING_DIURNAL_RANKING_ADDENDUM.md).",
    )
    parser.add_argument(
        "--month", type=int, default=None,
        help="Conditioning month override (1-12). Default: build_condition_vector's own default (6) if omitted.",
    )
    parser.add_argument(
        "--day-of-week", dest="day_of_week", type=int, default=None,
        help="Conditioning day-of-week override (0=Monday..6=Sunday). Default: build_condition_vector's own default (0) if omitted.",
    )
    parser.add_argument(
        "--temp-min", dest="temp_min", type=float, default=None,
        help="Conditioning daily min temperature override (degC). Default: build_condition_vector's own default (10.0) if omitted.",
    )
    parser.add_argument(
        "--temp-max", dest="temp_max", type=float, default=None,
        help="Conditioning daily max temperature override (degC). Default: build_condition_vector's own default (22.0) if omitted.",
    )
    parser.add_argument("--output-dir", dest="output_dir", required=True)

    args = parser.parse_args(argv)

    profile = generate_wind_profile(
        channel_method=args.channel_method,
        wind_regime=args.wind_regime,
        n_realizations=args.n_realizations,
        date=args.date,
        sampler=args.sampler,
        steps=args.steps,
        checkpoint=args.checkpoint,
        norm_stats=args.norm_stats,
        seed=args.seed,
        device=args.device,
        eta=args.eta,
        guidance_scale=args.guidance_scale,
        guidance_scope=args.guidance_scope,
        dynamic_threshold=args.dynamic_threshold,
        dynamic_threshold_percentile=args.dynamic_threshold_percentile,
        previous_day_windspeed=args.previous_day_windspeed,
        rejection_sampling=args.rejection_sampling,
        rejection_oversample_factor=args.rejection_oversample_factor,
        rejection_max_batches=args.rejection_max_batches,
        rejection_rank_by_diurnal_shape=args.rejection_rank_by_diurnal_shape,
        month=args.month,
        day_of_week=args.day_of_week,
        temp_min=args.temp_min,
        temp_max=args.temp_max,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    df = profile.to_dataframe()
    out_path = os.path.join(args.output_dir, "wind_profile.csv")
    df.to_csv(out_path)
    print(f"Wrote {len(df)} rows ({args.n_realizations} realizations x {profile.windspeed.shape[1]} steps) to {out_path}")
    print(f"metadata: {json.dumps(profile.metadata, default=str)}")


if __name__ == "__main__":
    main()
