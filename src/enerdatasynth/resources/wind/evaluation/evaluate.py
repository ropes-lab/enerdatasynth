"""Reconstruction, generation helpers, evaluation plots, and a console-script
entry point for sampling from a trained wind_diffusion checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import threading

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from enerdatasynth.resources.wind.config import PresetName, WindDiffusionConfig, get_config
from enerdatasynth.resources.wind.data.data import NormStats
from enerdatasynth.resources.wind.diffusion.diffusion import DiffusionSchedule
from enerdatasynth.resources.wind.models.models import TransformerDenoiser
from enerdatasynth.resources.wind.diffusion.samplers import sample as sampler_sample
from enerdatasynth.resources.wind.training.train import build_model
from enerdatasynth.resources.wind.data.transforms import ORDINAL_WIND_BUCKET_VALUES, decode_channels, destandardize_windows
from enerdatasynth.resources.wind.utils import resolve_device

sns.set_theme(style="whitegrid")

WIND_BUCKET_LABELS = ["low", "mid", "high"]


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def reconstruct_to_dataframe(
    samples: np.ndarray,
    seq_len: int,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    preset: PresetName,
) -> pd.DataFrame:
    """Convert standardized (N, channels, seq_len) samples into a tidy
    physical-units dataframe (windspeed_100m, winddirection_100m per step),
    decoding channels according to `preset`.
    """
    n_samples = samples.shape[0]
    samples = samples.transpose(0, 2, 1)  # (N, seq_len, channels)
    samples = destandardize_windows(samples, target_mean, target_std)

    speed, direction = decode_channels(samples, preset)  # each (N, seq_len)

    rows = []
    for i in range(n_samples):
        rows.append(
            pd.DataFrame(
                {
                    "synthetic_day_id": i,
                    "step_in_day": np.arange(seq_len),
                    "windspeed_100m": speed[i],
                    "winddirection_100m": direction[i],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------


def autocorr(series: np.ndarray, max_lag: int) -> np.ndarray:
    series = series - series.mean()
    acf = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf.append(1.0)
        else:
            num = np.sum(series[:-lag] * series[lag:])
            den = np.sum(series**2)
            acf.append(num / den if den != 0 else 0.0)
    return np.array(acf)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def generate_multiple_for_condition(
    model: TransformerDenoiser,
    schedule: DiffusionSchedule,
    cond_vector: np.ndarray,
    n_samples: int,
    n_channels: int,
    seq_len: int,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    preset: PresetName,
    clip_min: torch.Tensor,
    clip_max: torch.Tensor,
    device: str | torch.device = "cpu",
    sampler: str = "ddim",
    steps: int = 50,
    eta: float = 0.0,
    guidance_scale: float = 1.0,
    guidance_slice: slice | None = None,
    dynamic_threshold: bool = False,
    dynamic_threshold_percentile: float = 99.0,
) -> pd.DataFrame:
    """Generate n_samples independent synthetic days for a condition vector.

    `cond_vector` is normally 1D (cond_dim,) and gets tiled to n_samples
    identical rows. It may also be passed as a full 2D (n_samples, cond_dim)
    matrix when different realizations need to differ in part of their
    condition -- e.g. api.generate_wind_profile samples a distinct
    wind-persistence value per realization from the empirical distribution
    rather than sharing one fixed value across the whole batch, so the
    synthetic population's previous-day diversity matches real data's
    instead of every realization being conditioned on the same "yesterday."
    """
    if cond_vector.ndim == 2:
        cond_matrix = cond_vector
    else:
        cond_matrix = np.tile(cond_vector[None, :], (n_samples, 1))
    samples_norm = sampler_sample(
        model,
        schedule,
        cond_matrix,
        n_channels,
        seq_len,
        clip_min,
        clip_max,
        device=device,
        sampler=sampler,
        steps=steps,
        eta=eta,
        guidance_scale=guidance_scale,
        guidance_slice=guidance_slice,
        dynamic_threshold=dynamic_threshold,
        dynamic_threshold_percentile=dynamic_threshold_percentile,
    )
    return reconstruct_to_dataframe(samples_norm, seq_len, target_mean, target_std, preset)


def build_condition_vector(
    norm_stats: NormStats,
    condition_on: dict[str, bool],
    month: int = 6,
    wind_regime: str = "mid",
    temp_min: float = 10.0,
    temp_max: float = 22.0,
    day_of_week: int = 0,
    previous_day_windspeed: float | None = None,
) -> np.ndarray:
    """Build a single condition vector z from human-readable toggles. Mirrors
    the condition-vector construction inside the original
    `generate_interactive`.

    `previous_day_windspeed`: raw physical daily-mean windspeed (m/s) for
    "yesterday," standardized here using `norm_stats.wind_persistence_mean`/
    `wind_persistence_std`. Only meaningful when
    `condition_on["wind_persistence"]` is enabled; if that's on and no value
    is given, falls back to the neutral "no info" value (0.0 post-
    standardization, matching training-time dropout) -- callers that want a
    realistic (not neutral) default should sample one from
    `norm_stats.persistence_by_regime[wind_regime]` themselves (see
    api.generate_wind_profile).
    """
    parts = []

    if condition_on.get("month"):
        m_oh = np.zeros(len(norm_stats.months), dtype=np.float32)
        m_oh[norm_stats.months.index(month)] = 1.0
        parts.append(m_oh)

    if condition_on.get("day_of_week"):
        dow_oh = np.zeros(len(norm_stats.days_of_week), dtype=np.float32)
        dow_oh[norm_stats.days_of_week.index(day_of_week)] = 1.0
        parts.append(dow_oh)

    if condition_on.get("wind_regime_bucket"):
        if norm_stats.wind_regime_encoding == "ordinal":
            parts.append(np.array([ORDINAL_WIND_BUCKET_VALUES[wind_regime]], dtype=np.float32))
        else:
            w_oh = np.zeros(len(norm_stats.wind_buckets), dtype=np.float32)
            w_oh[norm_stats.wind_buckets.index(wind_regime)] = 1.0
            parts.append(w_oh)

    if condition_on.get("temp_min_max_spread"):
        t_spread = temp_max - temp_min
        numeric = np.array([temp_max, temp_min, t_spread], dtype=np.float32)
        numeric_norm = (numeric - norm_stats.numeric_cond_mean) / norm_stats.numeric_cond_std
        parts.append(numeric_norm.astype(np.float32))

    if condition_on.get("wind_persistence"):
        if previous_day_windspeed is None:
            parts.append(np.array([0.0], dtype=np.float32))
        else:
            standardized = (previous_day_windspeed - norm_stats.wind_persistence_mean) / norm_stats.wind_persistence_std
            parts.append(np.array([standardized], dtype=np.float32))

    return np.concatenate(parts)


def generate_interactive(
    model: TransformerDenoiser,
    schedule: DiffusionSchedule,
    norm_stats: NormStats,
    condition_on: dict[str, bool],
    preset: PresetName,
    clip_min: torch.Tensor,
    clip_max: torch.Tensor,
    seq_len: int = 288,
    device: str | torch.device = "cpu",
    month: int = 6,
    wind_regime: str = "mid",
    temp_min: float = 10.0,
    temp_max: float = 22.0,
    plot: bool = True,
    sampler: str = "ddim",
    steps: int = 50,
    eta: float = 0.0,
    previous_day_windspeed: float | None = None,
) -> pd.DataFrame:
    """Generate a synthetic day of wind profiles based on toggled condition
    values, optionally plotting the result.
    """
    z = build_condition_vector(
        norm_stats, condition_on, month=month, wind_regime=wind_regime, temp_min=temp_min, temp_max=temp_max,
        previous_day_windspeed=previous_day_windspeed,
    )[None, :]

    n_channels = len(norm_stats.model_channels)
    sample_norm = sampler_sample(
        model,
        schedule,
        z,
        n_channels,
        seq_len,
        clip_min,
        clip_max,
        device=device,
        sampler=sampler,
        steps=steps,
        eta=eta,
    )

    df_gen = reconstruct_to_dataframe(
        sample_norm, seq_len, norm_stats.target_mean, norm_stats.target_std, preset
    )

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        steps = df_gen["step_in_day"]

        axes[0].plot(steps, df_gen["windspeed_100m"], color="tab:blue", lw=2)
        axes[0].set_ylabel("Wind Speed 100m (m/s)", fontsize=11)
        axes[0].set_title(
            f"Generated Wind Speed (Daily Mean Regime: {wind_regime.upper()})", fontsize=12, fontweight="bold"
        )

        axes[1].plot(
            steps, df_gen["winddirection_100m"], color="tab:orange", lw=1.5, ls="none", marker="o", ms=3, alpha=0.7
        )
        axes[1].set_ylabel("Wind Direction 100m (deg)", fontsize=11)
        axes[1].set_ylim(0, 360)

        for ax in axes:
            ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()

    return df_gen


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_loss_curve(history_df: pd.DataFrame, save_path: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss (noise prediction)")
    plt.title("Training / validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_eval_distributions(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axes, ["windspeed_100m", "winddirection_100m"]):
        sns.histplot(real_df[col], stat="density", color="tab:blue", alpha=0.5, label="real", ax=ax, bins=40)
        sns.histplot(synthetic_df[col], stat="density", color="tab:orange", alpha=0.5, label="synthetic", ax=ax, bins=40)
        ax.set_title(col)
        ax.legend()
    plt.suptitle("Hourly-level value distributions: real vs synthetic")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_eval_first_diff(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axes, ["windspeed_100m", "winddirection_100m"]):
        real_diff = real_df.groupby("synthetic_day_id")[col].diff().dropna()
        synth_diff = synthetic_df.groupby("synthetic_day_id")[col].diff().dropna()
        sns.histplot(real_diff, stat="density", color="tab:blue", alpha=0.5, label="real", ax=ax, bins=40)
        sns.histplot(synth_diff, stat="density", color="tab:orange", alpha=0.5, label="synthetic", ax=ax, bins=40)
        ax.set_title(f"{col} (first difference)")
        ax.legend()
    plt.suptitle("First-difference distributions: real vs synthetic")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_eval_acf(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, save_path: str, max_lag: int = 36) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col in zip(axes, ["windspeed_100m", "winddirection_100m"]):
        real_acfs = [autocorr(day_df[col].values, max_lag) for _, day_df in real_df.groupby("synthetic_day_id")]
        synth_acfs = [
            autocorr(day_df[col].values, max_lag) for _, day_df in synthetic_df.groupby("synthetic_day_id")
        ]

        ax.plot(np.mean(real_acfs, axis=0), label="real", color="tab:blue")
        ax.plot(np.mean(synth_acfs, axis=0), label="synthetic", color="tab:orange")
        ax.set_title(f"{col} ACF (mean over days)")
        ax.set_xlabel("lag (5-min steps)")
        ax.legend()
    plt.suptitle("Autocorrelation: real vs synthetic")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_sanity_one_day(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, save_path: str) -> None:
    real_day0 = real_df[real_df["synthetic_day_id"] == 0]
    synth_day0 = synthetic_df[synthetic_df["synthetic_day_id"] == 0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, col in zip(axes, ["windspeed_100m", "winddirection_100m"]):
        ax.plot(real_day0["step_in_day"], real_day0[col], label="real", color="tab:blue")
        ax.plot(synth_day0["step_in_day"], synth_day0[col], label="synthetic", color="tab:orange")
        ax.set_ylabel(col)
        if col == "winddirection_100m":
            ax.set_ylim(0, 360)
        elif col == "windspeed_100m":
            ax.set_ylim(0, max(real_day0[col].max() * 1.2, 20))
        ax.legend()
    axes[-1].set_xlabel("step in day (5-min)")
    plt.suptitle("Sanity check: one real day vs one synthetic day")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_conditioning_wind_bucket(
    real_df: pd.DataFrame, synthetic_df: pd.DataFrame, gen_cond_df: pd.DataFrame, save_path: str
) -> None:
    real_daily_mean = real_df.groupby("synthetic_day_id")["windspeed_100m"].mean()
    synth_daily_mean = synthetic_df.groupby("synthetic_day_id")["windspeed_100m"].mean()

    diag_df = pd.DataFrame(
        {
            "real_mean_windspeed": real_daily_mean,
            "synth_mean_windspeed": synth_daily_mean,
            "wind_bucket": gen_cond_df["wind_bucket"].values,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for ax, bucket in zip(axes, WIND_BUCKET_LABELS):
        bucket_df = diag_df[diag_df["wind_bucket"] == bucket]
        sns.histplot(
            bucket_df["real_mean_windspeed"], stat="density", color="tab:blue", alpha=0.5, label="real", ax=ax,
            bins=15, kde=True,
        )
        sns.histplot(
            bucket_df["synth_mean_windspeed"], stat="density", color="tab:orange", alpha=0.5, label="synthetic",
            ax=ax, bins=15, kde=True,
        )
        ax.set_title(f"wind_bucket = {bucket} (n={len(bucket_df)})")
        ax.set_xlabel("daily mean windspeed_100m")
        ax.legend()
    plt.suptitle("Conditioning check: daily mean windspeed by wind-regime bucket")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Console entry point
# ---------------------------------------------------------------------------


def load_checkpoint_and_model(
    checkpoint_path: str,
    device: str | torch.device = "cpu",
    preset: PresetName | None = None,
):
    """Load a checkpoint and rebuild its model.

    `preset` lets a caller override which preset the checkpoint's saved
    config is interpreted as. This matters for legacy checkpoints saved by
    the original monolithic scripts, whose `checkpoint["config"]` dict has
    no `"preset"` key at all (only the newer package-produced checkpoints
    save one) -- without an override, those silently default to "2ch"
    below, which is wrong for e.g. a 1-channel legacy checkpoint and causes
    a model/state-dict shape mismatch at `load_state_dict`.

    When `preset` is None (default), behavior is unchanged from before:
    `config_dict.get("preset", "2ch")` is used.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = checkpoint["config"]
    resolved_preset: PresetName = preset if preset is not None else config_dict.get("preset", "2ch")
    config = get_config(
        resolved_preset, **{k: v for k, v in config_dict.items() if k not in ("preset", "model_channels")}
    )

    model = build_model(config, config.n_model_channels, cond_dim=_infer_cond_dim(checkpoint), seq_len=config.seq_len)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, checkpoint


def _infer_cond_dim(checkpoint: dict) -> int:
    """Infer cond_dim from the saved state dict's cond_mlp input layer shape."""
    key = "cond_mlp.net.0.weight"
    if key in checkpoint["model_state_dict"]:
        return int(checkpoint["model_state_dict"][key].shape[1])
    raise ValueError("Could not infer cond_dim from checkpoint state dict.")


# ---------------------------------------------------------------------------
# Process-level model cache
# ---------------------------------------------------------------------------
#
# Loading a checkpoint (torch.load + rebuilding the TransformerDenoiser)
# takes non-trivial time and is unnecessary to repeat across calls that
# target the same (checkpoint_path, device). This cache makes repeated
# calls -- e.g. from a long-lived server process -- pay that cost once.

_MODEL_CACHE: dict[tuple[str, str, str | None], tuple] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def get_cached_model(checkpoint_path: str, device: str | torch.device = "auto", preset: PresetName | None = None):
    """Like `load_checkpoint_and_model`, but memoized per (checkpoint_path,
    resolved device, preset) for the lifetime of the process. Thread-safe.

    `preset` is forwarded to `load_checkpoint_and_model` (see its docstring
    for why this matters for legacy checkpoints with no saved "preset"
    key). It's included in the cache key so a caller that explicitly
    overrides the preset never silently gets back a model built under a
    different (e.g. default-"2ch") preset from an earlier call with the
    same checkpoint_path/device.
    """
    device_obj = device if isinstance(device, torch.device) else resolve_device(device)
    key = (checkpoint_path, str(device_obj), preset)

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

    # Load outside the lock to avoid blocking other keys on a slow disk
    # read; re-check under the lock before storing to avoid a duplicate
    # load winning a race (last-writer-wins is fine here, but this reuses
    # whichever result won the race instead of any wasted result).
    result = load_checkpoint_and_model(checkpoint_path, device=device_obj, preset=preset)

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        _MODEL_CACHE[key] = result
        return result


def preload(checkpoint_path: str, device: str | torch.device = "auto", preset: PresetName | None = None) -> None:
    """Warm the model cache for `checkpoint_path`/`device`/`preset` ahead of time."""
    get_cached_model(checkpoint_path, device=device, preset=preset)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate + evaluate samples from a trained wind_diffusion checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm-stats", dest="norm_stats", required=True)
    parser.add_argument("--output-dir", dest="output_dir", required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sampler", choices=["ancestral", "ddim"], default="ddim")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)

    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    model, config, checkpoint = get_cached_model(args.checkpoint, device=device)

    with open(args.norm_stats) as f:
        norm_stats = NormStats.from_dict(json.load(f))

    schedule = DiffusionSchedule.build(
        T=config.diffusion_timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        beta_schedule=config.beta_schedule,
        device=device,
    )

    n_channels = config.n_model_channels
    clip_min = torch.tensor([-config.sample_clip_std] * n_channels, dtype=torch.float32, device=device).view(1, n_channels, 1)
    clip_max = torch.tensor([config.sample_clip_std] * n_channels, dtype=torch.float32, device=device).view(1, n_channels, 1)

    df_gen = generate_interactive(
        model,
        schedule,
        norm_stats,
        config.condition_on,
        config.preset,
        clip_min,
        clip_max,
        seq_len=config.seq_len,
        device=device,
        plot=False,
        sampler=args.sampler,
        steps=args.steps,
        eta=args.eta,
    )
    out_path = os.path.join(args.output_dir, "generated_sample.csv")
    df_gen.to_csv(out_path, index=False)
    print(f"Wrote {len(df_gen)} rows to {out_path}")


if __name__ == "__main__":
    main()
