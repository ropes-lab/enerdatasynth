"""Comparison plotting suite: generated ("synthetic") windspeed days vs real
held-out windspeed days.

Deliberately independent of `evaluate.py`'s older `plot_eval_*` functions,
which expect a two-column (windspeed + direction) tidy dataframe in the
`synthetic_day_id`/`step_in_day` shape produced by `reconstruct_to_dataframe`.
Those stay as-is (still used by the `wind-diffusion-evaluate` CLI). This
module instead works directly on raw `(N, seq_len)` windspeed arrays -- the
same convention `metrics.py` already uses -- so it applies uniformly across
all three presets (1ch has no meaningful direction channel at all; 2ch/3ch
do, but a windspeed-only comparison is still the common ground across every
preset and is what the fidelity gates in `metrics.py` actually measure).

Every function is loss-free with respect to `metrics.py`: same underlying
`evaluate.autocorr`, no new statistics invented here, just the visual view
of the same numbers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from wind_diffusion.evaluate import autocorr

sns.set_theme(style="whitegrid")


def plot_distributions(real_days: np.ndarray, synth_days: np.ndarray, save_path: str | Path) -> None:
    """Marginal value distribution: real vs synthetic, all timesteps pooled."""
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(real_days.ravel(), stat="density", color="tab:blue", alpha=0.5, label="real", ax=ax, bins=40)
    sns.histplot(synth_days.ravel(), stat="density", color="tab:orange", alpha=0.5, label="synthetic", ax=ax, bins=40)
    ax.set_xlabel("windspeed_100m")
    ax.set_title("Marginal value distribution: real vs synthetic")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_first_diff(real_days: np.ndarray, synth_days: np.ndarray, save_path: str | Path) -> None:
    """5-min first-difference (ramp) distribution: real vs synthetic."""
    real_diff = np.diff(real_days, axis=1).ravel()
    synth_diff = np.diff(synth_days, axis=1).ravel()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(real_diff, stat="density", color="tab:blue", alpha=0.5, label="real", ax=ax, bins=40)
    sns.histplot(synth_diff, stat="density", color="tab:orange", alpha=0.5, label="synthetic", ax=ax, bins=40)
    ax.set_xlabel("windspeed_100m first difference (5-min ramp)")
    ax.set_title("Ramp distribution: real vs synthetic")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_acf(real_days: np.ndarray, synth_days: np.ndarray, save_path: str | Path, max_lag: int = 36) -> None:
    """Mean-over-days autocorrelation curve: real vs synthetic."""
    real_acfs = [autocorr(day, max_lag) for day in real_days]
    synth_acfs = [autocorr(day, max_lag) for day in synth_days]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.mean(real_acfs, axis=0), label="real", color="tab:blue")
    ax.plot(np.mean(synth_acfs, axis=0), label="synthetic", color="tab:orange")
    ax.set_xlabel("lag (5-min steps)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("Autocorrelation (mean over days): real vs synthetic")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_sanity_day(
    real_days: np.ndarray, synth_days: np.ndarray, save_path: str | Path, day_index: int = 0
) -> None:
    """One real day overlaid with one synthetic day, same index."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(real_days[day_index % len(real_days)], label="real", color="tab:blue")
    ax.plot(synth_days[day_index % len(synth_days)], label="synthetic", color="tab:orange")
    ax.set_xlabel("step in day (5-min)")
    ax.set_ylabel("windspeed_100m")
    ax.set_ylim(0, max(real_days.max(), synth_days.max()) * 1.15)
    ax.set_title("Sanity check: one real day vs one synthetic day")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_envelope(
    real_days: np.ndarray,
    synth_days: np.ndarray,
    save_path: str | Path,
    percentiles: tuple[float, float] = (2.5, 97.5),
    time_index: pd.DatetimeIndex | None = None,
    real_reference_day: np.ndarray | None = None,
) -> None:
    """Generated mean + percentile envelope across `synth_days` realizations,
    with a shaded confidence band. If `real_reference_day` is given (a single
    real day, shape (seq_len,)) it's overlaid as a solid reference line;
    otherwise the real-day mean across `real_days` is used as the reference.
    """
    seq_len = synth_days.shape[1]
    x = time_index if time_index is not None else np.arange(seq_len)

    mean = synth_days.mean(axis=0)
    lo, hi = np.percentile(synth_days, percentiles[0], axis=0), np.percentile(synth_days, percentiles[1], axis=0)
    reference = real_reference_day if real_reference_day is not None else real_days.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(x, lo, hi, color="tab:orange", alpha=0.25, label=f"synthetic {percentiles[0]}-{percentiles[1]} pct")
    ax.plot(x, mean, color="tab:orange", label="synthetic mean")
    ax.plot(x, reference, color="tab:blue", label="real")
    ax.set_ylabel("windspeed_100m")
    ax.set_title(f"Generation envelope vs real ({synth_days.shape[0]} realizations)")
    ax.legend()
    if time_index is not None:
        fig.autofmt_xdate()
    else:
        ax.set_xlabel("step in day (5-min)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def compare_to_real(
    real_days: np.ndarray,
    synth_days: np.ndarray,
    output_dir: str | Path,
    prefix: str = "compare",
    time_index: pd.DatetimeIndex | None = None,
) -> dict[str, str]:
    """Run the full comparison suite and save every plot into `output_dir`.

    Returns a dict mapping plot name -> saved file path, so callers (e.g. a
    demo script) can report exactly what was produced.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "distributions": output_dir / f"{prefix}_distributions.png",
        "first_diff": output_dir / f"{prefix}_first_diff.png",
        "acf": output_dir / f"{prefix}_acf.png",
        "sanity_day": output_dir / f"{prefix}_sanity_day.png",
        "envelope": output_dir / f"{prefix}_envelope.png",
    }

    plot_distributions(real_days, synth_days, paths["distributions"])
    plot_first_diff(real_days, synth_days, paths["first_diff"])
    plot_acf(real_days, synth_days, paths["acf"])
    plot_sanity_day(real_days, synth_days, paths["sanity_day"])
    plot_envelope(real_days, synth_days, paths["envelope"], time_index=time_index)

    return {name: str(path) for name, path in paths.items()}
