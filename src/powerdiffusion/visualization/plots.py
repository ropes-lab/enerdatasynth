"""
src/powerdiffusion/visualization/plots.py
===========================================
All plotting for the pipeline lives here, sharing one color palette,
instead of being scattered/duplicated across training and evaluation
code (previously `PRIMARY_COLOR`/`SECONDARY_COLOR` were hardcoded
independently in both `src/validation/metrics.py` and
`src/training/train.py` with the same two hex values).

Includes the three validation plots requested by the PI:
  1. Hourly distribution comparison (overlapping boxplots)
  2. First-difference distribution comparison
  3. Autocorrelation comparison

All three compare REAL vs GENERATED profiles built from the SAME
conditioning vectors (in-sample validation, as the PI requested).

Also includes the training loss-curve plot (moved here from
`training/train.py` so both call sites share one palette) and
`denorm`, an inverse-MinMax helper for reporting in MW units — not
currently called by any pipeline stage (all validation plots operate
on normalized [0,1] load), kept here rather than deleted since it's
the natural companion to `preprocess.py`'s forward MinMax transform.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

from powerdiffusion.utils.config import FIGURES_DIR

# Shared palette — single source of truth for "real/train" vs "generated/val" colors.
PRIMARY_COLOR   = "#00A8E8"
SECONDARY_COLOR = "#F5A623"

# Kept as aliases so the validation-plot call sites below read naturally.
REAL_COLOR = PRIMARY_COLOR
GEN_COLOR  = SECONDARY_COLOR


def denorm(X_scaled: np.ndarray, data_min: np.ndarray, data_max: np.ndarray) -> np.ndarray:
    """Invert per-hour MinMax scaling back to MW."""
    return X_scaled * (data_max - data_min) + data_min


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Hourly distribution comparison
# ═════════════════════════════════════════════════════════════════════════════

def plot_hourly_distribution_comparison(X_real: np.ndarray, X_gen: np.ndarray,
                                         run_id: str,
                                         out_path: Path | None = None) -> Path:
    """Overlapping boxplots, real vs generated, for each of the 24 hours."""
    n_hours = X_real.shape[1]
    fig, ax = plt.subplots(figsize=(15, 5))

    positions_real = np.arange(n_hours) * 2.2
    positions_gen  = positions_real + 0.9

    bp_real = ax.boxplot(
        [X_real[:, h] for h in range(n_hours)],
        positions=positions_real, widths=0.7, patch_artist=True,
        medianprops={"color": "black", "lw": 1.2},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.2},
    )
    bp_gen = ax.boxplot(
        [X_gen[:, h] for h in range(n_hours)],
        positions=positions_gen, widths=0.7, patch_artist=True,
        medianprops={"color": "black", "lw": 1.2},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.2},
    )
    for patch in bp_real["boxes"]:
        patch.set_facecolor(REAL_COLOR); patch.set_alpha(0.6)
    for patch in bp_gen["boxes"]:
        patch.set_facecolor(GEN_COLOR); patch.set_alpha(0.6)

    ax.set_xticks(positions_real + 0.45)
    ax.set_xticklabels([f"h{h:02d}" for h in range(n_hours)], rotation=45, fontsize=8)
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Normalized Load [0,1]")
    ax.set_title(f"Hourly Distribution Comparison — {run_id}")
    ax.legend([bp_real["boxes"][0], bp_gen["boxes"][0]], ["Real", "Generated"],
              loc="upper right")
    fig.tight_layout()

    out_path = out_path or FIGURES_DIR / f"hourly_distribution_comparison_{run_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 — First-difference distribution comparison
# ═════════════════════════════════════════════════════════════════════════════

def plot_first_difference_comparison(X_real: np.ndarray, X_gen: np.ndarray,
                                      run_id: str,
                                      out_path: Path | None = None) -> Path:
    """delta_x[t] = x[t] - x[t-1], compared real vs generated."""
    diff_real = np.diff(X_real, axis=1).flatten()
    diff_gen  = np.diff(X_gen,  axis=1).flatten()

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(
        min(diff_real.min(), diff_gen.min()),
        max(diff_real.max(), diff_gen.max()),
        100,
    )
    ax.hist(diff_real, bins=bins, alpha=0.55, color=REAL_COLOR,
            label="Real", density=True)
    ax.hist(diff_gen,  bins=bins, alpha=0.55, color=GEN_COLOR,
            label="Generated", density=True)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Δx (hour-over-hour, normalized)")
    ax.set_ylabel("Density")
    ax.set_title(f"First Difference Distribution — {run_id}")
    ax.legend()
    fig.tight_layout()

    out_path = out_path or FIGURES_DIR / f"first_difference_comparison_{run_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Autocorrelation comparison
# ═════════════════════════════════════════════════════════════════════════════

def plot_autocorrelation_comparison(X_real: np.ndarray, X_gen: np.ndarray,
                                     run_id: str, n_lags: int = 23,
                                     out_path: Path | None = None) -> Path:
    """
    ACF computed on the flattened series for both real and generated.
    Since each sample is only 24 hours, intra-day lags (1-23) are
    the meaningful range here.
    """
    series_real = X_real.flatten()
    series_gen  = X_gen.flatten()

    acf_real = acf(series_real, nlags=n_lags, fft=True)
    acf_gen  = acf(series_gen,  nlags=n_lags, fft=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(n_lags + 1)
    width = 0.4
    ax.bar(x - width/2, acf_real, width=width, color=REAL_COLOR,
           alpha=0.8, label="Real")
    ax.bar(x + width/2, acf_gen,  width=width, color=GEN_COLOR,
           alpha=0.8, label="Generated")
    ax.set_xlabel("Lag (hours)")
    ax.set_ylabel("ACF")
    ax.set_title(f"Autocorrelation Comparison — {run_id}")
    ax.legend()
    fig.tight_layout()

    out_path = out_path or FIGURES_DIR / f"autocorrelation_comparison_{run_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path, acf_real, acf_gen


# ═════════════════════════════════════════════════════════════════════════════
# Training loss curve (moved here from training/train.py — see module docstring)
# ═════════════════════════════════════════════════════════════════════════════

def plot_loss_curve(history: dict, run_id: str, best_epoch: int,
                     out_path: Path) -> Path:
    """Train/val MSE loss vs. epoch, log-scaled, with the best epoch marked."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train", color=PRIMARY_COLOR)
    ax.plot(history["epoch"], history["val_loss"],   label="val",   color=SECONDARY_COLOR)
    ax.axvline(best_epoch, color="gray", ls="--", lw=0.8,
               label=f"best epoch ({best_epoch})")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title(f"Training Loss — {run_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# Run all three validation plots for a given run_id
# ═════════════════════════════════════════════════════════════════════════════

def run_all_validation_plots(run_id: str) -> dict:
    from powerdiffusion.utils.config import PROCESSED_DIR, GENERATED_DIR

    X_real = np.load(PROCESSED_DIR / "X_daily_profiles_val.npy")
    X_gen  = np.load(GENERATED_DIR / f"X_generated_{run_id}.npy")

    p1 = plot_hourly_distribution_comparison(X_real, X_gen, run_id)
    p2 = plot_first_difference_comparison(X_real, X_gen, run_id)
    p3, acf_real, acf_gen = plot_autocorrelation_comparison(X_real, X_gen, run_id)

    return {
        "hourly_distribution_plot": str(p1),
        "first_difference_plot": str(p2),
        "autocorrelation_plot": str(p3),
        "acf_real": acf_real.tolist(),
        "acf_gen": acf_gen.tolist(),
    }


if __name__ == "__main__":
    from powerdiffusion.utils.config import RunConfig
    cfg = RunConfig()
    result = run_all_validation_plots(cfg.run_id)
    print(result)
