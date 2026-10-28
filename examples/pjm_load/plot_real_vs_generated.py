"""
examples/pjm_load/plot_real_vs_generated.py
===================================
Six representative validation days, each showing the real daily load profile
against ONE generated profile sampled from that day's exact conditioning
vector. Produces docs/assets/pjm_load_real_vs_generated.png.

Days are chosen by a fixed rule on the validation set's daily maximum
temperature (never by looking at generated output): the coldest day, the
hottest day, and the day nearest the median Tmax within spring (Mar-May),
fall (Sep-Nov), weekends, and overall. The sampling seed is fixed, so the
figure is reproducible and not a best-of-N pick.

Usage:
    python examples/pjm_load/plot_real_vs_generated.py                  # shipped default
    python examples/pjm_load/plot_real_vs_generated.py --run-id T100_H256_L4

Reproduce:
    1. From a clone of this repository: `pip install -e ".[research]"`.
    2. Get the PJM and NOAA source files yourself (sources: data/README.md) and put
       them in data/raw/pjm_load/. The repository ships no real data.
    3. `enerdatasynth preprocess`
    4. `python examples/pjm_load/plot_real_vs_generated.py`
    Output: docs/assets/pjm_load_real_vs_generated.png. Same seed, same days, same figure.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from enerdatasynth.utils.config import load_processed, processed_path, CHECKPOINT_DIR, PROJECT_ROOT
from enerdatasynth.generators import LoadDiffusionModel
from enerdatasynth.generators.load_conditioning import _discover_best_checkpoint

SEED = 0


def select_days(df: pd.DataFrame) -> list[tuple[str, int]]:
    """(panel label, row index into the validation set) for the six panels."""
    tmax = df["Tmax"].to_numpy()

    def nearest_median(mask: np.ndarray) -> int:
        idx = np.flatnonzero(mask)
        return int(idx[np.argmin(np.abs(tmax[idx] - np.median(tmax[idx])))])

    month, dow = df["month"].to_numpy(), df["dow"].to_numpy()
    return [
        ("Coldest day", int(np.argmin(tmax))),
        ("Spring (median Tmax)", nearest_median(np.isin(month, [3, 4, 5]))),
        ("Hottest day", int(np.argmax(tmax))),
        ("Fall (median Tmax)", nearest_median(np.isin(month, [9, 10, 11]))),
        ("Weekend (median Tmax)", nearest_median(dow >= 5)),
        ("Median Tmax, overall", nearest_median(np.ones(len(df), dtype=bool))),
    ]


def main(run_id: str, out_path: Path) -> None:
    df = pd.read_parquet(processed_path("df_val.parquet"))
    X_val = load_processed("X_daily_profiles_val.npy")
    Z_val = load_processed("Z_conditions_val.npy")

    days = select_days(df)
    rows = [i for _, i in days]
    model = LoadDiffusionModel.from_run_id(run_id)
    generated = model.sample(Z_val[rows], seed=SEED)
    calibrated = model.calibration is not None

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharey=True)
    for ax, (label, i), gen in zip(axes.ravel(), days, generated):
        date = pd.Timestamp(df["date"].iloc[i]).date()
        ax.plot(range(24), X_val[i], color="black", lw=2.2, label="Real")
        ax.plot(range(24), gen, color="#4466ee", lw=2.0, ls="--", label="Generated")
        ax.set_title(f"{label}\n{date} — Tmax {df['Tmax'].iloc[i]:.0f}°F", fontsize=10)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.grid(alpha=0.25)
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized load [0,1]")
    for ax in axes[1]:
        ax.set_xlabel("Hour of day")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.94))
    arch = model.architecture
    cal = ", calibrated" if calibrated else ", uncalibrated"
    fig.suptitle(f"PJM Load (zone PS) — real vs. generated, {arch} checkpoint “{run_id}”{cal}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None, help="default: the checkpoint enerdatasynth.model('pjm_load') loads")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs/assets/pjm_load_real_vs_generated.png")
    args = ap.parse_args()
    main(args.run_id or _discover_best_checkpoint(CHECKPOINT_DIR), args.out)
