"""
examples/exploratory_data_analysis.py
=======================================
Small demonstration of inspecting the processed dataset that
`pjm_diffusion.data.preprocess` produces: mean daily load shape,
per-hour spread, and the train/val split sizes from metadata.json.

This is intentionally lightweight — it demonstrates how to work with
the package's own processed output, not a full exploratory analysis.
The original, more exhaustive EDA (zone correlations, load-duration
curves, COVID-period visualization, temperature-vs-load V-curves) is
preserved as historical research in research_history/eda_exploration.py.

Usage:
    python examples/exploratory_data_analysis.py
"""

from __future__ import annotations
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pjm_diffusion.utils.config import PROCESSED_DIR
from pjm_diffusion.visualization.plots import PRIMARY_COLOR, SECONDARY_COLOR


def main():
    X_train = np.load(PROCESSED_DIR / "X_daily_profiles.npy")
    X_val   = np.load(PROCESSED_DIR / "X_daily_profiles_val.npy")
    with open(PROCESSED_DIR / "metadata.json") as f:
        meta = json.load(f)

    print(f"Zone: {meta['zone']}")
    print(f"Train days: {meta['sample_count']['train']}  "
          f"Val days: {meta['sample_count']['val']}")
    print(f"Excluded days: {meta['exclusion_counts']}")

    hours = np.arange(X_train.shape[1])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hours, X_train.mean(axis=0), color=PRIMARY_COLOR, label="Train mean")
    ax.plot(hours, X_val.mean(axis=0), color=SECONDARY_COLOR, label="Val mean")
    ax.fill_between(
        hours,
        X_train.mean(axis=0) - X_train.std(axis=0),
        X_train.mean(axis=0) + X_train.std(axis=0),
        color=PRIMARY_COLOR, alpha=0.15,
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Normalized load [0,1]")
    ax.set_title(f"Mean daily load shape — {meta['zone']} zone")
    ax.legend()
    fig.tight_layout()

    out_path = "examples_eda_mean_profile.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
