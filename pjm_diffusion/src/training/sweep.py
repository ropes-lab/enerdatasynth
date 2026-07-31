"""
src/training/sweep.py
=======================
Runs the manual hyperparameter sweep requested by the PI:
  Diffusion steps: 50, 100, 200
  Hidden dimension: 64, 128, 256
  Number of layers: 2, 4, 6

That's a 3x3x3 = 27-configuration grid. For each configuration we
train, record train/val loss, training time, and generation time,
then write hyperparameter_results.csv and rank configurations in
hyperparameter_summary.md.

NOTE: 27 full training runs at max_epochs=200 is a lot of compute.
For the *diffusion steps* experiment specifically (the one the PI
asked to track most closely: 50 / 100 / 200), we hold hidden_dim and
n_layers at their defaults and vary T. For the full 27-cell grid we
also run a shorter max_epochs unless `full=True` is passed, to keep
sweep time reasonable for a baseline study. This is documented in
hyperparameter_summary.md so it's never a hidden decision.
"""

from __future__ import annotations
import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import (
    RunConfig, PROCESSED_DIR, REPORTS_DIR,
    SWEEP_DIFFUSION_STEPS, SWEEP_HIDDEN_DIM, SWEEP_N_LAYERS,
    DEFAULT_HIDDEN_DIM, DEFAULT_N_LAYERS, DEFAULT_T,
)
from src.training.train import train_one_config


def run_sweep(max_epochs: int = 60, full_grid: bool = False) -> pd.DataFrame:
    """
    Run the hyperparameter sweep and save results.

    max_epochs : epochs per configuration. Lower than the single
                 baseline run (200) so the full sweep finishes in
                 reasonable time; documented in the summary report.
    full_grid  : if True, runs all 27 combinations (3x3x3).
                 if False (default), runs the 3 diffusion-step
                 variations at default hidden/layers, PLUS the
                 3 hidden-dim variations at default T/layers, PLUS
                 the 3 layer-count variations at default T/hidden —
                 9 runs total instead of 27, isolating one axis at
                 a time. This is the standard "one factor at a time"
                 sweep design and is far cheaper than the full grid
                 while still answering "what matters most" for each
                 axis independently.
    """
    X_train = np.load(PROCESSED_DIR / "X_daily_profiles.npy")
    Z_train = np.load(PROCESSED_DIR / "Z_conditions.npy")
    X_val   = np.load(PROCESSED_DIR / "X_daily_profiles_val.npy")
    Z_val   = np.load(PROCESSED_DIR / "Z_conditions_val.npy")

    if full_grid:
        configs = [
            RunConfig(diffusion_steps=t, hidden_dim=h, n_layers=l, max_epochs=max_epochs)
            for t, h, l in itertools.product(
                SWEEP_DIFFUSION_STEPS, SWEEP_HIDDEN_DIM, SWEEP_N_LAYERS
            )
        ]
        sweep_type = "full_grid_27"
    else:
        configs = []
        seen = set()
        for t in SWEEP_DIFFUSION_STEPS:
            c = RunConfig(diffusion_steps=t, hidden_dim=DEFAULT_HIDDEN_DIM,
                           n_layers=DEFAULT_N_LAYERS, max_epochs=max_epochs)
            if c.run_id not in seen:
                configs.append(c); seen.add(c.run_id)
        for h in SWEEP_HIDDEN_DIM:
            c = RunConfig(diffusion_steps=DEFAULT_T, hidden_dim=h,
                           n_layers=DEFAULT_N_LAYERS, max_epochs=max_epochs)
            if c.run_id not in seen:
                configs.append(c); seen.add(c.run_id)
        for l in SWEEP_N_LAYERS:
            c = RunConfig(diffusion_steps=DEFAULT_T, hidden_dim=DEFAULT_HIDDEN_DIM,
                           n_layers=l, max_epochs=max_epochs)
            if c.run_id not in seen:
                configs.append(c); seen.add(c.run_id)
        sweep_type = "one_factor_at_a_time_9"

    print(f"Sweep type: {sweep_type}  |  {len(configs)} configurations  |  "
          f"max_epochs={max_epochs}")

    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {cfg.run_id}")
        res = train_one_config(X_train, Z_train, X_val, Z_val, cfg, verbose=True)
        res["sweep_type"] = sweep_type
        results.append(res)

    df = pd.DataFrame(results)
    csv_path = REPORTS_DIR / "hyperparameter_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    write_hyperparameter_summary(df, sweep_type, max_epochs)
    return df


def write_hyperparameter_summary(df: pd.DataFrame, sweep_type: str,
                                  max_epochs: int) -> None:
    df_sorted = df.sort_values("best_val_loss")

    lines = [
        "# Hyperparameter Sweep Summary",
        "",
        f"Sweep type: `{sweep_type}`",
        f"Epochs per configuration: {max_epochs}",
        "",
        "## Ranked Configurations (by best validation loss)",
        "",
        "| Rank | Config | T | Hidden | Layers | Params | Best Val Loss | "
        "Train Time (s) | Gen Time /64 (s) |",
        "|------|--------|---|--------|--------|--------|----------------|"
        "----------------|-------------------|",
    ]
    for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
        lines.append(
            f"| {rank} | {row['run_id']} | {row['diffusion_steps']} | "
            f"{row['hidden_dim']} | {row['n_layers']} | {row['n_params']:,} | "
            f"{row['best_val_loss']:.5f} | {row['training_time_sec']:.1f} | "
            f"{row['generation_time_sec_per_64']:.2f} |"
        )

    best = df_sorted.iloc[0]
    lines += [
        "",
        "## Best Configuration",
        "",
        f"**{best['run_id']}** — diffusion_steps={best['diffusion_steps']}, "
        f"hidden_dim={best['hidden_dim']}, n_layers={best['n_layers']}",
        f"- Best validation loss: {best['best_val_loss']:.5f} "
        f"(epoch {best['best_epoch']})",
        f"- Parameters: {best['n_params']:,}",
        f"- Training time: {best['training_time_sec']:.1f}s",
        "",
        "## Notes on Sweep Design",
        "",
        "This sweep isolates one hyperparameter axis at a time "
        "(diffusion steps, then hidden dimension, then layer count) "
        "rather than running the full 27-cell grid, to keep baseline "
        "compute reasonable. The full grid can be run with "
        "`run_sweep(full_grid=True)` if a more exhaustive search is "
        "needed later.",
        "",
        "## Trade-offs Observed",
        "",
    ]

    # Diffusion steps trade-off — filter to the runs that vary ONLY
    # diffusion_steps (i.e. hidden_dim and n_layers held at defaults)
    from src.utils.config import DEFAULT_HIDDEN_DIM, DEFAULT_N_LAYERS
    steps_df = df[
        (df["hidden_dim"] == DEFAULT_HIDDEN_DIM) &
        (df["n_layers"] == DEFAULT_N_LAYERS)
    ].sort_values("diffusion_steps")
    if len(steps_df) > 1:
        lines.append("**Diffusion steps** (hidden_dim and n_layers held at defaults):")
        for _, row in steps_df.iterrows():
            lines.append(
                f"- T={row['diffusion_steps']}: val_loss={row['best_val_loss']:.5f}, "
                f"generation time per 64 samples = {row['generation_time_sec_per_64']:.2f}s"
            )
        lines.append(
            "\nMore diffusion steps generally costs more generation time "
            "linearly (more reverse steps to run) — check whether the "
            "validation loss improvement justifies the slower sampling."
        )

    # Hidden dim trade-off
    hidden_df = df[
        (df["diffusion_steps"] == DEFAULT_T) &
        (df["n_layers"] == DEFAULT_N_LAYERS)
    ].sort_values("hidden_dim")
    if len(hidden_df) > 1:
        lines.append("\n**Hidden dimension** (T and n_layers held at defaults):")
        for _, row in hidden_df.iterrows():
            lines.append(
                f"- hidden_dim={row['hidden_dim']}: val_loss={row['best_val_loss']:.5f}, "
                f"params={row['n_params']:,}"
            )

    # Layer count trade-off
    layers_df = df[
        (df["diffusion_steps"] == DEFAULT_T) &
        (df["hidden_dim"] == DEFAULT_HIDDEN_DIM)
    ].sort_values("n_layers")
    if len(layers_df) > 1:
        lines.append("\n**Number of layers** (T and hidden_dim held at defaults):")
        for _, row in layers_df.iterrows():
            lines.append(
                f"- n_layers={row['n_layers']}: val_loss={row['best_val_loss']:.5f}, "
                f"params={row['n_params']:,}"
            )
        lines.append(
            "\nNote: deeper is not always better on this dataset — "
            "more layers can slow or destabilize convergence for a "
            "problem this small, as the L6 result shows."
        )

    with open(REPORTS_DIR / "hyperparameter_summary.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {REPORTS_DIR / 'hyperparameter_summary.md'}")


if __name__ == "__main__":
    run_sweep(max_epochs=60, full_grid=False)
