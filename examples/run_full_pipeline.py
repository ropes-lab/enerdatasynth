"""
examples/run_full_pipeline.py
===============================
Small, fast walkthrough of the pipeline's building blocks — this is a
demonstration, not the production entry point. For the real sweep
(9-27 configurations, up to 200 epochs each), use `scripts/run_all.py`.

This example runs preprocessing (if needed), trains ONE small model
for a handful of epochs, generates synthetic profiles from it, and
produces the three validation plots — enough to see every stage of
the pipeline connect, in a couple of minutes rather than a couple of
hours.

Usage (from the project root, with `src/` on PYTHONPATH or the
package installed):
    python examples/run_full_pipeline.py
"""

from __future__ import annotations

from powerdiffusion.data.preprocess import run_preprocessing
from powerdiffusion.training.train import train_one_config
from powerdiffusion.sampling.generate import generate_for_run
from powerdiffusion.visualization.plots import run_all_validation_plots
from powerdiffusion.utils.config import RunConfig, PROCESSED_DIR
import numpy as np


def main():
    if not (PROCESSED_DIR / "X_daily_profiles.npy").exists():
        print("No processed data found — running preprocessing first...")
        run_preprocessing()

    X_train = np.load(PROCESSED_DIR / "X_daily_profiles.npy")
    Z_train = np.load(PROCESSED_DIR / "Z_conditions.npy")
    X_val   = np.load(PROCESSED_DIR / "X_daily_profiles_val.npy")
    Z_val   = np.load(PROCESSED_DIR / "Z_conditions_val.npy")

    # A small, fast configuration — NOT the tuned baseline. See
    # reports/hyperparameter_summary.md for the actual best config
    # found by the full sweep.
    demo_cfg = RunConfig(diffusion_steps=50, hidden_dim=64, n_layers=2, max_epochs=5)
    result = train_one_config(X_train, Z_train, X_val, Z_val, demo_cfg)

    generate_for_run(
        run_id=demo_cfg.run_id,
        diffusion_steps=demo_cfg.diffusion_steps,
        hidden_dim=demo_cfg.hidden_dim,
        n_layers=demo_cfg.n_layers,
    )
    plots = run_all_validation_plots(demo_cfg.run_id)

    print(f"\nDemo run complete: {demo_cfg.run_id}")
    print(f"Best val loss: {result['best_val_loss']:.5f}")
    print("Validation plots:")
    for name, path in plots.items():
        if name.startswith("acf_"):
            continue
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
