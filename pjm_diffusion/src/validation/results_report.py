"""
src/validation/results_report.py
===================================
Generates reports/results_summary.md automatically from:
  - data/processed/metadata.json
  - reports/hyperparameter_results.csv
  - reports/failure_analysis.md (already written)
  - the best run's checkpoint result.json
"""

from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config import PROCESSED_DIR, REPORTS_DIR, CHECKPOINT_DIR


def generate_results_summary() -> Path:
    with open(PROCESSED_DIR / "metadata.json") as f:
        meta = json.load(f)

    hp_csv = REPORTS_DIR / "hyperparameter_results.csv"
    df = pd.read_csv(hp_csv) if hp_csv.exists() else None
    best_row = df.sort_values("best_val_loss").iloc[0] if df is not None else None

    failure_path = REPORTS_DIR / "failure_analysis.md"
    failure_text = failure_path.read_text() if failure_path.exists() else \
        "_Not yet generated. Run src/validation/failure_analysis.py first._"

    lines = [
        "# Results Summary",
        "",
        "## Dataset Summary",
        "",
        f"- Zone: {meta['zone']}",
        f"- Training samples: {meta['sample_count']['train']}",
        f"- Validation samples: {meta['sample_count']['val']}",
        f"- Total raw days before cleaning: {meta['sample_count']['total_raw']}",
        f"- X shape (train): {meta['X_shape_train']}",
        f"- Z shape (train): {meta['Z_shape_train']}, dim={meta['condition_dim']}",
        "",
        "## Model Architecture",
        "",
        "Simple feedforward MLP denoiser (no attention, no transformers, "
        "no Fourier features). Inputs: noisy profile x_t, diffusion "
        "timestep t (sinusoidal embedding), condition vector z "
        "(concatenated). Output: predicted noise. Loss: standard "
        "diffusion MSE only.",
        "",
    ]

    if best_row is not None:
        lines += [
            "## Training Results",
            "",
            f"Best configuration: **{best_row['run_id']}**",
            f"- Diffusion steps: {best_row['diffusion_steps']}",
            f"- Hidden dimension: {best_row['hidden_dim']}",
            f"- Layers: {best_row['n_layers']}",
            f"- Parameters: {best_row['n_params']:,}",
            f"- Best validation loss: {best_row['best_val_loss']:.5f} "
            f"(epoch {best_row['best_epoch']})",
            f"- Training time: {best_row['training_time_sec']:.1f}s",
            f"- Generation time (64 samples): "
            f"{best_row['generation_time_sec_per_64']:.2f}s",
            "",
            "## Best Hyperparameters",
            "",
            "See `hyperparameter_results.csv` and `hyperparameter_summary.md` "
            "for the full ranked sweep.",
            "",
        ]
    else:
        lines += [
            "## Training Results",
            "",
            "_Sweep not yet run. Run src/training/sweep.py first._",
            "",
        ]

    lines += [
        "## Validation Results",
        "",
        "Three validation plots generated per configuration:",
        "1. `hourly_distribution_comparison_<run_id>.png` — real vs "
        "generated boxplots per hour",
        "2. `first_difference_comparison_<run_id>.png` — hour-over-hour "
        "delta distributions",
        "3. `autocorrelation_comparison_<run_id>.png` — ACF comparison",
        "",
        "All comparisons use IN-SAMPLE validation: generated profiles "
        "are produced from the SAME conditioning vectors as the real "
        "validation set, as requested.",
        "",
        "## Failure Modes",
        "",
        failure_text,
        "",
        "## Recommended Next Experiments",
        "",
        "See `reports/future_experiments.md` for the full list. "
        "Per the PI's direction, these are NOT implemented yet:",
        "- One-hot vs cyclic encoding",
        "- Attention layers",
        "- MW anchor (multi-day continuity)",
        "- Unified PJM model (all zones)",
        "- Advanced conditioning (humidity, hourly temperatures)",
        "",
    ]

    out_path = REPORTS_DIR / "results_summary.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return out_path


if __name__ == "__main__":
    path = generate_results_summary()
    print(f"Saved: {path}")
