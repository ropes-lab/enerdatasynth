"""
src/enerdatasynth/resources/pjm_load/evaluation/results_report.py
=================================================================
Generates reports/results_summary.md automatically from:
  - data/processed/metadata.json
  - the shipped default checkpoint's result.json / calibration.json
  - reports/hyperparameter_results.csv (the sweep, if one was run)
  - reports/failure_analysis.md (already written)
"""

from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

from enerdatasynth.utils.config import PROCESSED_DIR, REPORTS_DIR, CHECKPOINT_DIR, ensure_dirs


_ARCH_BLURB = {
    "transformer": (
        "Transformer-encoder denoiser. Each of the 24 hours is one token with "
        "a learned positional embedding; the diffusion timestep (sinusoidal "
        "embedding) and the condition vector are projected through a shared "
        "MLP and added to every token; a standard `nn.TransformerEncoder` "
        "(GELU, 4x feedforward) and a linear head predict the noise. Loss: "
        "standard diffusion MSE only."
    ),
    "mlp": (
        "Simple feedforward MLP denoiser (no attention, no transformers, no "
        "Fourier features). Inputs: noisy profile x_t, diffusion timestep t "
        "(sinusoidal embedding), condition vector z (concatenated). Output: "
        "predicted noise. Loss: standard diffusion MSE only."
    ),
}


def _default_checkpoint_section() -> list[str]:
    """Describe the checkpoint `enerdatasynth.model("pjm_load")` actually loads."""
    from enerdatasynth.generators.load_conditioning import _discover_best_checkpoint

    try:
        run_id = _discover_best_checkpoint(CHECKPOINT_DIR)
    except FileNotFoundError:
        return ["## Shipped Default Checkpoint", "", "_No trained checkpoint found. Run `enerdatasynth train` first._", ""]

    with open(CHECKPOINT_DIR / run_id / "result.json") as f:
        res = json.load(f)
    arch = res.get("architecture", "mlp")
    cal_path = CHECKPOINT_DIR / run_id / "calibration.json"
    if cal_path.exists():
        with open(cal_path) as f:
            cal = json.load(f)
        calibration = f"scalar offset {cal['scalar']:+.5f}, fit on {cal['n_days_used']} training days"
    else:
        calibration = "none (uncalibrated)"

    return [
        "## Shipped Default Checkpoint",
        "",
        f"Run: **{run_id}** (architecture: {arch})",
        f"- Diffusion steps: {res['diffusion_steps']}",
        f"- Hidden dimension: {res['hidden_dim']}, layers: {res['n_layers']}",
        f"- Parameters: {res['n_params']:,}",
        f"- Best validation loss: {res['best_val_loss']:.5f} (epoch {res['best_epoch']} of {res['epochs_trained']} trained)",
        f"- Calibration: {calibration}",
        "",
        "## Model Architecture",
        "",
        _ARCH_BLURB.get(arch, f"{arch} denoiser."),
        "",
    ]


def generate_results_summary() -> Path:
    ensure_dirs()
    with open(PROCESSED_DIR / "metadata.json") as f:
        meta = json.load(f)

    hp_csv = REPORTS_DIR / "hyperparameter_results.csv"
    df = pd.read_csv(hp_csv) if hp_csv.exists() else None
    best_row = df.sort_values("best_val_loss").iloc[0] if df is not None else None

    failure_path = REPORTS_DIR / "failure_analysis.md"
    failure_text = failure_path.read_text() if failure_path.exists() else \
        "_Not yet generated. Run `python -m enerdatasynth.resources.pjm_load.evaluation.failure_analysis` first._"

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
    ]
    lines += _default_checkpoint_section()

    if best_row is not None:
        lines += [
            "## Hyperparameter Sweep",
            "",
            "Best configuration in `hyperparameter_results.csv`, ranked by "
            "validation loss. This is the sweep's own winner and is separate "
            "from the shipped default described above (a Transformer is "
            "selected over any MLP run regardless of validation loss, since "
            "losses are not comparable across diffusion-step counts).",
            "",
            f"Best sweep configuration: **{best_row['run_id']}**",
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
            "See `hyperparameter_results.csv` and `hyperparameter_summary.md` "
            "for the full ranked sweep.",
            "",
        ]
    else:
        lines += [
            "## Hyperparameter Sweep",
            "",
            "_Sweep not yet run. Run `python -m enerdatasynth.resources.pjm_load.training.sweep` first._",
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
        "Not yet implemented in this package:",
        "- Re-sweeping hyperparameters for the transformer architecture "
        "(the current defaults come from research_history, not a "
        "from-scratch sweep against the package's own conditioning pipeline)",
        "- One-hot vs the research lineage's raw (non-one-hot) conditioning encoding",
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
