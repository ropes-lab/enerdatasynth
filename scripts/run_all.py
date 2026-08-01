"""
scripts/run_all.py
====================
Top-level orchestrator. Runs the entire pipeline end-to-end:

  1. Preprocess raw data -> data/processed/ (+ frozen backup)
  2. Run hyperparameter sweep -> reports/hyperparameter_results.csv
  3. Generate synthetic data for the best config ->
     outputs/generated/X_generated_<run_id>.npy
  4. Run the three validation plots for the best config
  5. Run failure analysis -> reports/failure_analysis.md
  6. Generate results_summary.md

Usage (from the project root, with the package installed or
`src/` on PYTHONPATH):
    python scripts/run_all.py                  # quick sweep (9 configs, 60 epochs each)
    python scripts/run_all.py --full-grid      # full 27-config grid
    python scripts/run_all.py --epochs 200     # more epochs per config
"""

from __future__ import annotations
import argparse

from powerdiffusion.data.preprocess import run_preprocessing
from powerdiffusion.training.sweep import run_sweep
from powerdiffusion.sampling.generate import generate_for_run
from powerdiffusion.visualization.plots import run_all_validation_plots
from powerdiffusion.evaluation.failure_analysis import analyze, write_failure_analysis_md
from powerdiffusion.evaluation.results_report import generate_results_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60,
                         help="Epochs per sweep configuration")
    parser.add_argument("--full-grid", action="store_true",
                         help="Run all 27 hyperparameter combinations "
                              "instead of the 9-run one-factor-at-a-time sweep")
    parser.add_argument("--skip-preprocess", action="store_true",
                         help="Skip preprocessing if data/processed/ already exists")
    args = parser.parse_args()

    if not args.skip_preprocess:
        print("\n" + "#" * 70)
        print("# STAGE 1: PREPROCESSING")
        print("#" * 70)
        run_preprocessing()

    print("\n" + "#" * 70)
    print("# STAGE 2: HYPERPARAMETER SWEEP")
    print("#" * 70)
    df = run_sweep(max_epochs=args.epochs, full_grid=args.full_grid)
    best_row = df.sort_values("best_val_loss").iloc[0]
    run_id = best_row["run_id"]
    print(f"\nBest config: {run_id}  (val_loss={best_row['best_val_loss']:.5f})")

    print("\n" + "#" * 70)
    print("# STAGE 3: GENERATION (best config)")
    print("#" * 70)
    generate_for_run(
        run_id=run_id,
        diffusion_steps=int(best_row["diffusion_steps"]),
        hidden_dim=int(best_row["hidden_dim"]),
        n_layers=int(best_row["n_layers"]),
    )

    print("\n" + "#" * 70)
    print("# STAGE 4: VALIDATION PLOTS")
    print("#" * 70)
    run_all_validation_plots(run_id)

    print("\n" + "#" * 70)
    print("# STAGE 5: FAILURE ANALYSIS")
    print("#" * 70)
    result = analyze(run_id)
    write_failure_analysis_md(result)
    print(f"Issues detected: {len(result['issues'])}")

    print("\n" + "#" * 70)
    print("# STAGE 6: RESULTS REPORT")
    print("#" * 70)
    generate_results_summary()

    print("\n" + "#" * 70)
    print("# PIPELINE COMPLETE")
    print("#" * 70)
    print(f"""
  Best run: {run_id}
  See:
    reports/results_summary.md
    reports/hyperparameter_results.csv
    reports/hyperparameter_summary.md
    reports/failure_analysis.md
    reports/future_experiments.md
    reports/figures/
    data/processed/README.md
    data/processed_backup/   (frozen backup of training data)
""")


if __name__ == "__main__":
    main()
