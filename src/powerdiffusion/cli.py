"""
powerdiffusion.cli
====================
Console-script entry point (`powerdiffusion`, installed via
`[project.scripts]` in pyproject.toml). Thin argparse wrapper around
the existing pipeline functions — no new logic lives here, only
argument parsing and dispatch, so behavior matches calling the
underlying functions directly.

Usage:
    powerdiffusion preprocess
    powerdiffusion train [--diffusion-steps T] [--hidden-dim H] [--n-layers L] ...
    powerdiffusion sweep [--epochs N] [--full-grid]
    powerdiffusion generate --diffusion-steps T --hidden-dim H --n-layers L
    powerdiffusion evaluate --run-id T100_H128_L4
    powerdiffusion report
    powerdiffusion pipeline [--epochs N] [--full-grid] [--skip-preprocess]
"""

from __future__ import annotations
import argparse

import numpy as np

from powerdiffusion.utils.config import RunConfig, PROCESSED_DIR, DEFAULT_T, DEFAULT_HIDDEN_DIM, DEFAULT_N_LAYERS


def _add_run_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--diffusion-steps", type=int, default=DEFAULT_T)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)


def _run_config_from_args(args: argparse.Namespace) -> RunConfig:
    kwargs = dict(
        diffusion_steps=args.diffusion_steps,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    )
    for field in ("batch_size", "lr", "max_epochs", "patience", "seed"):
        value = getattr(args, field, None)
        if value is not None:
            kwargs[field] = value
    return RunConfig(**kwargs)


def cmd_preprocess(args: argparse.Namespace) -> None:
    from powerdiffusion.data.preprocess import run_preprocessing
    run_preprocessing()


def cmd_train(args: argparse.Namespace) -> None:
    from powerdiffusion.training.train import train_one_config
    cfg = _run_config_from_args(args)
    X_train = np.load(PROCESSED_DIR / "X_daily_profiles.npy")
    Z_train = np.load(PROCESSED_DIR / "Z_conditions.npy")
    X_val   = np.load(PROCESSED_DIR / "X_daily_profiles_val.npy")
    Z_val   = np.load(PROCESSED_DIR / "Z_conditions_val.npy")
    train_one_config(X_train, Z_train, X_val, Z_val, cfg)


def cmd_sweep(args: argparse.Namespace) -> None:
    from powerdiffusion.training.sweep import run_sweep
    run_sweep(max_epochs=args.epochs, full_grid=args.full_grid)


def cmd_generate(args: argparse.Namespace) -> None:
    from powerdiffusion.sampling.generate import generate_for_run
    cfg = _run_config_from_args(args)
    generate_for_run(
        run_id=cfg.run_id,
        diffusion_steps=cfg.diffusion_steps,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    from powerdiffusion.visualization.plots import run_all_validation_plots
    from powerdiffusion.evaluation.failure_analysis import analyze, write_failure_analysis_md
    run_all_validation_plots(args.run_id)
    result = analyze(args.run_id)
    write_failure_analysis_md(result)
    print(f"Issues detected: {len(result['issues'])}")


def cmd_report(args: argparse.Namespace) -> None:
    from powerdiffusion.evaluation.results_report import generate_results_summary
    path = generate_results_summary()
    print(f"Saved: {path}")


def cmd_pipeline(args: argparse.Namespace) -> None:
    from powerdiffusion.data.preprocess import run_preprocessing
    from powerdiffusion.training.sweep import run_sweep
    from powerdiffusion.sampling.generate import generate_for_run
    from powerdiffusion.visualization.plots import run_all_validation_plots
    from powerdiffusion.evaluation.failure_analysis import analyze, write_failure_analysis_md
    from powerdiffusion.evaluation.results_report import generate_results_summary

    if not args.skip_preprocess:
        run_preprocessing()

    df = run_sweep(max_epochs=args.epochs, full_grid=args.full_grid)
    best_row = df.sort_values("best_val_loss").iloc[0]
    run_id = best_row["run_id"]

    generate_for_run(
        run_id=run_id,
        diffusion_steps=int(best_row["diffusion_steps"]),
        hidden_dim=int(best_row["hidden_dim"]),
        n_layers=int(best_row["n_layers"]),
    )
    run_all_validation_plots(run_id)
    result = analyze(run_id)
    write_failure_analysis_md(result)
    generate_results_summary()
    print(f"\nPipeline complete. Best run: {run_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="powerdiffusion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_preprocess = subparsers.add_parser("preprocess", help="Build X/Z tensors from raw CSVs")
    p_preprocess.set_defaults(func=cmd_preprocess)

    p_train = subparsers.add_parser("train", help="Train a single model configuration")
    _add_run_config_args(p_train)
    p_train.set_defaults(func=cmd_train)

    p_sweep = subparsers.add_parser("sweep", help="Run the hyperparameter sweep")
    p_sweep.add_argument("--epochs", type=int, default=60)
    p_sweep.add_argument("--full-grid", action="store_true")
    p_sweep.set_defaults(func=cmd_sweep)

    p_generate = subparsers.add_parser("generate", help="Generate synthetic profiles from a trained checkpoint")
    _add_run_config_args(p_generate)
    p_generate.set_defaults(func=cmd_generate)

    p_evaluate = subparsers.add_parser("evaluate", help="Run validation plots + failure analysis for a run")
    p_evaluate.add_argument("--run-id", required=True)
    p_evaluate.set_defaults(func=cmd_evaluate)

    p_report = subparsers.add_parser("report", help="Regenerate reports/results_summary.md")
    p_report.set_defaults(func=cmd_report)

    p_pipeline = subparsers.add_parser("pipeline", help="Run the full pipeline end-to-end")
    p_pipeline.add_argument("--epochs", type=int, default=60)
    p_pipeline.add_argument("--full-grid", action="store_true")
    p_pipeline.add_argument("--skip-preprocess", action="store_true")
    p_pipeline.set_defaults(func=cmd_pipeline)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
