"""
examples/pjm_load/run_benchmark.py
==================================
Reproducible classical-vs-diffusion comparison, built entirely from
`enerdatasynth` library code (`evaluation.baselines`,
`evaluation.metrics`, `generators.LoadDiffusionModel`) — no logic is
duplicated here beyond argument parsing and table formatting. This is
the successor to the research lineage's `week5_benchmark.py` /
`week7.py --task evaluate` head-to-head.

Usage:
    python examples/pjm_load/run_benchmark.py --run-id transformer_T1000_H64_L2

Reproduce:
    1. From a clone of this repository: `pip install -e ".[research]"`.
    2. Get the PJM and NOAA source files yourself (sources: data/README.md) and put
       them in data/raw/pjm_load/. The repository ships no real data.
    3. `enerdatasynth preprocess`
    4. `python examples/pjm_load/run_benchmark.py --run-id transformer_T1000_H64_L2`
    Output: data/reports/pjm_load/benchmark_comparison.csv (needs `pip install -e ".[research]"` for statsmodels).
"""

from __future__ import annotations
import argparse
import csv

import numpy as np

from enerdatasynth.utils.config import load_processed, REPORTS_DIR
from enerdatasynth.generators import LoadDiffusionModel
from enerdatasynth.resources.pjm_load.evaluation.baselines import generate_all_baselines
from enerdatasynth.resources.pjm_load.evaluation.metrics import compute_metrics, METRIC_KEYS


def run_benchmark(run_id: str, n_days: int | None = None) -> dict:
    X_train = load_processed("X_daily_profiles.npy")
    X_val   = load_processed("X_daily_profiles_val.npy")
    Z_val   = load_processed("Z_conditions_val.npy")

    if n_days is not None:
        X_val, Z_val = X_val[:n_days], Z_val[:n_days]
    n_days = len(X_val)

    print(f"Fitting classical baselines ({n_days} days)...")
    classical_samples, classical_runtimes = generate_all_baselines(X_train, n_days)

    print(f"Generating from {run_id}...")
    model = LoadDiffusionModel.from_run_id(run_id)
    diffusion_samples = model.sample(Z_val, seed=0)

    all_samples = {**classical_samples, run_id: diffusion_samples}
    results = {name: compute_metrics(X_val, samples) for name, samples in all_samples.items()}

    print(f"\n{'Model':<28}" + "".join(f"{k:>12}" for k in METRIC_KEYS))
    for name, metrics in results.items():
        print(f"{name:<28}" + "".join(f"{metrics[k]:>12.4f}" for k in METRIC_KEYS))

    out_path = REPORTS_DIR / "benchmark_comparison.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + METRIC_KEYS)
        writer.writeheader()
        for name, metrics in results.items():
            writer.writerow({"model": name, **{k: round(v, 4) for k, v in metrics.items()}})
    print(f"\nSaved: {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="e.g. transformer_T1000_H64_L2")
    parser.add_argument("--n-days", type=int, default=None, help="Subsample the validation set for a faster run")
    args = parser.parse_args()
    run_benchmark(args.run_id, args.n_days)
