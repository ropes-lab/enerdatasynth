"""
src/enerdatasynth/evaluation/baselines.py
=============================================
Classical time-series baselines (AR / ARMA / ARIMA / SARIMA) used to
benchmark the diffusion model against non-deep-learning alternatives.

Extracted from the identical `fit_and_sample` implementation and its
calling loop, duplicated across week5.py / week5_benchmark.py / week7.py
/ sampler_fixed.py (see research_history/ and the Phase 1 audit) into
one place. Math is unchanged from the original.
"""

from __future__ import annotations
import time
from typing import Tuple

import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# (order, seasonal_order) — the same four classical specs used
# throughout the research-history benchmark scripts.
ARIMA_SPECS = {
    "AR":     ((2, 0, 0), (0, 0, 0, 0)),
    "ARMA":   ((2, 0, 2), (0, 0, 0, 0)),
    "ARIMA":  ((2, 1, 2), (0, 0, 0, 0)),
    "SARIMA": ((1, 0, 1), (1, 1, 1, 12)),
}


def fit_and_sample(train_profiles: np.ndarray, order: tuple, seasonal_order: tuple,
                    n_days: int, steps: int = 24, label: str = "",
                    clip_range: Tuple[float, float] = (-0.2, 1.2)) -> Tuple[np.ndarray, float]:
    """
    Fit an ARIMA-family model to the mean training profile and
    simulate `n_days` synthetic daily profiles from it.

    Returns (samples, fit_and_sample_runtime_seconds).
    """
    t0 = time.time()
    mean_profile = train_profiles.mean(axis=0)
    model = ARIMA(mean_profile, order=order, seasonal_order=seasonal_order,
                  enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(method_kwargs={"maxiter": 200})
    sims = np.zeros((n_days, steps))
    for i in range(n_days):
        s = res.simulate(nsimulations=steps, anchor="start", repetitions=1).reshape(-1)[:steps]
        sims[i] = s
    sims = sims - sims.mean(axis=0, keepdims=True) + mean_profile[None, :]
    runtime = time.time() - t0
    if label:
        print(f"  {label:8s} fit+sample {n_days} days in {runtime:.1f}s")
    return np.clip(sims, *clip_range), runtime


def generate_all_baselines(train_profiles: np.ndarray, n_days: int,
                            specs: dict = ARIMA_SPECS) -> Tuple[dict, dict]:
    """
    Fit and sample every model in `specs`, falling back to a plain
    AR(2) if a fit fails (matches the fallback behavior used in the
    original benchmark scripts).

    Returns (samples_by_name, runtime_seconds_by_name).
    """
    samples, runtimes = {}, {}
    for name, (order, seasonal_order) in specs.items():
        try:
            s, rt = fit_and_sample(train_profiles, order, seasonal_order, n_days, label=name)
        except Exception as e:
            print(f"  {name} failed ({e}); falling back to AR(2)")
            s, rt = fit_and_sample(train_profiles, (2, 0, 0), (0, 0, 0, 0), n_days, label=name)
        samples[name], runtimes[name] = s, rt
    return samples, runtimes
