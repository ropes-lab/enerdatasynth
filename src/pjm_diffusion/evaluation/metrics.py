"""
src/pjm_diffusion/evaluation/metrics.py
==========================================
Canonical numeric comparison metrics between real and generated load
profiles.

This computation was duplicated across the research-history scripts
with at least four incompatible interfaces: a dict with one key
schema, a dict with different key names, six standalone functions,
and a 3-tuple covering only half the metrics (see the Phase 1 audit's
"Cross-file duplication summary"). The version below reproduces the
math verbatim from the mutually-identical `metrics()`/`_acf()` pair
in week6.py / week7.py / arch_compare.py — the most widely used and
internally consistent of the variants — and is now the single place
this computation lives.
"""

from __future__ import annotations
import numpy as np

METRIC_KEYS = ["Hourly", "FirstDiff", "ACF", "RMSE", "Peak", "Energy", "Ramp"]


def autocorrelation(profiles: np.ndarray, max_lag: int = 12) -> np.ndarray:
    """Mean autocorrelation (lags 0..max_lag) across all rows of `profiles`."""
    out = []
    for row in profiles:
        row = row - row.mean()
        denom = (row * row).sum()
        if denom <= 0:
            continue
        out.append([(row[:len(row) - k] * row[k:]).sum() / denom for k in range(max_lag + 1)])
    return np.array(out).mean(0) if out else np.zeros(max_lag + 1)


def compute_metrics(real: np.ndarray, generated: np.ndarray) -> dict:
    """
    The 7-metric comparison suite used throughout the sweep/evaluation
    scripts: hourly-distribution error, first-difference error,
    ACF error, RMSE of the mean profile, peak error, daily-energy
    error, and ramp-rate error.
    """
    r, g = real, generated
    hourly = (
        np.abs(r.mean(0) - g.mean(0)).mean()
        + np.abs(
            (np.percentile(r, 75, 0) - np.percentile(r, 25, 0))
            - (np.percentile(g, 75, 0) - np.percentile(g, 25, 0))
        ).mean()
    )
    first_diff = abs(np.diff(r, axis=1).std() - np.diff(g, axis=1).std())
    acf_err = np.abs(autocorrelation(r) - autocorrelation(g)).mean()
    rmse = np.sqrt(((r.mean(0) - g.mean(0)) ** 2).mean())
    peak = abs(r.max(1).mean() - g.max(1).mean())
    energy = abs(r.sum(1).mean() - g.sum(1).mean())
    ramp = abs(np.abs(np.diff(r, axis=1)).mean() - np.abs(np.diff(g, axis=1)).mean())
    return dict(Hourly=hourly, FirstDiff=first_diff, ACF=acf_err, RMSE=rmse,
                Peak=peak, Energy=energy, Ramp=ramp)
