"""Tests for pjm_diffusion.evaluation.metrics: the canonical 7-metric comparison suite."""

from __future__ import annotations
import numpy as np

from pjm_diffusion.evaluation.metrics import compute_metrics, autocorrelation, METRIC_KEYS


def test_compute_metrics_zero_when_real_equals_generated(rng):
    X = rng.uniform(0, 1, size=(30, 24)).astype(np.float32)
    metrics = compute_metrics(X, X.copy())
    assert set(metrics) == set(METRIC_KEYS)
    for key, value in metrics.items():
        assert abs(value) < 1e-6, f"{key} should be ~0 for identical arrays, got {value}"


def test_compute_metrics_nonzero_for_different_distributions(rng):
    real = rng.uniform(0, 1, size=(30, 24)).astype(np.float32)
    generated = rng.uniform(0, 1, size=(30, 24)).astype(np.float32) * 0.2  # much smaller-scale
    metrics = compute_metrics(real, generated)
    assert metrics["Energy"] > 0
    assert metrics["Peak"] > 0


def test_autocorrelation_lag_zero_is_one(rng):
    """By construction (normalized by its own variance), lag-0 autocorrelation is exactly 1."""
    X = rng.normal(size=(10, 24)).astype(np.float32)
    acf = autocorrelation(X, max_lag=5)
    assert acf.shape == (6,)
    assert abs(acf[0] - 1.0) < 1e-6


def test_autocorrelation_handles_constant_rows_without_nan():
    """Constant rows have zero variance (denom<=0) and must be skipped, not produce NaN."""
    X = np.ones((5, 24), dtype=np.float32)
    acf = autocorrelation(X, max_lag=3)
    assert np.all(acf == 0)  # no valid rows contributed -> falls back to zeros
    assert not np.isnan(acf).any()
