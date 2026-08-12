"""Sanity tests for wind_diffusion.metrics (the B5 fidelity gates).

For each gate: comparing a distribution against itself (or a matching
distribution drawn with the same generator) should give a near-zero
(near-1.0 for regime_accuracy) result; comparing against a deliberately
different distribution should give a clearly worse value. Thresholds are
relative (worse > same_case + margin), not fragile hardcoded constants.
"""

from __future__ import annotations

import numpy as np
import pytest

from wind_diffusion.metrics import (
    acf_rmse,
    compute_all_gates,
    diurnal_rmse,
    marginal_wasserstein,
    ramp_wasserstein,
    regime_accuracy,
    regime_confusion,
    spectral_log_ratio,
)

SEQ_LEN = 288


def _make_smooth_days(n_days: int, seed: int, mean: float = 9.0, amplitude: float = 3.0) -> np.ndarray:
    """Smooth diurnal-like synthetic days: a sine cycle plus small noise,
    representative of a real wind-speed day."""
    rng = np.random.default_rng(seed)
    t = np.arange(SEQ_LEN)
    days = []
    for _ in range(n_days):
        phase = rng.uniform(0, 2 * np.pi)
        day = mean + amplitude * np.sin(2 * np.pi * t / SEQ_LEN + phase) + rng.normal(0, 0.3, SEQ_LEN)
        days.append(np.clip(day, 0.0, None))
    return np.array(days, dtype=np.float64)


def _make_shifted_days(n_days: int, seed: int) -> np.ndarray:
    """Same shape family but with a shifted mean and larger amplitude --
    should read as clearly different from `_make_smooth_days`'s default."""
    return _make_smooth_days(n_days, seed, mean=18.0, amplitude=6.0)


def _make_pure_noise_days(n_days: int, seed: int, mean: float = 9.0, scale: float = 4.0) -> np.ndarray:
    """White noise around the same mean -- no autocorrelation/spectral
    structure at all, unlike the smooth sine-based days."""
    rng = np.random.default_rng(seed)
    days = rng.normal(mean, scale, size=(n_days, SEQ_LEN))
    return np.clip(days, 0.0, None)


# ---------------------------------------------------------------------------
# marginal_wasserstein
# ---------------------------------------------------------------------------


def test_marginal_wasserstein_near_zero_for_same_distribution():
    real = _make_smooth_days(40, seed=1)
    synth = _make_smooth_days(40, seed=2)  # same generator, different draw
    same_case = marginal_wasserstein(real, synth)

    shifted = _make_shifted_days(40, seed=3)
    worse = marginal_wasserstein(real, shifted)

    assert same_case < 1.0
    assert worse > same_case + 1.0


def test_marginal_wasserstein_zero_for_identical_arrays():
    days = _make_smooth_days(20, seed=5)
    assert marginal_wasserstein(days, days) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# acf_rmse
# ---------------------------------------------------------------------------


def test_acf_rmse_low_for_same_process_high_for_noise():
    real = _make_smooth_days(40, seed=1)
    synth = _make_smooth_days(40, seed=2)
    same_case = acf_rmse(real, synth, max_lag=50)

    noise = _make_pure_noise_days(40, seed=3)
    worse = acf_rmse(real, noise, max_lag=50)

    assert same_case < 0.3
    assert worse > same_case + 0.2


def test_acf_rmse_zero_for_identical_arrays():
    days = _make_smooth_days(20, seed=5)
    assert acf_rmse(days, days, max_lag=30) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# spectral_log_ratio
# ---------------------------------------------------------------------------


def test_spectral_log_ratio_near_zero_for_same_process():
    real = _make_smooth_days(40, seed=1)
    synth = _make_smooth_days(40, seed=2)
    same_case = spectral_log_ratio(real, synth)

    assert same_case["low_band"] >= 0.0
    assert same_case["high_band"] >= 0.0

    noise = _make_pure_noise_days(40, seed=3)
    worse = spectral_log_ratio(real, noise)

    # Pure white noise has a flat spectrum (all power in the high band)
    # versus a smooth diurnal sine (power concentrated at low frequency) --
    # the low-band mismatch should be far larger than the same-process case.
    assert worse["low_band"] > same_case["low_band"] + 0.5


def test_spectral_log_ratio_zero_for_identical_arrays():
    days = _make_smooth_days(20, seed=5)
    result = spectral_log_ratio(days, days)
    assert result["low_band"] == pytest.approx(0.0, abs=1e-6)
    assert result["high_band"] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# ramp_wasserstein
# ---------------------------------------------------------------------------


def test_ramp_wasserstein_low_for_same_process_high_for_noise():
    real = _make_smooth_days(40, seed=1)
    synth = _make_smooth_days(40, seed=2)
    same_case = ramp_wasserstein(real, synth)

    noise = _make_pure_noise_days(40, seed=3)
    worse = ramp_wasserstein(real, noise)

    assert worse > same_case + 0.5


def test_ramp_wasserstein_zero_for_identical_arrays():
    days = _make_smooth_days(20, seed=5)
    assert ramp_wasserstein(days, days) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# diurnal_rmse
# ---------------------------------------------------------------------------


def test_diurnal_rmse_low_for_same_distribution_high_for_shifted():
    real = _make_smooth_days(60, seed=1)
    synth = _make_smooth_days(60, seed=2)
    same_case = diurnal_rmse(real, synth)

    shifted = _make_shifted_days(60, seed=3)
    worse = diurnal_rmse(real, shifted)

    assert worse > same_case + 1.0


def test_diurnal_rmse_zero_for_identical_arrays():
    days = _make_smooth_days(20, seed=5)
    assert diurnal_rmse(days, days) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# regime_accuracy
# ---------------------------------------------------------------------------


def test_regime_accuracy_high_when_means_match_requested_regime():
    # wind_bucket_edges = (q1, q2); "mid" days should have mean strictly
    # between the edges.
    edges = (6.0, 12.0)
    mid_means = np.array([7.0, 8.0, 9.0, 10.0, 11.0])
    acc_same = regime_accuracy(mid_means, "mid", edges)
    assert acc_same == pytest.approx(1.0)

    # "high" means requested against "mid" -> should score far worse.
    high_means = np.array([15.0, 16.0, 17.0, 18.0, 20.0])
    acc_worse = regime_accuracy(high_means, "mid", edges)
    assert acc_worse == pytest.approx(0.0)
    assert acc_same > acc_worse


def test_regime_accuracy_empty_array_returns_zero():
    assert regime_accuracy(np.array([]), "mid", (6.0, 12.0)) == 0.0


def test_regime_confusion_all_mid_when_means_are_mid():
    edges = (6.0, 12.0)
    mid_means = np.array([7.0, 8.0, 9.0, 10.0, 11.0])
    confusion = regime_confusion(mid_means, edges)
    assert set(confusion.keys()) == {"low", "mid", "high"}
    assert confusion["mid"] == pytest.approx(1.0)
    assert confusion["low"] == pytest.approx(0.0)
    assert confusion["high"] == pytest.approx(0.0)


def test_regime_confusion_splits_across_buckets():
    edges = (6.0, 12.0)
    means = np.array([3.0, 3.0, 9.0, 15.0])  # 2 low, 1 mid, 1 high
    confusion = regime_confusion(means, edges)
    assert confusion["low"] == pytest.approx(0.5)
    assert confusion["mid"] == pytest.approx(0.25)
    assert confusion["high"] == pytest.approx(0.25)
    assert sum(confusion.values()) == pytest.approx(1.0)


def test_regime_confusion_empty_array_returns_all_zero():
    confusion = regime_confusion(np.array([]), (6.0, 12.0))
    assert confusion == {"low": 0.0, "mid": 0.0, "high": 0.0}


# ---------------------------------------------------------------------------
# compute_all_gates
# ---------------------------------------------------------------------------


def test_compute_all_gates_returns_expected_keys_and_reasonable_values():
    real = _make_smooth_days(50, seed=1, mean=9.0)
    synth = _make_smooth_days(50, seed=2, mean=9.0)

    gates = compute_all_gates(real, synth, requested_regime="mid", wind_bucket_edges=(6.0, 12.0))

    expected_keys = {
        "marginal_wasserstein",
        "acf_rmse",
        "spectral_low_band",
        "spectral_high_band",
        "ramp_wasserstein",
        "diurnal_rmse",
        "regime_accuracy",
        "regime_confusion",
    }
    assert set(gates.keys()) == expected_keys
    for key, value in gates.items():
        if key == "regime_confusion":
            assert isinstance(value, dict)
            assert sum(value.values()) == pytest.approx(1.0)
            continue
        assert isinstance(value, float)
        assert value >= 0.0

    assert 0.0 <= gates["regime_accuracy"] <= 1.0


def test_compute_all_gates_worse_for_mismatched_distribution():
    real = _make_smooth_days(50, seed=1, mean=9.0)
    same = _make_smooth_days(50, seed=2, mean=9.0)
    shifted = _make_shifted_days(50, seed=3)

    gates_same = compute_all_gates(real, same, requested_regime="mid", wind_bucket_edges=(6.0, 12.0))
    gates_shifted = compute_all_gates(real, shifted, requested_regime="mid", wind_bucket_edges=(6.0, 12.0))

    assert gates_shifted["marginal_wasserstein"] > gates_same["marginal_wasserstein"]
    assert gates_shifted["diurnal_rmse"] > gates_same["diurnal_rmse"]
    assert gates_shifted["ramp_wasserstein"] > gates_same["ramp_wasserstein"]
