"""Fidelity gates (B5) for comparing synthetic wind-profile days against a
real held-out validation set.

Every gate takes `real_days`/`synth_days` as `(N, seq_len)` arrays of
physical-unit windspeed values (m/s), one row per day, `seq_len=288` at a
5-minute cadence. Lower is better for every distance/RMSE-shaped gate;
`regime_accuracy` is a fraction in [0, 1] where higher is better.

Design notes:
- `acf_rmse` reuses `evaluate.autocorr` (per-day, mean-subtracted,
  normalized by the day's own variance) rather than reimplementing
  autocorrelation. The "mean ACF" for a group of days is the elementwise
  mean of each day's ACF curve (matches `evaluate.plot_eval_acf`'s
  convention) -- concatenating days into one long series before computing
  a single ACF would blend in spurious day-boundary discontinuities, so
  per-day-then-average is preferred and is documented here explicitly.
- `spectral_log_ratio` averages *power* (not log-power) across days before
  taking the log-ratio, since averaging in the log domain would bias the
  result (Jensen's inequality) -- log(mean(power)) is the physically
  meaningful "typical power at this frequency", not mean(log(power)).
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats

from enerdatasynth.resources.wind.evaluation.evaluate import autocorr
from enerdatasynth.resources.wind.data.transforms import WIND_BUCKET_LABELS, assign_wind_bucket

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Marginal distribution
# ---------------------------------------------------------------------------


def marginal_wasserstein(real_days: np.ndarray, synth_days: np.ndarray) -> float:
    """Wasserstein-1 distance between the flattened windspeed value
    distributions of real vs synthetic days (units: m/s)."""
    real_flat = np.asarray(real_days).reshape(-1)
    synth_flat = np.asarray(synth_days).reshape(-1)
    return float(scipy_stats.wasserstein_distance(real_flat, synth_flat))


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------


def _mean_acf(days: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean ACF curve (length max_lag+1, lag 0..max_lag) across `days`, each
    day's ACF computed independently via `evaluate.autocorr` then averaged
    elementwise. See module docstring for why per-day-then-average is used
    instead of concatenating days into one long series.
    """
    acfs = [autocorr(day, max_lag) for day in days]
    return np.mean(acfs, axis=0)


def acf_rmse(real_days: np.ndarray, synth_days: np.ndarray, max_lag: int = 288) -> float:
    """RMSE between the mean real ACF curve and the mean synthetic ACF
    curve, over lags 1..max_lag (lag 0 is trivially 1.0 for both and is
    excluded so it can't dilute the metric)."""
    real_days = np.asarray(real_days)
    synth_days = np.asarray(synth_days)
    max_lag = min(max_lag, real_days.shape[1] - 1, synth_days.shape[1] - 1)

    real_acf = _mean_acf(real_days, max_lag)
    synth_acf = _mean_acf(synth_days, max_lag)

    diff = real_acf[1:] - synth_acf[1:]
    return float(np.sqrt(np.mean(diff**2)))


# ---------------------------------------------------------------------------
# Spectral content
# ---------------------------------------------------------------------------


def spectral_log_ratio(
    real_days: np.ndarray,
    synth_days: np.ndarray,
    sample_spacing_minutes: int = 5,
    high_band_period_hours: float = 1.0,
) -> dict:
    """Compare average power spectra of real vs synthetic days, split into a
    "low band" (periods >= high_band_period_hours, i.e. slow/diurnal-ish
    variation) and a "high band" (periods < high_band_period_hours, i.e.
    fast turbulent fluctuation).

    Returns {"low_band": x, "high_band": y}, each `mean(|log(real_psd /
    synth_psd)|)` over that frequency range (small epsilon guards against
    log(0)/div-by-0 in near-silent frequency bins).
    """
    real_days = np.asarray(real_days)
    synth_days = np.asarray(synth_days)

    fs = 1.0  # cycles per sample; periods below are expressed in samples too, so this cancels out

    freqs_real, psd_real_per_day = scipy_signal.periodogram(real_days, fs=fs, axis=1)
    freqs_synth, psd_synth_per_day = scipy_signal.periodogram(synth_days, fs=fs, axis=1)

    psd_real = psd_real_per_day.mean(axis=0)
    psd_synth = psd_synth_per_day.mean(axis=0)

    # High-band cutoff frequency: periods < high_band_period_hours means
    # frequency > 1 / (period_in_samples), where period_in_samples =
    # high_band_period_hours * 60 / sample_spacing_minutes.
    period_samples_cutoff = high_band_period_hours * 60.0 / sample_spacing_minutes
    freq_cutoff = 1.0 / period_samples_cutoff  # cycles/sample, e.g. 1/12 for 1h @ 5min

    log_ratio = np.log((psd_real + _EPS) / (psd_synth + _EPS))
    abs_log_ratio = np.abs(log_ratio)

    # freqs_real == freqs_synth (same seq_len / fs); use freqs_real. Exclude
    # the DC bin (freq==0) from both bands -- it's the day's mean level,
    # already covered by diurnal_rmse/marginal_wasserstein.
    nonzero = freqs_real > 0
    low_mask = nonzero & (freqs_real <= freq_cutoff)
    high_mask = nonzero & (freqs_real > freq_cutoff)

    low_band = float(np.mean(abs_log_ratio[low_mask])) if np.any(low_mask) else 0.0
    high_band = float(np.mean(abs_log_ratio[high_mask])) if np.any(high_mask) else 0.0

    return {"low_band": low_band, "high_band": high_band}


# ---------------------------------------------------------------------------
# Ramps (first differences)
# ---------------------------------------------------------------------------


def ramp_wasserstein(real_days: np.ndarray, synth_days: np.ndarray) -> float:
    """Wasserstein-1 distance between the flattened within-day first-
    difference distributions of real vs synthetic days."""
    real_diff = np.diff(np.asarray(real_days), axis=1).reshape(-1)
    synth_diff = np.diff(np.asarray(synth_days), axis=1).reshape(-1)
    return float(scipy_stats.wasserstein_distance(real_diff, synth_diff))


# ---------------------------------------------------------------------------
# Diurnal profile
# ---------------------------------------------------------------------------


def diurnal_rmse(real_days: np.ndarray, synth_days: np.ndarray) -> float:
    """RMSE between the mean daily profile curves (mean across days at each
    step-in-day) of real vs synthetic days."""
    real_profile = np.asarray(real_days).mean(axis=0)
    synth_profile = np.asarray(synth_days).mean(axis=0)
    diff = real_profile - synth_profile
    return float(np.sqrt(np.mean(diff**2)))


# ---------------------------------------------------------------------------
# Regime conditioning accuracy
# ---------------------------------------------------------------------------


def regime_accuracy(
    daily_means: np.ndarray,
    requested_regime: str,
    wind_bucket_edges: tuple[float, float],
) -> float:
    """Fraction of `daily_means` (one mean windspeed per synthetic day) that
    fall in the `requested_regime` wind bucket (per
    `transforms.assign_wind_bucket`)."""
    daily_means = np.asarray(daily_means)
    if daily_means.size == 0:
        return 0.0
    q1, q2 = wind_bucket_edges
    buckets = [assign_wind_bucket(float(m), q1, q2) for m in daily_means]
    matches = sum(1 for b in buckets if b == requested_regime)
    return float(matches) / len(buckets)


def regime_confusion(
    daily_means: np.ndarray,
    wind_bucket_edges: tuple[float, float],
) -> dict[str, float]:
    """Fraction of `daily_means` landing in each of low/mid/high, regardless
    of which regime was requested.

    `regime_accuracy` alone only reports the diagonal of this (the fraction
    matching the requested regime) -- it can't distinguish "conditioning has
    no effect" from "conditioning pushes generation toward the wrong bucket".
    Also note the buckets are NOT equal-sized: `wind_bucket_edges` is a
    (25th, 75th) percentile split, so an unconditioned model sampling the
    real marginal distribution would land here at roughly
    {"low": 0.25, "mid": 0.50, "high": 0.25} by construction -- that, not
    1/3 each, is the right "no signal" baseline to compare against.
    """
    daily_means = np.asarray(daily_means)
    if daily_means.size == 0:
        return {label: 0.0 for label in WIND_BUCKET_LABELS}
    q1, q2 = wind_bucket_edges
    buckets = [assign_wind_bucket(float(m), q1, q2) for m in daily_means]
    n = len(buckets)
    return {label: buckets.count(label) / n for label in WIND_BUCKET_LABELS}


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


def compute_all_gates(
    real_days: np.ndarray,
    synth_days: np.ndarray,
    requested_regime: str,
    wind_bucket_edges: tuple[float, float],
) -> dict:
    """Bundle every fidelity gate into one flat dict for a single
    (real_days, synth_days) comparison."""
    real_days = np.asarray(real_days)
    synth_days = np.asarray(synth_days)

    spectral = spectral_log_ratio(real_days, synth_days)
    synth_daily_means = synth_days.mean(axis=1)

    return {
        "marginal_wasserstein": marginal_wasserstein(real_days, synth_days),
        "acf_rmse": acf_rmse(real_days, synth_days),
        "spectral_low_band": spectral["low_band"],
        "spectral_high_band": spectral["high_band"],
        "ramp_wasserstein": ramp_wasserstein(real_days, synth_days),
        "diurnal_rmse": diurnal_rmse(real_days, synth_days),
        "regime_accuracy": regime_accuracy(synth_daily_means, requested_regime, wind_bucket_edges),
        "regime_confusion": regime_confusion(synth_daily_means, wind_bucket_edges),
    }
