"""Tests for pjm_diffusion.evaluation.calibration: bias measurement and correction."""

from __future__ import annotations
import numpy as np

from pjm_diffusion.evaluation.calibration import (
    compute_bias, apply_calibration, save_calibration, load_calibration,
)


def test_compute_bias_recovers_known_scalar_offset():
    real = np.full((10, 24), 0.5, dtype=np.float32)
    generated = real - 0.02  # generator sits low by a known constant
    bias = compute_bias(real, generated)
    assert abs(bias["scalar"] - 0.02) < 1e-6
    assert np.allclose(bias["per_hour"], 0.02, atol=1e-6)


def test_apply_calibration_scalar_shifts_every_value():
    generated = np.zeros((3, 24), dtype=np.float32)
    calibration = {"scalar": 0.05, "per_hour": [0.0] * 24}
    corrected = apply_calibration(generated, calibration, mode="scalar")
    assert np.allclose(corrected, 0.05)


def test_apply_calibration_per_hour_shifts_columns_independently():
    generated = np.zeros((3, 24), dtype=np.float32)
    per_hour = list(np.arange(24) * 0.01)
    calibration = {"scalar": 0.0, "per_hour": per_hour}
    corrected = apply_calibration(generated, calibration, mode="per_hour")
    assert np.allclose(corrected[0], per_hour)
    assert np.allclose(corrected[1], per_hour)  # same correction applied to every row


def test_apply_calibration_unknown_mode_raises():
    try:
        apply_calibration(np.zeros((1, 24)), {"scalar": 0.0}, mode="bogus")
        assert False, "should have raised"
    except ValueError:
        pass


def test_save_and_load_calibration_round_trip(tmp_path):
    calibration = {"scalar": 0.01234, "per_hour": [0.0] * 24, "mode_recommended": "scalar"}
    path = tmp_path / "calibration.json"
    save_calibration(path, calibration)
    reloaded = load_calibration(path)
    assert reloaded == calibration


def test_load_calibration_missing_file_returns_none(tmp_path):
    assert load_calibration(tmp_path / "does_not_exist.json") is None


def test_calibrate_from_samples_reports_scalar_recommended():
    """The research evidence (see MIGRATION.md) showed per-hour calibration is
    actually worse on several metrics; this module must keep recommending scalar."""
    import torch
    from pjm_diffusion.evaluation.calibration import calibrate_from_samples
    from pjm_diffusion.diffusion.ddpm import DDPM
    from pjm_diffusion.models.denoiser import SimpleMLPDenoiser

    rng = np.random.default_rng(0)
    X_train = rng.uniform(0, 1, size=(20, 24)).astype(np.float32)
    Z_train = rng.normal(size=(20, 23)).astype(np.float32)
    model = SimpleMLPDenoiser(x_dim=24, cond_dim=23, hidden_dim=8, n_layers=1)
    ddpm = DDPM(T=5)

    result = calibrate_from_samples(model, ddpm, X_train, Z_train, n_fit=10, seed=0)
    assert result["mode_recommended"] == "scalar"
    assert result["n_days_used"] == 10
    assert isinstance(result["scalar"], float)
    assert len(result["per_hour"]) == 24
