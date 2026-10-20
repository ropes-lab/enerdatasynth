"""Tests for enerdatasynth.resources.pjm_load.evaluation.failure_analysis.

Mirrors tests/wind/test_failure_analysis.py. This repository ships no real PJM
data, so these tests build small synthetic "real" and "generated" sets in a temp
dir where the right answer is known: a faithful generator must flag nothing, a
flat one must be flagged for peak and ramp, and a noisy one for ramp.
"""

from __future__ import annotations

import numpy as np
import pytest

import enerdatasynth.utils.config as config_module
from enerdatasynth.resources.pjm_load.evaluation import failure_analysis as fa

N_DAYS = 240
RUN_ID = "synthetic"


@pytest.fixture
def synthetic_split(tmp_path, monkeypatch):
    """Point the analysis at a temp dir holding a synthetic validation set."""
    rng = np.random.default_rng(0)
    hours = np.arange(24)
    months = np.arange(N_DAYS) % 12
    daily_shape = 0.45 + 0.25 * np.sin(2 * np.pi * (hours - 6) / 24)
    X_real = daily_shape + 0.04 * rng.standard_normal((N_DAYS, 24))

    Z = np.zeros((N_DAYS, 23), dtype=np.float32)
    Z[np.arange(N_DAYS), months] = 1.0  # month one-hot in the first 12 columns

    np.save(tmp_path / "X_daily_profiles_val.npy", X_real)
    np.save(tmp_path / "Z_conditions_val.npy", Z)
    monkeypatch.setattr(config_module, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(fa, "GENERATED_DIR", tmp_path)
    return X_real, tmp_path, rng


def _save_generated(path, X_gen):
    np.save(path / f"X_generated_{RUN_ID}.npy", X_gen)


def test_faithful_generator_is_not_flagged(synthetic_split):
    X_real, path, rng = synthetic_split
    _save_generated(path, X_real + 0.005 * rng.standard_normal(X_real.shape))
    result = fa.analyze(RUN_ID)
    assert result["run_id"] == RUN_ID
    assert result["issues"] == []


def test_flat_generator_is_flagged_for_peak_and_ramp(synthetic_split):
    X_real, path, _ = synthetic_split
    _save_generated(path, np.tile(X_real.mean(axis=0), (N_DAYS, 1)) * 0.7)
    issues = {i["issue"] for i in fa.analyze(RUN_ID)["issues"]}
    assert "Peak underestimation" in issues
    assert "Ramp rate mismatch (too small)" in issues


def test_noisy_generator_is_flagged_for_ramp(synthetic_split):
    X_real, path, rng = synthetic_split
    _save_generated(path, X_real + 0.3 * rng.standard_normal(X_real.shape))
    issues = {i["issue"] for i in fa.analyze(RUN_ID)["issues"]}
    assert "Ramp rate mismatch (too large)" in issues


def test_every_issue_is_fully_described(synthetic_split):
    X_real, path, _ = synthetic_split
    _save_generated(path, np.tile(X_real.mean(axis=0), (N_DAYS, 1)) * 0.7)
    for issue in fa.analyze(RUN_ID)["issues"]:
        assert {"issue", "metric", "possible_cause", "potential_future_fix"} <= set(issue)


def test_missing_data_explains_how_to_get_it(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "PROCESSED_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="enerdatasynth preprocess"):
        fa.analyze(RUN_ID)


def test_write_failure_analysis_md_creates_real_file(synthetic_split, tmp_path, monkeypatch):
    X_real, path, _ = synthetic_split
    _save_generated(path, np.tile(X_real.mean(axis=0), (N_DAYS, 1)) * 0.7)
    out = tmp_path / "reports"
    out.mkdir()
    monkeypatch.setattr(fa, "REPORTS_DIR", out)
    written = fa.write_failure_analysis_md(fa.analyze(RUN_ID))
    assert written.parent == out
    text = written.read_text()
    assert text.startswith("# Failure Analysis")
    assert RUN_ID in text and "Peak underestimation" in text


def test_write_failure_analysis_md_handles_no_issues(tmp_path, monkeypatch):
    monkeypatch.setattr(fa, "REPORTS_DIR", tmp_path)
    path = fa.write_failure_analysis_md({"run_id": "x", "issues": []})
    assert "No failure modes detected" in path.read_text()
