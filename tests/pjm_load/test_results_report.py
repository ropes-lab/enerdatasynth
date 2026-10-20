"""Tests for enerdatasynth.resources.pjm_load.evaluation.results_report.

Mirrors tests/wind/test_results_report.py. Uses the shipped scaler
metadata and the real sweep results CSV, writing to a temp dir so the
tracked report in data/reports/pjm_load/ is left untouched.
"""

from __future__ import annotations

import shutil

import pytest

from enerdatasynth.resources.pjm_load.evaluation import results_report as rr
from enerdatasynth.utils.config import REPORTS_DIR

SWEEP_CSV = REPORTS_DIR / "hyperparameter_results.csv"


def test_generate_results_summary_reports_dataset_and_best_run(tmp_path, monkeypatch):
    if not SWEEP_CSV.exists():
        pytest.skip("bundled sweep results not present")
    shutil.copy(SWEEP_CSV, tmp_path / "hyperparameter_results.csv")
    monkeypatch.setattr(rr, "REPORTS_DIR", tmp_path)

    path = rr.generate_results_summary()
    assert path.parent == tmp_path
    text = path.read_text()
    assert text.startswith("# Results Summary")
    assert "Training samples: 2785" in text
    assert "Validation samples: 600" in text
    # the shipped default and the sweep's own winner are reported separately
    assert "Shipped Default Checkpoint" in text
    assert "transformer_T1000_H64_L2" in text
    assert "Best sweep configuration: **T100_H256_L4**" in text  # lowest val loss in the bundled MLP sweep


def test_generate_results_summary_without_sweep_results(tmp_path, monkeypatch):
    monkeypatch.setattr(rr, "REPORTS_DIR", tmp_path)
    text = rr.generate_results_summary().read_text()
    assert "Dataset Summary" in text
    assert "Shipped Default Checkpoint" in text
    assert "Best sweep configuration" not in text
