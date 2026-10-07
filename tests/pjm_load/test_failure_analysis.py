"""Tests for enerdatasynth.resources.pjm_load.evaluation.failure_analysis.

Mirrors tests/wind/test_failure_analysis.py. Runs against the real bundled
validation set and the real generated sample set for the default checkpoint
(no mocks), writing its report to a temp dir so the tracked report in
data/reports/pjm_load/ is left untouched.
"""

from __future__ import annotations

import pytest

from enerdatasynth.resources.pjm_load.evaluation import failure_analysis as fa
from enerdatasynth.utils.config import GENERATED_DIR

RUN_ID = "T100_H256_L4"

needs_bundled_samples = pytest.mark.skipif(
    not (GENERATED_DIR / f"X_generated_{RUN_ID}.npy").exists(),
    reason="bundled generated sample set not present",
)

ISSUE_KEYS = {"issue", "metric", "possible_cause", "potential_future_fix"}


@needs_bundled_samples
def test_analyze_returns_structured_result():
    result = fa.analyze(RUN_ID)
    assert result["run_id"] == RUN_ID
    assert isinstance(result["issues"], list)
    for issue in result["issues"]:
        assert ISSUE_KEYS <= set(issue)


def test_analyze_missing_generated_samples_raises():
    with pytest.raises(FileNotFoundError):
        fa.analyze("no_such_run_id")


@needs_bundled_samples
def test_write_failure_analysis_md_creates_real_file(tmp_path, monkeypatch):
    monkeypatch.setattr(fa, "REPORTS_DIR", tmp_path)
    path = fa.write_failure_analysis_md(fa.analyze(RUN_ID))
    assert path.parent == tmp_path
    text = path.read_text()
    assert text.startswith("# Failure Analysis")
    assert RUN_ID in text


def test_write_failure_analysis_md_handles_no_issues(tmp_path, monkeypatch):
    monkeypatch.setattr(fa, "REPORTS_DIR", tmp_path)
    path = fa.write_failure_analysis_md({"run_id": "x", "issues": []})
    assert "No failure modes detected" in path.read_text()
