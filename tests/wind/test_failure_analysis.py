"""Tests for enerdatasynth.resources.wind.evaluation.failure_analysis — added for
parity with pjm_load's failure_analysis.py."""

from __future__ import annotations

from enerdatasynth.resources.wind.evaluation.failure_analysis import analyze, write_failure_analysis_md


def test_analyze_returns_real_flagged_issues_for_known_weak_regimes():
    # mid-regime steering is a real, documented weakness for 1ch/3ch (see
    # docs/models/wind.md §8) -- this must surface as a flagged issue, not be
    # silently missed.
    result = analyze("1ch")
    assert result["preset"] == "1ch"
    assert len(result["issues"]) > 0


def test_analyze_2ch_uses_top_level_known_limitation_fallback():
    # 2ch's manifest entry has no structured "metrics" dict -- its real history
    # lives in a top-level known_limitation string instead. This must still
    # surface as a flagged issue, not silently report zero.
    result = analyze("2ch")
    assert len(result["issues"]) >= 1


def test_write_failure_analysis_md_creates_real_file(tmp_path, monkeypatch):
    import enerdatasynth.resources.wind.evaluation.failure_analysis as fa_module

    monkeypatch.setattr(fa_module, "DATA_ROOT", tmp_path)
    result = analyze("3ch")
    path = write_failure_analysis_md(result)
    assert path.exists()
    text = path.read_text()
    assert "3ch" in text
