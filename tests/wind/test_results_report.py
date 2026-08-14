"""Tests for enerdatasynth.resources.wind.evaluation.results_report — added for
parity with pjm_load's results_report.py."""

from __future__ import annotations

from enerdatasynth.resources.wind.evaluation.results_report import generate_results_summary


def test_generate_results_summary_creates_real_file_covering_all_presets(tmp_path, monkeypatch):
    import enerdatasynth.resources.wind.evaluation.results_report as rr_module

    monkeypatch.setattr(rr_module, "DATA_ROOT", tmp_path)
    path = generate_results_summary()
    assert path.exists()
    text = path.read_text()
    for preset in ("1ch", "2ch", "3ch"):
        assert preset in text
    # 2ch has no structured metrics -- its real content must still appear
    # via the known_limitation fallback, not be silently dropped.
    assert "known-limitation" in text.lower() or "known_limitation" in text.lower()
