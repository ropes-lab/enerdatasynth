from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wind_diffusion.plotting import (
    compare_to_real,
    plot_acf,
    plot_distributions,
    plot_envelope,
    plot_first_diff,
    plot_sanity_day,
)


def _fake_days(n: int, seq_len: int = 48, seed: int = 0, level: float = 8.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, seq_len)
    diurnal = level + 3.0 * np.sin(t)
    return diurnal[None, :] + rng.normal(0, 0.5, size=(n, seq_len))


def test_plot_distributions_writes_file(tmp_path):
    real = _fake_days(20, seed=1)
    synth = _fake_days(15, seed=2)
    out = tmp_path / "dist.png"
    plot_distributions(real, synth, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_first_diff_writes_file(tmp_path):
    real = _fake_days(20, seed=1)
    synth = _fake_days(15, seed=2)
    out = tmp_path / "diff.png"
    plot_first_diff(real, synth, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_acf_writes_file(tmp_path):
    real = _fake_days(20, seed=1)
    synth = _fake_days(15, seed=2)
    out = tmp_path / "acf.png"
    plot_acf(real, synth, out, max_lag=10)
    assert out.exists() and out.stat().st_size > 0


def test_plot_sanity_day_writes_file(tmp_path):
    real = _fake_days(20, seed=1)
    synth = _fake_days(15, seed=2)
    out = tmp_path / "sanity.png"
    plot_sanity_day(real, synth, out, day_index=0)
    assert out.exists() and out.stat().st_size > 0


def test_plot_envelope_writes_file_without_time_index(tmp_path):
    real = _fake_days(20, seed=1)
    synth = _fake_days(15, seed=2)
    out = tmp_path / "envelope.png"
    plot_envelope(real, synth, out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_envelope_writes_file_with_time_index_and_explicit_reference(tmp_path):
    real = _fake_days(20, seed=1)
    synth = _fake_days(15, seed=2, seq_len=48)
    time_index = pd.date_range("2026-01-01", periods=48, freq="5min")
    out = tmp_path / "envelope_time.png"
    plot_envelope(real, synth, out, time_index=time_index, real_reference_day=real[0])
    assert out.exists() and out.stat().st_size > 0


def test_compare_to_real_produces_all_five_plots(tmp_path):
    real = _fake_days(20, seed=1)
    synth = _fake_days(15, seed=2)
    paths = compare_to_real(real, synth, tmp_path, prefix="unit")
    assert set(paths.keys()) == {"distributions", "first_diff", "acf", "sanity_day", "envelope"}
    for name, path in paths.items():
        p = Path(path)
        assert p.exists() and p.stat().st_size > 0, f"{name} plot missing/empty"
        assert p.parent == tmp_path
        assert p.name.startswith("unit_")


def test_compare_to_real_creates_output_dir_if_missing(tmp_path):
    nested = tmp_path / "nested" / "dir"
    real = _fake_days(10, seed=3)
    synth = _fake_days(10, seed=4)
    paths = compare_to_real(real, synth, nested)
    assert nested.exists()
    for path in paths.values():
        assert Path(path).exists()
