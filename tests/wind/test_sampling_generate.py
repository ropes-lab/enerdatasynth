"""Tests for enerdatasynth.resources.wind.sampling.generate — added for structural
parity with pjm_load's sampling/generate.py. Kept deliberately light (few regimes,
n_per_regime=1): this wraps already-tested generation, so it only needs to verify
the wrapper's own loop/seed-offset logic, not re-verify generation itself."""

from __future__ import annotations

import numpy as np

from enerdatasynth.resources.wind.sampling.generate import generate_for_regimes


def test_generate_for_regimes_returns_one_profile_per_regime():
    profiles = generate_for_regimes(preset="1ch", regimes=("low", "high"), n_per_regime=1, seed=0)
    assert set(profiles) == {"low", "high"}
    for profile in profiles.values():
        assert profile.windspeed.shape == (1, 288)
        assert np.isfinite(profile.windspeed).all()


def test_generate_for_regimes_offsets_seed_per_regime():
    profiles = generate_for_regimes(preset="1ch", regimes=("low", "high"), n_per_regime=1, seed=0)
    # different regimes get different (seed + i) offsets -- their outputs
    # shouldn't be byte-identical from reusing the exact same seed.
    assert not np.array_equal(profiles["low"].windspeed, profiles["high"].windspeed)
