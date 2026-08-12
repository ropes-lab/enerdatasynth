"""
Tests for enerdatasynth.sampling.generate: generate_for_run against a
real (tiny) trained checkpoint. Reads real val-set X/Z from
data/processed/ (must exist — `enerdatasynth preprocess` already run
in this repo) since generate_for_run's whole job is to condition on
the real validation set; the tiny checkpoint's dims (x_dim=24,
cond_dim=23) match the real processed data's dims exactly, so this
runs against genuine data, not a mock.
"""

from __future__ import annotations
import numpy as np
import pytest

from enerdatasynth.utils.config import PROCESSED_DIR, GENERATED_DIR


pytestmark = pytest.mark.skipif(
    not (PROCESSED_DIR / "X_daily_profiles_val.npy").exists(),
    reason="requires data/processed/ (run `enerdatasynth preprocess` first)",
)


def test_generate_for_run_matches_real_validation_set_size(tiny_trained_checkpoint):
    from enerdatasynth.sampling.generate import generate_for_run

    result = tiny_trained_checkpoint
    out_path = generate_for_run(
        run_id=result["run_id"], diffusion_steps=result["diffusion_steps"],
        hidden_dim=result["hidden_dim"], n_layers=result["n_layers"],
        architecture=result["architecture"], n_heads=result["n_heads"], seed=0,
    )
    try:
        X_gen = np.load(out_path)
        X_val = np.load(PROCESSED_DIR / "X_daily_profiles_val.npy")
        assert X_gen.shape == X_val.shape
        assert np.isfinite(X_gen).all()
    finally:
        out_path.unlink(missing_ok=True)


def test_generate_for_run_seed_reproducible(tiny_trained_checkpoint):
    from enerdatasynth.sampling.generate import generate_for_run

    result = tiny_trained_checkpoint
    kwargs = dict(
        run_id=result["run_id"], diffusion_steps=result["diffusion_steps"],
        hidden_dim=result["hidden_dim"], n_layers=result["n_layers"],
        architecture=result["architecture"], n_heads=result["n_heads"],
    )
    out1 = generate_for_run(seed=0, **kwargs)
    a = np.load(out1)
    out2 = generate_for_run(seed=0, **kwargs)
    b = np.load(out2)
    try:
        assert np.array_equal(a, b)
    finally:
        out1.unlink(missing_ok=True)
