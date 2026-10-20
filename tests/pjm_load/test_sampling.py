"""
Tests for enerdatasynth.resources.pjm_load.sampling.generate: generate_for_run against a
real (tiny) trained checkpoint.

generate_for_run's whole job is to condition on a validation set, one sample per
condition vector. This repository ships no real validation data, so the tests build
a small synthetic one (x_dim=24, cond_dim=23, matching the checkpoint's dims) in a
temp dir. The checkpoint, diffusion process, and sampler are all real.
"""

from __future__ import annotations
import numpy as np
import pytest

import enerdatasynth.utils.config as config_module

N_VAL_DAYS = 40


@pytest.fixture
def synthetic_val_set(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    np.save(tmp_path / "X_daily_profiles_val.npy", rng.random((N_VAL_DAYS, 24)).astype(np.float32))
    Z = np.zeros((N_VAL_DAYS, 23), dtype=np.float32)
    Z[np.arange(N_VAL_DAYS), np.arange(N_VAL_DAYS) % 12] = 1.0
    np.save(tmp_path / "Z_conditions_val.npy", Z)
    monkeypatch.setattr(config_module, "PROCESSED_DIR", tmp_path)
    return tmp_path


def test_generate_for_run_matches_validation_set_size(tiny_trained_checkpoint, synthetic_val_set):
    from enerdatasynth.resources.pjm_load.sampling.generate import generate_for_run

    result = tiny_trained_checkpoint
    out_path = generate_for_run(
        run_id=result["run_id"], diffusion_steps=result["diffusion_steps"],
        hidden_dim=result["hidden_dim"], n_layers=result["n_layers"],
        architecture=result["architecture"], n_heads=result["n_heads"], seed=0,
    )
    try:
        X_gen = np.load(out_path)
        assert X_gen.shape == (N_VAL_DAYS, 24)
        assert np.isfinite(X_gen).all()
    finally:
        out_path.unlink(missing_ok=True)


def test_generate_for_run_seed_reproducible(tiny_trained_checkpoint, synthetic_val_set):
    from enerdatasynth.resources.pjm_load.sampling.generate import generate_for_run

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


def test_generate_for_run_without_data_explains_how_to_get_it(tiny_trained_checkpoint, tmp_path, monkeypatch):
    from enerdatasynth.resources.pjm_load.sampling.generate import generate_for_run

    monkeypatch.setattr(config_module, "PROCESSED_DIR", tmp_path)
    result = tiny_trained_checkpoint
    with pytest.raises(FileNotFoundError, match="enerdatasynth preprocess"):
        generate_for_run(
            run_id=result["run_id"], diffusion_steps=result["diffusion_steps"],
            hidden_dim=result["hidden_dim"], n_layers=result["n_layers"],
            architecture=result["architecture"], n_heads=result["n_heads"],
        )
