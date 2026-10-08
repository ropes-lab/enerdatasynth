"""Tests for enerdatasynth.resources.pjm_load.api — the standalone entry point
added for parity with resources/wind/api.py."""

from __future__ import annotations

import pytest

from enerdatasynth.resources.pjm_load.api import (
    LoadProfile,
    describe_model,
    generate_load_profile,
    list_models,
)
from enerdatasynth.resources.pjm_load.exceptions import CheckpointNotFoundError, InvalidConditionError


def test_list_models_returns_real_bundled_checkpoints():
    models = list_models()
    assert "checkpoints" in models
    assert "default_run_id" in models
    assert models["default_run_id"] in models["checkpoints"]
    # the shipped default is the calibrated Transformer, and it is the only
    # Transformer run; the seven legacy runs predate the architecture field
    # and must keep being reported honestly as "mlp"
    default = models["checkpoints"][models["default_run_id"]]
    assert default["architecture"] == "transformer"
    assert default["calibrated"] is True
    assert models["transformer_runs_available"] == [models["default_run_id"]]
    legacy = [e for rid, e in models["checkpoints"].items() if rid != models["default_run_id"]]
    assert len(legacy) == 7
    assert all(e["architecture"] == "mlp" for e in legacy)


def test_manifest_default_matches_what_the_sdk_actually_selects():
    # the manifest is hand-regenerated from result.json files; this keeps it
    # from silently drifting away from the real checkpoint-selection logic
    from enerdatasynth.generators.load_conditioning import _discover_best_checkpoint
    from enerdatasynth.utils.config import CHECKPOINT_DIR

    assert list_models()["default_run_id"] == _discover_best_checkpoint(CHECKPOINT_DIR)


def test_describe_model_known_run_id():
    models = list_models()
    run_id = models["default_run_id"]
    entry = describe_model(run_id)
    assert entry["best_val_loss"] == models["checkpoints"][run_id]["best_val_loss"]


def test_describe_model_unknown_run_id_raises():
    with pytest.raises(CheckpointNotFoundError):
        describe_model("nonexistent_run_id")


def test_generate_load_profile_shape_and_finiteness(tiny_trained_checkpoint):
    import numpy as np

    profile = generate_load_profile(
        month=7, Tmax=86, Tmin=70, weekday=True,
        n_realizations=3, run_id=tiny_trained_checkpoint["run_id"],
        seed=0, units="normalized",
    )
    assert isinstance(profile, LoadProfile)
    assert profile.load.shape == (3, 24)
    assert np.isfinite(profile.load).all()
    assert profile.metadata["run_id"] == tiny_trained_checkpoint["run_id"]


def test_generate_load_profile_seed_reproducible(tiny_trained_checkpoint):
    import numpy as np

    kwargs = dict(
        month=7, Tmax=86, Tmin=70, weekday=True,
        n_realizations=2, run_id=tiny_trained_checkpoint["run_id"],
        seed=0, units="normalized",
    )
    a = generate_load_profile(**kwargs)
    b = generate_load_profile(**kwargs)
    np.testing.assert_array_equal(a.load, b.load)


def test_generate_load_profile_mw_units_rescales(tiny_trained_checkpoint):
    normalized = generate_load_profile(
        month=7, Tmax=86, Tmin=70, n_realizations=2,
        run_id=tiny_trained_checkpoint["run_id"], seed=0, units="normalized",
    )
    mw = generate_load_profile(
        month=7, Tmax=86, Tmin=70, n_realizations=2,
        run_id=tiny_trained_checkpoint["run_id"], seed=0, units="MW",
    )
    # MW values should be on a real-load scale (thousands), not [0,1]-ish
    assert mw.load.mean() > normalized.load.mean() * 10


def test_generate_load_profile_invalid_units_raises(tiny_trained_checkpoint):
    with pytest.raises(InvalidConditionError):
        generate_load_profile(
            month=7, Tmax=86, Tmin=70,
            run_id=tiny_trained_checkpoint["run_id"], units="not-a-real-unit",
        )


def test_load_profile_to_dataframe_and_summary_methods(tiny_trained_checkpoint):
    profile = generate_load_profile(
        month=7, Tmax=86, Tmin=70, n_realizations=3,
        run_id=tiny_trained_checkpoint["run_id"], seed=0, units="normalized",
    )
    df = profile.to_dataframe()
    assert len(df) == 3 * 24
    assert profile.mean().shape == (24,)
    lo, hi = profile.percentile_envelope()
    assert lo.shape == (24,) and hi.shape == (24,)
