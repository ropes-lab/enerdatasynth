"""
tests/test_wind_integration.py
=================================
Integration tests for the "wind" resource (`generators/wind.py`'s
`WindDiffusionModel`, `generators/wind_conditioning.py`'s
`condition_builder`/`construct` wiring, and `sdk.py`'s `_REGISTRY["wind"]`
entry) -- the structural/API skeleton described in `generators/base.py`.

Builds a genuinely tiny real checkpoint via the vendored `wind_diffusion`
package (same pattern `wind_diffusion`'s own `tests/wind/test_api.py`
uses for its end-to-end test) rather than mocking, so this exercises the
actual `enerdatasynth.resources.wind.registry.resolve_weights` -> `get_cached_model` ->
`generate_multiple_for_condition` call chain `WindDiffusionModel` wraps.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from enerdatasynth.resources.wind.config import get_config
from enerdatasynth.resources.wind.data.data import NormStats
from enerdatasynth.resources.wind.exceptions import CheckpointNotFoundError
from enerdatasynth.resources.wind.training.train import build_model

SEQ_LEN = 16
HIDDEN_DIM = 8
CONDITION_ON = {
    "month": True, "day_of_week": True,
    "temp_min_max_spread": False, "wind_regime_bucket": True,
}
COND_DIM = 12 + 7 + 3  # month one-hot + day_of_week one-hot + wind_regime one-hot


@pytest.fixture
def tiny_wind_checkpoint_dir(tmp_path, monkeypatch):
    """Builds a real (tiny) wind_diffusion "2ch" checkpoint + norm_stats.json
    under tmp_path/2ch/, and points WIND_DIFFUSION_CHECKPOINTS at tmp_path
    so registry.resolve_weights finds it via its env-dir precedence tier."""
    preset_dir = tmp_path / "2ch"
    preset_dir.mkdir()

    config = get_config(
        "2ch", hidden_dim=HIDDEN_DIM, num_conv_blocks=1, num_transformer_layers=1,
        num_heads=2, kernel_size=3, time_emb_dim=8, cond_emb_dim=8,
        seq_len=SEQ_LEN, diffusion_timesteps=10, condition_on=CONDITION_ON,
    )
    model = build_model(config, config.n_model_channels, cond_dim=COND_DIM, seq_len=config.seq_len)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config.to_dict(), "epoch": 1, "val_loss": 0.5},
        preset_dir / "model_checkpoint.pt",
    )

    n_channels = config.n_model_channels
    norm_stats = NormStats(
        target_mean=np.zeros(n_channels, dtype=np.float32),
        target_std=np.ones(n_channels, dtype=np.float32),
        numeric_cond_mean=np.zeros(3, dtype=np.float32),
        numeric_cond_std=np.ones(3, dtype=np.float32),
        wind_bucket_edges=(3.0, 7.0),
        model_channels=config.model_channels,
        cond_dim=COND_DIM,
    )
    with open(preset_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats.to_dict(), f)

    monkeypatch.setenv("WIND_DIFFUSION_CHECKPOINTS", str(tmp_path))
    monkeypatch.delenv("WIND_DIFFUSION_OFFLINE", raising=False)

    from enerdatasynth.resources.wind.evaluation import evaluate as wind_evaluate
    wind_evaluate._MODEL_CACHE.clear()

    return tmp_path


def test_wind_no_checkpoint_raises_wind_diffusions_own_exception(tmp_path, monkeypatch):
    """No wiring bug when no weights are available anywhere -- must be
    wind_diffusion's own, actionable CheckpointNotFoundError, not some
    generic AttributeError/KeyError from bad registry plumbing. All three
    presets now have real bundled checkpoints (data/checkpoints/wind/),
    so this forces the fallback path by pointing wind_conditioning's
    bundled-checkpoint root at an empty directory instead."""
    monkeypatch.delenv("WIND_DIFFUSION_CHECKPOINTS", raising=False)
    monkeypatch.setenv("WIND_DIFFUSION_OFFLINE", "1")

    from enerdatasynth.generators import wind_conditioning
    monkeypatch.setattr(wind_conditioning, "_WIND_CHECKPOINT_ROOT", tmp_path / "no_checkpoints_here")

    import enerdatasynth
    with pytest.raises(CheckpointNotFoundError):
        enerdatasynth.model("wind", zone="1ch")


def test_wind_bundled_checkpoints_work_without_any_env_setup(monkeypatch):
    """The whole point of bundling data/checkpoints/wind/<preset>/: a
    fresh clone with no WIND_DIFFUSION_CHECKPOINTS/manifest download setup
    must still be able to generate wind profiles for every preset out of
    the box, each with its own correct recommended generation settings
    (see WindDiffusionModel.__init__'s _recommended_generation)."""
    monkeypatch.delenv("WIND_DIFFUSION_CHECKPOINTS", raising=False)
    monkeypatch.delenv("WIND_DIFFUSION_OFFLINE", raising=False)

    from enerdatasynth.resources.wind.evaluation import evaluate as wind_evaluate
    wind_evaluate._MODEL_CACHE.clear()

    import enerdatasynth
    expected_eta = {"1ch": 1.0, "2ch": 0.0, "3ch": 1.0}
    for preset in ("1ch", "2ch", "3ch"):
        wind = enerdatasynth.model("wind", zone=preset)
        assert wind._generator._recommended_generation["eta"] == expected_eta[preset]
        wind.set(wind_regime="mid", month=7)
        samples = wind.sample(n=2, seed=0)
        assert samples.shape == (2, 288, 2)
        assert np.isfinite(samples).all()


def test_wind_list_and_describe_resources(tiny_wind_checkpoint_dir):
    import enerdatasynth

    info = enerdatasynth.list_resources()
    assert "wind" in info
    assert info["wind"]["zones"] == ["1ch", "2ch", "3ch"]
    assert info["wind"]["domain"] == "wind"
    assert set(info["wind"]["condition_fields"]) == {
        "wind_regime", "month", "day_of_week", "temp_min", "temp_max", "previous_day_windspeed",
    }
    # wind's checkpoint_status hook delegates to wind_diffusion's own
    # list_models() rather than reusing load's CHECKPOINT_DIR-scanning logic.
    assert "per_preset" in info["wind"]

    assert enerdatasynth.describe_resource("wind") == info["wind"]


def test_wind_end_to_end_sample_shape_and_finiteness(tiny_wind_checkpoint_dir):
    import enerdatasynth

    wind = enerdatasynth.model("wind", zone="2ch")
    wind.set(wind_regime="mid", month=3, day_of_week=1)
    samples = wind.sample(n=4, seed=0)

    assert samples.shape == (4, SEQ_LEN, 2)  # [..., 0]=windspeed, [..., 1]=direction
    assert np.isfinite(samples).all()


def test_wind_repr_does_not_leak_internals(tiny_wind_checkpoint_dir):
    import enerdatasynth

    wind = enerdatasynth.model("wind", zone="2ch")
    text = repr(wind)
    assert "ResourceModel" in text
    assert "wind" in text
    assert "torch" not in text.lower()
    assert "tensor" not in text.lower()


def test_wind_invalid_units_raises(tiny_wind_checkpoint_dir):
    import enerdatasynth

    wind = enerdatasynth.model("wind", zone="2ch")
    with pytest.raises(ValueError, match="units"):
        wind.sample(n=1, units="normalized")


def test_wind_unknown_set_key_raises(tiny_wind_checkpoint_dir):
    import enerdatasynth

    wind = enerdatasynth.model("wind", zone="2ch")
    with pytest.raises(ValueError, match="Unknown condition"):
        wind.set(not_a_real_key=1)


def test_wind_invalid_wind_regime_raises(tiny_wind_checkpoint_dir):
    import enerdatasynth

    wind = enerdatasynth.model("wind", zone="2ch")
    wind.set(wind_regime="not_a_real_bucket", month=3)
    with pytest.raises(ValueError, match="wind_regime"):
        wind.sample(n=1)


def test_wind_sample_seed_reproducible(tiny_wind_checkpoint_dir):
    import enerdatasynth

    wind = enerdatasynth.model("wind", zone="2ch")
    wind.set(wind_regime="mid", month=3)
    a = wind.sample(n=2, seed=42)
    b = wind.sample(n=2, seed=42)
    np.testing.assert_array_equal(a, b)


def test_wind_reset_clears_conditions(tiny_wind_checkpoint_dir):
    import enerdatasynth

    wind = enerdatasynth.model("wind", zone="2ch")
    wind.set(wind_regime="mid", month=3)
    result = wind.reset()
    assert result is wind
    assert wind._conditions == {}


def test_wind_sample_with_metadata(tiny_wind_checkpoint_dir):
    import enerdatasynth

    wind = enerdatasynth.model("wind", zone="2ch")
    wind.set(wind_regime="mid", month=3)
    samples, meta = wind.sample(n=2, seed=0, with_metadata=True)
    assert samples.shape == (2, SEQ_LEN, 2)
    assert meta["name"] == "wind"
    assert meta["zone"] == "2ch"
    assert meta["units"] == "m/s"
    assert meta["conditions"] == {"wind_regime": "mid", "month": 3}
    assert wind.last_metadata() == meta
