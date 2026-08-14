"""Tests for enerdatasynth.resources.wind.api: the v2 public API (generate_wind_profile,
WindProfile, list_models, describe_model). Zero real network calls -- the
end-to-end happy path builds a tiny synthetic checkpoint + norm_stats.json
fixture entirely in tmp_path, following the same pattern as
tests/test_samplers.py's `_make_fake_checkpoint`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from enerdatasynth.resources.wind import api
from enerdatasynth.resources.wind.config import get_config
from enerdatasynth.resources.wind.data.data import NormStats
from enerdatasynth.resources.wind.exceptions import CheckpointNotFoundError, InvalidConditionError
from enerdatasynth.resources.wind.training.train import build_model

SEQ_LEN = 16
HIDDEN_DIM = 8
COND_DIM = 3  # only wind_regime_bucket enabled -> 3 one-hot dims


def _tiny_condition_on() -> dict:
    return {
        "month": False,
        "day_of_week": False,
        "temp_min_max_spread": False,
        "wind_regime_bucket": True,
    }


def _build_tiny_checkpoint_and_norm_stats(tmp_path, preset: str = "2ch"):
    """Build+save a tiny real checkpoint (matching load_checkpoint_and_model's
    expectations) and a matching norm_stats.json, both small enough to run
    fast on CPU in a test."""
    config = get_config(
        preset,
        hidden_dim=HIDDEN_DIM,
        num_conv_blocks=1,
        num_transformer_layers=1,
        num_heads=2,
        kernel_size=3,
        time_emb_dim=8,
        cond_emb_dim=8,
        seq_len=SEQ_LEN,
        diffusion_timesteps=10,
        condition_on=_tiny_condition_on(),
    )
    model = build_model(config, config.n_model_channels, cond_dim=COND_DIM, seq_len=config.seq_len)

    ckpt_path = tmp_path / "model_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "epoch": 3,
            "val_loss": 0.1234,
        },
        ckpt_path,
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
    norm_stats_path = tmp_path / "norm_stats.json"
    with open(norm_stats_path, "w") as f:
        json.dump(norm_stats.to_dict(), f)

    return str(ckpt_path), str(norm_stats_path), config


# ---------------------------------------------------------------------------
# list_models / describe_model passthroughs
# ---------------------------------------------------------------------------


def test_list_models_reflects_three_pending_presets():
    models = api.list_models()
    assert set(models.keys()) == {"1ch", "2ch", "3ch"}
    for preset, entry in models.items():
        assert entry["status"] == "pending"


def test_describe_model_passthrough():
    info = api.describe_model("3ch")
    assert info["status"] == "pending"
    assert "cached_locally" in info


# ---------------------------------------------------------------------------
# CheckpointNotFoundError (real, pending manifest)
# ---------------------------------------------------------------------------


def test_generate_wind_profile_no_checkpoint_raises_cleanly(monkeypatch):
    monkeypatch.delenv("WIND_DIFFUSION_CHECKPOINTS", raising=False)
    monkeypatch.delenv("WIND_DIFFUSION_OFFLINE", raising=False)

    with pytest.raises(CheckpointNotFoundError):
        api.generate_wind_profile("2ch")


# ---------------------------------------------------------------------------
# InvalidConditionError, raised before any generation work
# ---------------------------------------------------------------------------


def test_generate_wind_profile_invalid_wind_regime_raises_before_generation(tmp_path, monkeypatch):
    # A stub checkpoint file is enough: with an invalid wind_regime, the
    # function must fail validation (reading only norm_stats.json) before
    # ever calling get_cached_model/torch.load on the checkpoint.
    ckpt_path = tmp_path / "stub_checkpoint.pt"
    ckpt_path.write_bytes(b"not a real checkpoint")

    norm_stats = NormStats(
        target_mean=np.zeros(2, dtype=np.float32),
        target_std=np.ones(2, dtype=np.float32),
        numeric_cond_mean=np.zeros(3, dtype=np.float32),
        numeric_cond_std=np.ones(3, dtype=np.float32),
        wind_bucket_edges=(3.0, 7.0),
        model_channels=["U", "V"],
        cond_dim=COND_DIM,
    )
    norm_stats_path = tmp_path / "norm_stats.json"
    with open(norm_stats_path, "w") as f:
        json.dump(norm_stats.to_dict(), f)

    load_calls = {"count": 0}
    real_torch_load = torch.load

    def counting_torch_load(*args, **kwargs):
        load_calls["count"] += 1
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_torch_load)

    with pytest.raises(InvalidConditionError):
        api.generate_wind_profile(
            "2ch",
            wind_regime="not_a_real_bucket",
            checkpoint=str(ckpt_path),
            norm_stats=str(norm_stats_path),
        )

    assert load_calls["count"] == 0


# ---------------------------------------------------------------------------
# End-to-end happy path
# ---------------------------------------------------------------------------


def test_generate_wind_profile_end_to_end(tmp_path):
    ckpt_path, norm_stats_path, config = _build_tiny_checkpoint_and_norm_stats(tmp_path)

    import enerdatasynth.resources.wind.evaluation.evaluate as evaluate_module

    evaluate_module._MODEL_CACHE.clear()

    n_realizations = 3
    profile = api.generate_wind_profile(
        "2ch",
        wind_regime="mid",
        checkpoint=ckpt_path,
        norm_stats=norm_stats_path,
        n_realizations=n_realizations,
        steps=5,
        device="cpu",
        seed=123,
    )

    assert isinstance(profile, api.WindProfile)
    assert profile.windspeed.shape == (n_realizations, SEQ_LEN)
    assert profile.direction.shape == (n_realizations, SEQ_LEN)

    assert isinstance(profile.time, pd.DatetimeIndex)
    assert len(profile.time) == SEQ_LEN
    assert pd.infer_freq(profile.time) in ("5min", "5T")

    assert profile.condition["wind_regime"] == "mid"

    mean = profile.mean()
    assert mean.shape == (SEQ_LEN,)

    lo, hi = profile.percentile_envelope()
    assert lo.shape == (SEQ_LEN,)
    assert hi.shape == (SEQ_LEN,)
    assert np.all(hi >= lo)

    df = profile.to_dataframe()
    assert len(df) == n_realizations * SEQ_LEN
    assert isinstance(df.index, pd.DatetimeIndex)
    assert set(["realization", "windspeed_100m", "winddirection_100m"]).issubset(df.columns)

    assert profile.metadata["preset"] == "2ch"
    assert profile.metadata["sampler"] == "ddim"
    assert profile.metadata["steps"] == 5
    assert profile.metadata["device"] == "cpu"
    assert profile.metadata["wall_time_seconds"] >= 0


def test_generate_wind_profile_anchors_date(tmp_path):
    ckpt_path, norm_stats_path, config = _build_tiny_checkpoint_and_norm_stats(tmp_path)

    import enerdatasynth.resources.wind.evaluation.evaluate as evaluate_module

    evaluate_module._MODEL_CACHE.clear()

    profile = api.generate_wind_profile(
        "2ch",
        wind_regime="low",
        checkpoint=ckpt_path,
        norm_stats=norm_stats_path,
        n_realizations=1,
        steps=3,
        device="cpu",
        date="2024-03-15",
    )

    assert profile.time[0] == pd.Timestamp("2024-03-15")
    assert profile.condition["wind_regime"] == "low"


def _build_tiny_checkpoint_and_norm_stats_with_persistence(tmp_path, persistence_by_regime):
    """Variant of _build_tiny_checkpoint_and_norm_stats with wind_persistence
    enabled, for testing generate_wind_profile's 'sample a realistic default
    previous-day value' UX (see api.generate_wind_profile)."""
    condition_on = {"wind_regime_bucket": True, "wind_persistence": True}
    config = get_config(
        "2ch",
        hidden_dim=HIDDEN_DIM,
        num_conv_blocks=1,
        num_transformer_layers=1,
        num_heads=2,
        kernel_size=3,
        time_emb_dim=8,
        cond_emb_dim=8,
        seq_len=SEQ_LEN,
        diffusion_timesteps=10,
        condition_on=condition_on,
    )
    cond_dim = 4  # 3-dim wind_regime one-hot + 1-dim persistence
    model = build_model(config, config.n_model_channels, cond_dim=cond_dim, seq_len=config.seq_len)

    ckpt_path = tmp_path / "model_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "epoch": 3,
            "val_loss": 0.1234,
        },
        ckpt_path,
    )

    n_channels = config.n_model_channels
    norm_stats = NormStats(
        target_mean=np.zeros(n_channels, dtype=np.float32),
        target_std=np.ones(n_channels, dtype=np.float32),
        numeric_cond_mean=np.zeros(3, dtype=np.float32),
        numeric_cond_std=np.ones(3, dtype=np.float32),
        wind_bucket_edges=(3.0, 7.0),
        model_channels=config.model_channels,
        cond_dim=cond_dim,
        wind_persistence_mean=10.0,
        wind_persistence_std=5.0,
        persistence_by_regime=persistence_by_regime,
    )
    norm_stats_path = tmp_path / "norm_stats.json"
    with open(norm_stats_path, "w") as f:
        json.dump(norm_stats.to_dict(), f)

    return str(ckpt_path), str(norm_stats_path), config


def test_generate_wind_profile_rejection_sampling_disabled_by_default(tmp_path):
    ckpt_path, norm_stats_path, config = _build_tiny_checkpoint_and_norm_stats(tmp_path)

    import enerdatasynth.resources.wind.evaluation.evaluate as evaluate_module

    evaluate_module._MODEL_CACHE.clear()

    profile = api.generate_wind_profile(
        "2ch", wind_regime="mid", checkpoint=ckpt_path, norm_stats=norm_stats_path,
        n_realizations=3, steps=3, device="cpu", seed=0,
    )
    assert profile.metadata["rejection_sampling"] is False
    assert profile.metadata["rejection_stats"] is None
    assert profile.windspeed.shape == (3, SEQ_LEN)


def test_generate_wind_profile_rejection_sampling_guarantees_regime_match(tmp_path):
    """The whole point of rejection sampling: every returned realization's
    OWN realized daily-mean windspeed must land in the requested regime's
    bucket, unlike the non-rejection path which only relies on the model's
    conditioning to steer it there."""
    from enerdatasynth.resources.wind.data.transforms import assign_wind_bucket

    ckpt_path, norm_stats_path, config = _build_tiny_checkpoint_and_norm_stats(tmp_path)

    import enerdatasynth.resources.wind.evaluation.evaluate as evaluate_module

    evaluate_module._MODEL_CACHE.clear()

    profile = api.generate_wind_profile(
        "2ch", wind_regime="mid", checkpoint=ckpt_path, norm_stats=norm_stats_path,
        n_realizations=5, steps=3, device="cpu", seed=0,
        rejection_sampling=True, rejection_oversample_factor=20.0, rejection_max_batches=20,
    )

    assert profile.metadata["rejection_sampling"] is True
    assert profile.metadata["rejection_stats"]["fully_satisfied"] is True
    assert profile.windspeed.shape == (5, SEQ_LEN)

    q1, q2 = profile.condition["wind_bucket_edges"]
    for day in profile.windspeed:
        assert assign_wind_bucket(float(day.mean()), q1, q2) == "mid"


def test_generate_wind_profile_rejection_sampling_reports_shortfall(tmp_path):
    """If rejection_max_batches is exhausted before enough matches are
    found, the caller must be told via metadata rather than silently getting
    back an under-sized (or worse, padded/fabricated) batch."""
    ckpt_path, norm_stats_path, config = _build_tiny_checkpoint_and_norm_stats(tmp_path)

    import enerdatasynth.resources.wind.evaluation.evaluate as evaluate_module

    evaluate_module._MODEL_CACHE.clear()

    profile = api.generate_wind_profile(
        "2ch", wind_regime="mid", checkpoint=ckpt_path, norm_stats=norm_stats_path,
        n_realizations=5, steps=3, device="cpu", seed=0,
        rejection_sampling=True, rejection_max_batches=0,
    )

    assert profile.metadata["rejection_stats"]["fully_satisfied"] is False
    assert profile.metadata["rejection_stats"]["n_returned"] == 0
    assert profile.metadata["n_realizations_returned"] == 0
    assert profile.windspeed.shape == (0, SEQ_LEN)


def test_rejection_rank_by_diurnal_shape_default_off_keeps_generation_order(monkeypatch):
    """Off by default -- must not rank/reorder even when a reference profile
    is available, so rejection_sampling=True alone keeps its
    already-reported, unranked behavior."""
    from enerdatasynth.resources.wind import api as api_module

    seq_len = 4
    n_realizations = 2
    # Two batches of candidates: first batch has 3 matches (indices 0,1,2 in
    # generation order), already more than n_realizations=2 needed.
    batch_windspeed = np.array([
        [10.0, 10.0, 10.0, 10.0],  # far from reference
        [5.0, 5.0, 5.0, 5.0],      # exactly the reference
        [5.1, 5.1, 5.1, 5.1],      # close to reference
    ])
    batch_direction = np.zeros_like(batch_windspeed)

    def fake_generate_batch(model, schedule, norm_stats_obj, config, wind_regime, previous_day_windspeed, batch_n, *rest, **kwargs):
        return batch_windspeed[:batch_n], batch_direction[:batch_n], "disabled", None

    monkeypatch.setattr(api_module, "_generate_batch", fake_generate_batch)

    class FakeConfig:
        preset = "1ch"
        condition_on = {}

    norm_stats = NormStats(
        target_mean=np.zeros(1), target_std=np.ones(1),
        numeric_cond_mean=np.zeros(3), numeric_cond_std=np.ones(3),
        wind_bucket_edges=(3.0, 7.0), model_channels=["u"],
        diurnal_profile_by_regime={"mid": [5.0, 5.0, 5.0, 5.0]},
    )
    monkeypatch.setattr(api_module, "assign_wind_bucket", lambda *a, **k: "mid")

    windspeed, direction, mode, resolved, stats = api_module._generate_with_rejection_sampling(
        None, None, norm_stats, FakeConfig(), "mid", None, n_realizations,
        1, seq_len, None, None, "cpu",
        "ddim", 5, 0.0, 1.0, None, False, 99.0,
        oversample_factor=3.0, max_batches=1, rank_by_diurnal_shape=False,
    )

    assert stats["diurnal_shape_ranking_applied"] is False
    # Unranked: keeps the first n_realizations matches in generation order (rows 0, 1).
    np.testing.assert_array_equal(windspeed, batch_windspeed[:2])


def test_rejection_rank_by_diurnal_shape_keeps_closest_matches(monkeypatch):
    """Enabled: among more matches than needed, keep the ones closest to the
    real reference diurnal profile -- not just the first found."""
    from enerdatasynth.resources.wind import api as api_module

    seq_len = 4
    n_realizations = 2
    batch_windspeed = np.array([
        [10.0, 10.0, 10.0, 10.0],  # far from reference -- should be dropped
        [5.0, 5.0, 5.0, 5.0],      # exactly the reference -- kept
        [5.1, 5.1, 5.1, 5.1],      # close to reference -- kept
    ])
    batch_direction = np.zeros_like(batch_windspeed)

    def fake_generate_batch(model, schedule, norm_stats_obj, config, wind_regime, previous_day_windspeed, batch_n, *rest, **kwargs):
        return batch_windspeed[:batch_n], batch_direction[:batch_n], "disabled", None

    monkeypatch.setattr(api_module, "_generate_batch", fake_generate_batch)

    class FakeConfig:
        preset = "1ch"
        condition_on = {}

    norm_stats = NormStats(
        target_mean=np.zeros(1), target_std=np.ones(1),
        numeric_cond_mean=np.zeros(3), numeric_cond_std=np.ones(3),
        wind_bucket_edges=(3.0, 7.0), model_channels=["u"],
        diurnal_profile_by_regime={"mid": [5.0, 5.0, 5.0, 5.0]},
    )
    monkeypatch.setattr(api_module, "assign_wind_bucket", lambda *a, **k: "mid")

    windspeed, direction, mode, resolved, stats = api_module._generate_with_rejection_sampling(
        None, None, norm_stats, FakeConfig(), "mid", None, n_realizations,
        1, seq_len, None, None, "cpu",
        "ddim", 5, 0.0, 1.0, None, False, 99.0,
        oversample_factor=3.0, max_batches=1, rank_by_diurnal_shape=True,
    )

    assert stats["diurnal_shape_ranking_applied"] is True
    kept = {tuple(row) for row in windspeed}
    assert kept == {(5.0, 5.0, 5.0, 5.0), (5.1, 5.1, 5.1, 5.1)}
    assert (10.0, 10.0, 10.0, 10.0) not in kept


def test_generate_wind_profile_samples_persistence_default_when_not_given(tmp_path):
    """The whole point of the persistence feature's API design: the common
    call (no previous_day_windspeed argument at all) must still work.
    Sampling happens per-realization (not one shared value for the whole
    batch, see generate_multiple_for_condition's docstring), so metadata
    reports the mode rather than a single resolved scalar."""
    ckpt_path, norm_stats_path, config = _build_tiny_checkpoint_and_norm_stats_with_persistence(
        tmp_path, persistence_by_regime={"low": [1.0], "mid": [42.0], "high": [99.0]},
    )
    import enerdatasynth.resources.wind.evaluation.evaluate as evaluate_module

    evaluate_module._MODEL_CACHE.clear()

    profile = api.generate_wind_profile(
        "2ch", wind_regime="mid", checkpoint=ckpt_path, norm_stats=norm_stats_path,
        n_realizations=1, steps=3, device="cpu", seed=0,
    )
    assert profile.metadata["previous_day_windspeed_mode"] == "sampled_per_realization"
    assert profile.metadata["previous_day_windspeed"] is None


def test_generate_wind_profile_persistence_sampling_varies_per_realization(tmp_path, monkeypatch):
    """The whole point of sampling per-realization instead of once for the
    batch: different realizations should see different (standardized)
    previous-day values when the regime's empirical distribution has more
    than one distinct value."""
    ckpt_path, norm_stats_path, config = _build_tiny_checkpoint_and_norm_stats_with_persistence(
        tmp_path, persistence_by_regime={"low": [1.0], "mid": [10.0, 20.0, 30.0, 40.0], "high": [99.0]},
    )
    import enerdatasynth.resources.wind.evaluation.evaluate as evaluate_module

    evaluate_module._MODEL_CACHE.clear()

    seen_cond_matrix = {}
    original_sample = evaluate_module.sampler_sample

    def spy_sample(model, schedule, cond_vector, *args, **kwargs):
        seen_cond_matrix["z"] = np.asarray(cond_vector).copy()
        return original_sample(model, schedule, cond_vector, *args, **kwargs)

    monkeypatch.setattr(evaluate_module, "sampler_sample", spy_sample)

    api.generate_wind_profile(
        "2ch", wind_regime="mid", checkpoint=ckpt_path, norm_stats=norm_stats_path,
        n_realizations=20, steps=3, device="cpu", seed=0,
    )

    persistence_column = seen_cond_matrix["z"][:, -1]  # last dim: wind_persistence (see condition_on order)
    assert len(np.unique(persistence_column)) > 1


def test_generate_wind_profile_explicit_previous_day_windspeed_overrides_sampling(tmp_path):
    ckpt_path, norm_stats_path, config = _build_tiny_checkpoint_and_norm_stats_with_persistence(
        tmp_path, persistence_by_regime={"low": [1.0], "mid": [42.0], "high": [99.0]},
    )
    import enerdatasynth.resources.wind.evaluation.evaluate as evaluate_module

    evaluate_module._MODEL_CACHE.clear()

    profile = api.generate_wind_profile(
        "2ch", wind_regime="mid", checkpoint=ckpt_path, norm_stats=norm_stats_path,
        n_realizations=1, steps=3, device="cpu", seed=0, previous_day_windspeed=7.0,
    )
    assert profile.metadata["previous_day_windspeed"] == pytest.approx(7.0)
    assert profile.metadata["previous_day_windspeed_mode"] == "explicit"


def test_build_cond_matrix_threads_month_day_of_week_temp_through():
    """api._build_cond_matrix_for_batch must actually pass month/day_of_week/
    temp_min/temp_max through to build_condition_vector instead of always
    falling back to its hardcoded defaults (month=6, day_of_week=0,
    temp_min=10.0, temp_max=22.0) regardless of what a caller requests --
    previously true even when explicit values were given, since the only
    caller from the public generate_wind_profile path never forwarded them."""
    condition_on = {
        "month": True, "day_of_week": True,
        "temp_min_max_spread": True, "wind_regime_bucket": True,
    }
    norm_stats = NormStats(
        target_mean=np.zeros(1, dtype=np.float32), target_std=np.ones(1, dtype=np.float32),
        numeric_cond_mean=np.zeros(3, dtype=np.float32), numeric_cond_std=np.ones(3, dtype=np.float32),
        wind_bucket_edges=(3.0, 7.0), model_channels=["u"],
    )

    class FakeConfig:
        condition_on = {
            "month": True, "day_of_week": True,
            "temp_min_max_spread": True, "wind_regime_bucket": True,
        }

    default_matrix, _, _ = api._build_cond_matrix_for_batch(
        norm_stats, FakeConfig(), "mid", None, batch_n=1,
    )
    overridden_matrix, _, _ = api._build_cond_matrix_for_batch(
        norm_stats, FakeConfig(), "mid", None, batch_n=1,
        month=3, day_of_week=5, temp_min=-2.0, temp_max=4.0,
    )

    assert not np.allclose(default_matrix, overridden_matrix)
    # month one-hot occupies columns 0-11; default build_condition_vector
    # uses month=6 (index 5), the override requests month=3 (index 2).
    assert default_matrix[0, 5] == 1.0 and default_matrix[0, 2] == 0.0
    assert overridden_matrix[0, 2] == 1.0 and overridden_matrix[0, 5] == 0.0


def test_generate_wind_profile_default_wind_regime(tmp_path):
    """When wind_regime is None, a sensible default from NormStats.wind_buckets is used."""
    ckpt_path, norm_stats_path, config = _build_tiny_checkpoint_and_norm_stats(tmp_path)

    import enerdatasynth.resources.wind.evaluation.evaluate as evaluate_module

    evaluate_module._MODEL_CACHE.clear()

    profile = api.generate_wind_profile(
        "2ch",
        wind_regime=None,
        checkpoint=ckpt_path,
        norm_stats=norm_stats_path,
        n_realizations=1,
        steps=3,
        device="cpu",
    )

    assert profile.condition["wind_regime"] in ("low", "mid", "high")
