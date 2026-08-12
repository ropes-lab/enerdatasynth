"""Round-trip tests for wind speed/direction <-> u/v transforms and for
WindTransformPipeline.fit/transform/inverse_transform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_diffusion.transforms import (
    DAYS_OF_WEEK,
    MONTHS,
    ORDINAL_WIND_BUCKET_VALUES,
    WIND_BUCKET_LABELS,
    WindTransformPipeline,
    build_condition_matrix,
    speed_dir_to_uv,
    uv_to_speed_dir,
    wind_regime_slice,
)


def test_wind_regime_slice_all_features_enabled():
    condition_on = {
        "month": True,
        "day_of_week": True,
        "wind_regime_bucket": True,
        "temp_min_max_spread": True,
    }
    s = wind_regime_slice(condition_on)
    assert s == slice(len(MONTHS) + len(DAYS_OF_WEEK), len(MONTHS) + len(DAYS_OF_WEEK) + len(WIND_BUCKET_LABELS))
    assert s.stop - s.start == len(WIND_BUCKET_LABELS)


def test_wind_regime_slice_matches_build_condition_vector_layout():
    """The slice must land exactly on the one-hot block build_condition_vector
    actually produces for wind_regime -- verified end-to-end rather than just
    re-deriving the same offset arithmetic."""
    from wind_diffusion.data import NormStats
    from wind_diffusion.evaluate import build_condition_vector

    norm_stats = NormStats(
        target_mean=np.zeros(1),
        target_std=np.ones(1),
        numeric_cond_mean=np.zeros(3),
        numeric_cond_std=np.ones(3),
        wind_bucket_edges=(6.0, 12.0),
        model_channels=["u"],
        months=MONTHS,
        days_of_week=DAYS_OF_WEEK,
        wind_buckets=WIND_BUCKET_LABELS,
    )
    condition_on = {
        "month": True,
        "day_of_week": True,
        "wind_regime_bucket": True,
        "temp_min_max_spread": True,
    }
    s = wind_regime_slice(condition_on)

    for regime in WIND_BUCKET_LABELS:
        z = build_condition_vector(norm_stats, condition_on, wind_regime=regime)
        expected_oh = np.zeros(len(WIND_BUCKET_LABELS), dtype=np.float32)
        expected_oh[WIND_BUCKET_LABELS.index(regime)] = 1.0
        assert np.array_equal(z[s], expected_oh)


def test_wind_regime_slice_offsets_when_earlier_features_disabled():
    condition_on = {"month": False, "day_of_week": True, "wind_regime_bucket": True, "temp_min_max_spread": False}
    s = wind_regime_slice(condition_on)
    assert s == slice(len(DAYS_OF_WEEK), len(DAYS_OF_WEEK) + len(WIND_BUCKET_LABELS))


def test_wind_regime_slice_raises_when_wind_regime_not_enabled():
    condition_on = {"month": True, "day_of_week": True, "wind_regime_bucket": False, "temp_min_max_spread": True}
    with pytest.raises(ValueError):
        wind_regime_slice(condition_on)


# ---------------------------------------------------------------------------
# Ordinal wind-regime encoding (alternative to one-hot; see
# ORDINAL_WIND_BUCKET_VALUES's docstring for the "mid" hypothesis this tests)
# ---------------------------------------------------------------------------


def test_ordinal_wind_bucket_values_never_collide_with_the_cfg_null_vector():
    """The whole point of shifting the scale to {1, 2, 3} instead of the more
    natural-looking {-1, 0, 1}: classifier-free guidance zeroes out the
    regime slice for its "unconditional" branch, so if any real category
    equaled 0.0, guidance would be silently inert for requests matching that
    category (cond_pred - uncond_pred == 0 by construction). This must never
    be true for any label."""
    assert 0.0 not in ORDINAL_WIND_BUCKET_VALUES.values()


def test_ordinal_wind_bucket_values_mid_is_the_true_midpoint():
    low, mid, high = (ORDINAL_WIND_BUCKET_VALUES[k] for k in ["low", "mid", "high"])
    assert mid == pytest.approx((low + high) / 2)


def test_wind_regime_slice_ordinal_is_one_dim():
    condition_on = {"month": True, "day_of_week": True, "wind_regime_bucket": True, "temp_min_max_spread": True}
    s = wind_regime_slice(condition_on, wind_regime_encoding="ordinal")
    assert s.stop - s.start == 1
    assert s.start == len(MONTHS) + len(DAYS_OF_WEEK)


def test_build_condition_matrix_ordinal_encodes_correct_scalar_per_row():
    cond_df = pd.DataFrame({
        "month": [1, 2, 3],
        "day_of_week": [0, 1, 2],
        "wind_bucket": ["low", "mid", "high"],
        "temp_max": [20.0, 20.0, 20.0],
        "temp_min": [10.0, 10.0, 10.0],
        "temp_spread": [10.0, 10.0, 10.0],
    })
    condition_on = {"month": False, "day_of_week": False, "wind_regime_bucket": True, "temp_min_max_spread": False}
    matrix = build_condition_matrix(
        cond_df, condition_on, numeric_mean=np.zeros(3), numeric_std=np.ones(3), wind_regime_encoding="ordinal",
    )
    assert matrix.shape == (3, 1)
    expected = np.array([[ORDINAL_WIND_BUCKET_VALUES["low"]], [ORDINAL_WIND_BUCKET_VALUES["mid"]], [ORDINAL_WIND_BUCKET_VALUES["high"]]])
    assert np.array_equal(matrix, expected)


def test_build_condition_matrix_rejects_unknown_encoding():
    cond_df = pd.DataFrame({"wind_bucket": ["low"]})
    condition_on = {"wind_regime_bucket": True}
    with pytest.raises(ValueError):
        build_condition_matrix(cond_df, condition_on, numeric_mean=np.zeros(3), numeric_std=np.ones(3), wind_regime_encoding="bogus")


def test_norm_stats_round_trips_wind_regime_encoding():
    from wind_diffusion.data import NormStats

    norm_stats = NormStats(
        target_mean=np.zeros(1),
        target_std=np.ones(1),
        numeric_cond_mean=np.zeros(3),
        numeric_cond_std=np.ones(3),
        wind_bucket_edges=(6.0, 12.0),
        model_channels=["u"],
        wind_regime_encoding="ordinal",
    )
    restored = NormStats.from_dict(norm_stats.to_dict())
    assert restored.wind_regime_encoding == "ordinal"


def test_norm_stats_wind_regime_encoding_defaults_to_onehot_for_old_checkpoints():
    """A norm_stats.json saved before this field existed has no
    wind_regime_encoding key at all -- must default to "onehot" (the only
    encoding that existed then), not error or silently pick "ordinal"."""
    from wind_diffusion.data import NormStats

    legacy_dict = {
        "target_mean": [0.0],
        "target_std": [1.0],
        "numeric_cond_mean": [0.0, 0.0, 0.0],
        "numeric_cond_std": [1.0, 1.0, 1.0],
        "wind_bucket_edges": [6.0, 12.0],
        "model_channels": ["u"],
    }
    restored = NormStats.from_dict(legacy_dict)
    assert restored.wind_regime_encoding == "onehot"


def test_build_condition_vector_ordinal_matches_norm_stats_encoding():
    from wind_diffusion.data import NormStats
    from wind_diffusion.evaluate import build_condition_vector

    norm_stats = NormStats(
        target_mean=np.zeros(1),
        target_std=np.ones(1),
        numeric_cond_mean=np.zeros(3),
        numeric_cond_std=np.ones(3),
        wind_bucket_edges=(6.0, 12.0),
        model_channels=["u"],
        wind_regime_encoding="ordinal",
    )
    condition_on = {"month": False, "day_of_week": False, "wind_regime_bucket": True, "temp_min_max_spread": False}
    for regime in WIND_BUCKET_LABELS:
        z = build_condition_vector(norm_stats, condition_on, wind_regime=regime)
        assert z.shape == (1,)
        assert z[0] == pytest.approx(ORDINAL_WIND_BUCKET_VALUES[regime])


# ---------------------------------------------------------------------------
# Wind persistence conditioning (yesterday's actual daily-mean windspeed)
# ---------------------------------------------------------------------------


def test_wind_persistence_slice_is_last_and_one_dim():
    from wind_diffusion.transforms import wind_persistence_slice

    condition_on = {
        "month": True,
        "day_of_week": True,
        "wind_regime_bucket": True,
        "temp_min_max_spread": True,
        "wind_persistence": True,
    }
    s = wind_persistence_slice(condition_on)
    expected_start = len(MONTHS) + len(DAYS_OF_WEEK) + len(WIND_BUCKET_LABELS) + 3
    assert s == slice(expected_start, expected_start + 1)


def test_wind_persistence_slice_does_not_disturb_wind_regime_slice():
    """Persistence must be appended last so adding it never shifts the
    already-shipped regime slice used by classifier-free guidance."""
    condition_on_without = {
        "month": True, "day_of_week": True, "wind_regime_bucket": True, "temp_min_max_spread": True,
    }
    condition_on_with = {**condition_on_without, "wind_persistence": True}
    assert wind_regime_slice(condition_on_without) == wind_regime_slice(condition_on_with)


def test_wind_persistence_slice_raises_when_not_enabled():
    from wind_diffusion.transforms import wind_persistence_slice

    condition_on = {"month": True, "wind_regime_bucket": True}
    with pytest.raises(ValueError):
        wind_persistence_slice(condition_on)


def test_build_condition_matrix_wind_persistence_standardizes_correctly():
    cond_df = pd.DataFrame({"prev_day_wind_mean": [5.0, 10.0, 15.0]})
    condition_on = {"wind_persistence": True}
    matrix = build_condition_matrix(
        cond_df, condition_on, numeric_mean=np.zeros(3), numeric_std=np.ones(3),
        wind_persistence_mean=10.0, wind_persistence_std=5.0,
    )
    np.testing.assert_allclose(matrix[:, 0], [-1.0, 0.0, 1.0])


def test_build_condition_matrix_wind_persistence_requires_stats():
    cond_df = pd.DataFrame({"prev_day_wind_mean": [5.0]})
    condition_on = {"wind_persistence": True}
    with pytest.raises(ValueError):
        build_condition_matrix(cond_df, condition_on, numeric_mean=np.zeros(3), numeric_std=np.ones(3))


def test_build_condition_vector_wind_persistence_defaults_to_neutral_zero():
    """No previous_day_windspeed given -> standardizes to exactly 0.0 (the
    same 'no info' convention used by training-time dropout)."""
    from wind_diffusion.data import NormStats
    from wind_diffusion.evaluate import build_condition_vector

    norm_stats = NormStats(
        target_mean=np.zeros(1), target_std=np.ones(1),
        numeric_cond_mean=np.zeros(3), numeric_cond_std=np.ones(3),
        wind_bucket_edges=(6.0, 12.0), model_channels=["u"],
        wind_persistence_mean=10.0, wind_persistence_std=5.0,
    )
    condition_on = {"wind_persistence": True}
    z = build_condition_vector(norm_stats, condition_on)
    assert z[0] == pytest.approx(0.0)


def test_build_condition_vector_wind_persistence_standardizes_given_value():
    from wind_diffusion.data import NormStats
    from wind_diffusion.evaluate import build_condition_vector

    norm_stats = NormStats(
        target_mean=np.zeros(1), target_std=np.ones(1),
        numeric_cond_mean=np.zeros(3), numeric_cond_std=np.ones(3),
        wind_bucket_edges=(6.0, 12.0), model_channels=["u"],
        wind_persistence_mean=10.0, wind_persistence_std=5.0,
    )
    condition_on = {"wind_persistence": True}
    z = build_condition_vector(norm_stats, condition_on, previous_day_windspeed=20.0)
    assert z[0] == pytest.approx(2.0)


def test_norm_stats_round_trips_wind_persistence_fields():
    from wind_diffusion.data import NormStats

    norm_stats = NormStats(
        target_mean=np.zeros(1), target_std=np.ones(1),
        numeric_cond_mean=np.zeros(3), numeric_cond_std=np.ones(3),
        wind_bucket_edges=(6.0, 12.0), model_channels=["u"],
        wind_persistence_mean=7.5, wind_persistence_std=2.5,
        persistence_by_regime={"low": [1.0, 2.0], "mid": [3.0], "high": [4.0, 5.0]},
    )
    restored = NormStats.from_dict(norm_stats.to_dict())
    assert restored.wind_persistence_mean == pytest.approx(7.5)
    assert restored.wind_persistence_std == pytest.approx(2.5)
    assert restored.persistence_by_regime == {"low": [1.0, 2.0], "mid": [3.0], "high": [4.0, 5.0]}


def test_norm_stats_wind_persistence_defaults_for_old_checkpoints():
    """A norm_stats.json saved before this feature existed has none of these
    keys -- must default to neutral values, not error."""
    from wind_diffusion.data import NormStats

    legacy_dict = {
        "target_mean": [0.0], "target_std": [1.0],
        "numeric_cond_mean": [0.0, 0.0, 0.0], "numeric_cond_std": [1.0, 1.0, 1.0],
        "wind_bucket_edges": [6.0, 12.0], "model_channels": ["u"],
    }
    restored = NormStats.from_dict(legacy_dict)
    assert restored.wind_persistence_mean == 0.0
    assert restored.wind_persistence_std == 1.0
    assert restored.persistence_by_regime == {}
    assert restored.diurnal_profile_by_regime == {}


def test_norm_stats_round_trips_diurnal_profile_by_regime():
    from wind_diffusion.data import NormStats

    norm_stats = NormStats(
        target_mean=np.zeros(1), target_std=np.ones(1),
        numeric_cond_mean=np.zeros(3), numeric_cond_std=np.ones(3),
        wind_bucket_edges=(6.0, 12.0), model_channels=["u"],
        diurnal_profile_by_regime={"low": [1.0, 2.0, 3.0], "mid": [4.0, 5.0, 6.0], "high": [7.0, 8.0, 9.0]},
    )
    restored = NormStats.from_dict(norm_stats.to_dict())
    assert restored.diurnal_profile_by_regime == {"low": [1.0, 2.0, 3.0], "mid": [4.0, 5.0, 6.0], "high": [7.0, 8.0, 9.0]}


def test_speed_dir_uv_roundtrip():
    speed = np.array([5.0, 10.0, 0.0, 3.0, 7.5])
    direction = np.array([0.0, 90.0, 180.0, 359.0, 45.0])

    u, v = speed_dir_to_uv(speed, direction)
    speed2, direction2 = uv_to_speed_dir(u, v)

    np.testing.assert_allclose(speed, speed2, atol=1e-8)
    # direction is undefined when speed == 0, so skip that index.
    nonzero = speed > 0
    np.testing.assert_allclose(direction[nonzero], direction2[nonzero], atol=1e-6)


def test_speed_dir_uv_roundtrip_random():
    rng = np.random.default_rng(0)
    speed = rng.uniform(0.1, 25.0, size=200)
    direction = rng.uniform(0.0, 360.0, size=200)

    u, v = speed_dir_to_uv(speed, direction)
    speed2, direction2 = uv_to_speed_dir(u, v)

    np.testing.assert_allclose(speed, speed2, atol=1e-6)
    np.testing.assert_allclose(direction, direction2, atol=1e-4)


def _make_wind_df(n_days: int = 30, freq: str = "h", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_rows = n_days * 24
    index = pd.date_range("2021-01-01", periods=n_rows, freq=freq)

    t = np.arange(n_rows)
    speed = 6.0 + 2.0 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.5, n_rows)
    speed = np.clip(speed, 0.1, None)
    direction = (200.0 + 30.0 * np.sin(2 * np.pi * t / (24 * 7)) + rng.normal(0, 3, n_rows)) % 360.0

    return pd.DataFrame(
        {"windspeed100m": speed, "winddirection100m": direction},
        index=index,
    )


def test_wind_transform_pipeline_fit_transform_inverse_roundtrip():
    df = _make_wind_df(n_days=20)
    train_df = df.iloc[: 15 * 24]
    test_df = df.iloc[15 * 24 :]

    pipe = WindTransformPipeline()
    pipe.fit(train_df, speed_col="windspeed100m", dir_col="winddirection100m")

    train_features = pipe.transform(train_df)
    test_features = pipe.transform(test_df)

    assert "ustream_deseas_diff1" in test_features.columns
    assert "vcross_deseas_diff1" in test_features.columns

    pred_du = test_features["ustream_deseas_diff1"].fillna(0).values
    pred_dv = test_features["vcross_deseas_diff1"].fillna(0).values

    init_u = test_features["ustream_deseas"].iloc[0]
    init_v = test_features["vcross_deseas"].iloc[0]

    reconstructed = pipe.inverse_transform(
        index=test_features.index,
        ustream_deseas_diff1=pred_du,
        vcross_deseas_diff1=pred_dv,
        init_ustream_deseas=init_u,
        init_vcross_deseas=init_v,
    )

    # Reconstructing from the pipeline's own (non-lossy) forward-transform
    # outputs should recover the original speed/direction closely.
    np.testing.assert_allclose(
        reconstructed["windspeed100m"].values,
        test_df["windspeed100m"].values,
        atol=1e-6,
    )

    # Direction comparison needs wraparound-aware diffing.
    dir_diff = (reconstructed["winddirection100m"].values - test_df["winddirection100m"].values + 180) % 360 - 180
    np.testing.assert_allclose(dir_diff, np.zeros_like(dir_diff), atol=1e-4)


def test_wind_transform_pipeline_requires_datetime_index():
    df = _make_wind_df(n_days=5).reset_index(drop=True)
    pipe = WindTransformPipeline()
    with pytest.raises(ValueError):
        pipe.fit(df, speed_col="windspeed100m", dir_col="winddirection100m")


def test_wind_transform_pipeline_requires_fit_before_transform():
    df = _make_wind_df(n_days=5)
    pipe = WindTransformPipeline()
    with pytest.raises(RuntimeError):
        pipe.transform(df, speed_col="windspeed100m", dir_col="winddirection100m")
