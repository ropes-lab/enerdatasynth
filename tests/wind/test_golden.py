"""Golden characterization test for the extracted wind_diffusion package.

Not a byte-for-byte comparison against the original monolithic scripts
(that's impractical without the real multi-GB dataset / a trained
checkpoint) — instead this proves the extracted data-loading +
channel-derivation pipeline is internally correct, importable, and
produces the expected tensor shapes on a small synthetic fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wind_diffusion.config import get_config
from wind_diffusion.data import prepare_datasets
from wind_diffusion.transforms import destandardize_windows, standardize_windows
from wind_diffusion.utils import set_seed

SEQ_LEN = 288  # 24h @ 5min


def make_synthetic_csv(tmp_path, n_days: int = 6, seed: int = 0) -> str:
    """Build a tiny synthetic 5-min dataset with the raw columns the
    pipeline expects: timestamp, windspeed_100m, winddirection_100m,
    temperature_100m. Complete calendar days only, so every day survives
    `drop_incomplete_days`.
    """
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2020-01-01T00:00:00Z")
    n_rows = n_days * SEQ_LEN
    timestamps = pd.date_range(start, periods=n_rows, freq="5min")

    # Smooth-ish synthetic signals so nothing pathological (e.g. std=0) sneaks in.
    t = np.arange(n_rows)
    windspeed = 5.0 + 2.0 * np.sin(2 * np.pi * t / SEQ_LEN) + rng.normal(0, 0.3, n_rows)
    windspeed = np.clip(windspeed, 0.1, None)
    winddirection = (180.0 + 90.0 * np.sin(2 * np.pi * t / (SEQ_LEN * 2)) + rng.normal(0, 5, n_rows)) % 360.0
    temperature = 15.0 + 5.0 * np.sin(2 * np.pi * t / SEQ_LEN) + rng.normal(0, 0.5, n_rows)

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "windspeed_100m": windspeed,
            "winddirection_100m": winddirection,
            "temperature_100m": temperature,
        }
    )
    path = tmp_path / "synthetic_wind.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_2ch_pipeline_produces_expected_window_shape(tmp_path):
    set_seed(42)
    csv_path = make_synthetic_csv(tmp_path, n_days=6)

    config = get_config(
        "2ch",
        data_path=csv_path,
        split_frac=0.7,
        drop_incomplete_days=True,
    )

    prepared = prepare_datasets(config)

    # 6 days total, split_frac=0.7 -> 4 train / 2 val
    assert prepared.train_dataset.windows.shape == (4, 2, SEQ_LEN)
    assert prepared.val_dataset.windows.shape == (2, 2, SEQ_LEN)

    n_windows = prepared.train_dataset.windows.shape[0] + prepared.val_dataset.windows.shape[0]
    assert n_windows == 6

    # cond vector dims line up between norm_stats and the actual dataset tensors
    _, z = prepared.train_dataset[0]
    assert z.shape[0] == prepared.norm_stats.cond_dim

    assert prepared.norm_stats.model_channels == ["U", "V"]


def test_1ch_pipeline_produces_expected_window_shape(tmp_path):
    set_seed(0)
    csv_path = make_synthetic_csv(tmp_path, n_days=4)
    config = get_config("1ch", data_path=csv_path, split_frac=0.5)

    prepared = prepare_datasets(config)

    assert prepared.train_dataset.windows.shape == (2, 1, SEQ_LEN)
    assert prepared.val_dataset.windows.shape == (2, 1, SEQ_LEN)


def test_3ch_pipeline_produces_expected_window_shape(tmp_path):
    set_seed(7)
    csv_path = make_synthetic_csv(tmp_path, n_days=4)
    config = get_config("3ch", data_path=csv_path, split_frac=0.5)

    prepared = prepare_datasets(config)

    assert prepared.train_dataset.windows.shape == (2, 3, SEQ_LEN)
    assert prepared.val_dataset.windows.shape == (2, 3, SEQ_LEN)


def test_standardize_destandardize_roundtrip():
    rng = np.random.default_rng(123)
    windows = rng.normal(loc=5.0, scale=2.0, size=(10, SEQ_LEN, 2)).astype(np.float32)

    mean = windows.reshape(-1, 2).mean(axis=0)
    std = windows.reshape(-1, 2).std(axis=0)

    normed = standardize_windows(windows, mean, std)
    reconstructed = destandardize_windows(normed, mean, std)

    np.testing.assert_allclose(reconstructed, windows, rtol=1e-5, atol=1e-5)

    # Sanity: normalized data should be roughly zero-mean, unit-std.
    assert abs(normed.mean()) < 0.1
    assert abs(normed.std() - 1.0) < 0.1


def test_incomplete_days_are_dropped(tmp_path):
    """A CSV with one short trailing day should not produce a window for it."""
    set_seed(1)
    csv_path = make_synthetic_csv(tmp_path, n_days=3)

    # Truncate the file to lop off the last day partway through.
    df = pd.read_csv(csv_path)
    df = df.iloc[: 2 * SEQ_LEN + 50]  # 2 complete days + a partial 3rd day
    df.to_csv(csv_path, index=False)

    config = get_config("1ch", data_path=csv_path, split_frac=0.5, drop_incomplete_days=True)
    prepared = prepare_datasets(config)

    total_windows = prepared.train_dataset.windows.shape[0] + prepared.val_dataset.windows.shape[0]
    assert total_windows == 2


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        get_config("4ch")


# ---------------------------------------------------------------------------
# Wind persistence conditioning (yesterday's actual daily-mean windspeed)
# ---------------------------------------------------------------------------


def test_wind_persistence_matches_yesterdays_real_wind_mean(tmp_path):
    """For a dataset with no gaps, every day (except the first) should get
    exactly its chronological predecessor's real wind_mean as
    prev_day_wind_mean -- verified against an independent recomputation from
    train_meta/val_meta directly, not just re-deriving the same lookup
    logic."""
    set_seed(0)
    csv_path = make_synthetic_csv(tmp_path, n_days=10)
    config = get_config(
        "1ch", data_path=csv_path, split_frac=0.7,
        condition_on={"month": True, "day_of_week": True, "wind_regime_bucket": True, "temp_min_max_spread": True, "wind_persistence": True},
    )
    prepared = prepare_datasets(config)

    all_cond_df = pd.concat([prepared.train_cond_df, prepared.val_cond_df], ignore_index=True)
    all_meta = prepared.train_meta + prepared.val_meta
    wind_mean_by_date = {m["date"]: float(np.mean(m["windspeed_100m"])) for m in all_meta}

    for _, row in all_cond_df.iterrows():
        expected = wind_mean_by_date.get(row["date"] - pd.Timedelta(days=1))
        if expected is None:
            # No valid predecessor (first day) -> imputed to the train mean.
            assert row["prev_day_wind_mean"] == pytest.approx(prepared.norm_stats.wind_persistence_mean, rel=1e-4)
        else:
            # train_cond_df/val_cond_df hold the raw (pre-standardization)
            # value -- build_condition_matrix standardizes it separately.
            assert row["prev_day_wind_mean"] == pytest.approx(expected, rel=1e-4)


def test_wind_persistence_increases_cond_dim_by_one(tmp_path):
    set_seed(0)
    csv_path = make_synthetic_csv(tmp_path, n_days=8)
    base_condition_on = {"month": True, "day_of_week": True, "wind_regime_bucket": True, "temp_min_max_spread": True}
    config_without = get_config("1ch", data_path=csv_path, split_frac=0.6, condition_on=base_condition_on)
    config_with = get_config(
        "1ch", data_path=csv_path, split_frac=0.6,
        condition_on={**base_condition_on, "wind_persistence": True},
    )
    prepared_without = prepare_datasets(config_without)
    prepared_with = prepare_datasets(config_with)
    assert prepared_with.norm_stats.cond_dim == prepared_without.norm_stats.cond_dim + 1


def test_wind_persistence_gap_day_imputed_not_misaligned(tmp_path):
    """Dropping a day from the middle of the dataset (simulating a data gap)
    must not silently pair the day after the gap with a non-adjacent
    'yesterday' -- it should be treated as having no valid predecessor
    (imputed), not incorrectly linked to whatever day happens to sit at the
    previous array position."""
    set_seed(0)
    csv_path = make_synthetic_csv(tmp_path, n_days=10)
    df = pd.read_csv(csv_path)
    # Drop day index 5 (rows [5*SEQ_LEN, 6*SEQ_LEN)) entirely, creating a gap.
    gap_start, gap_end = 5 * SEQ_LEN, 6 * SEQ_LEN
    df = pd.concat([df.iloc[:gap_start], df.iloc[gap_end:]], ignore_index=True)
    df.to_csv(csv_path, index=False)

    config = get_config(
        "1ch", data_path=csv_path, split_frac=0.9,
        condition_on={"month": True, "day_of_week": True, "wind_regime_bucket": True, "temp_min_max_spread": True, "wind_persistence": True},
    )
    prepared = prepare_datasets(config)

    all_cond_df = pd.concat([prepared.train_cond_df, prepared.val_cond_df], ignore_index=True).sort_values("date").reset_index(drop=True)
    all_meta = prepared.train_meta + prepared.val_meta
    wind_mean_by_date = {m["date"]: float(np.mean(m["windspeed_100m"])) for m in all_meta}

    # The day immediately after the gap has no real predecessor in the data
    # -> must be imputed to the train mean, not linked to day index 4's
    # value via a stale positional shift.
    after_gap_date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=6)
    after_gap_rows = all_cond_df[all_cond_df["date"] == after_gap_date]
    assert len(after_gap_rows) == 1
    day_before_gap_value = wind_mean_by_date[after_gap_date - pd.Timedelta(days=2)]
    actual = after_gap_rows.iloc[0]["prev_day_wind_mean"]
    assert actual != pytest.approx(day_before_gap_value, rel=1e-4)
    assert actual == pytest.approx(prepared.norm_stats.wind_persistence_mean, rel=1e-4)


def test_persistence_by_regime_only_contains_real_train_observations(tmp_path):
    """persistence_by_regime must be built from real (non-imputed) previous-
    day values on the TRAIN split only -- it feeds api.generate_wind_profile's
    'sample a realistic default' path, so it must never contain the
    fabricated imputed placeholder."""
    set_seed(0)
    csv_path = make_synthetic_csv(tmp_path, n_days=12)
    config = get_config(
        "1ch", data_path=csv_path, split_frac=0.75,
        condition_on={"month": True, "day_of_week": True, "wind_regime_bucket": True, "temp_min_max_spread": True, "wind_persistence": True},
    )
    prepared = prepare_datasets(config)

    n_by_regime = sum(len(v) for v in prepared.norm_stats.persistence_by_regime.values())
    # The dataset's very first day always lacks a real predecessor and is
    # excluded via dropna() -- so real observations are strictly fewer than
    # total train days, never equal.
    assert n_by_regime < len(prepared.train_cond_df)
    assert set(prepared.norm_stats.persistence_by_regime.keys()) == {"low", "mid", "high"}


def test_diurnal_profile_by_regime_matches_independent_recomputation(tmp_path):
    """diurnal_profile_by_regime must equal the mean per-timestep windspeed
    of TRAIN days in that regime -- verified against an independent
    recomputation directly from train_meta, not just re-deriving the same
    averaging logic."""
    set_seed(0)
    csv_path = make_synthetic_csv(tmp_path, n_days=20)
    config = get_config("1ch", data_path=csv_path, split_frac=0.8)
    prepared = prepare_datasets(config)

    for regime in ["low", "mid", "high"]:
        regime_days = [
            m["windspeed_100m"] for m, bucket in zip(prepared.train_meta, prepared.train_cond_df["wind_bucket"])
            if bucket == regime
        ]
        if not regime_days:
            continue
        expected_profile = np.mean(np.stack(regime_days), axis=0)
        np.testing.assert_allclose(
            prepared.norm_stats.diurnal_profile_by_regime[regime], expected_profile, rtol=1e-4,
        )


def test_diurnal_profile_by_regime_has_seq_len_entries():
    set_seed(0)
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        csv_path = make_synthetic_csv(pathlib.Path(d), n_days=10)
        config = get_config("1ch", data_path=csv_path, split_frac=0.7)
        prepared = prepare_datasets(config)
        for regime in ["low", "mid", "high"]:
            profile = prepared.norm_stats.diurnal_profile_by_regime.get(regime, [])
            if profile:
                assert len(profile) == SEQ_LEN
