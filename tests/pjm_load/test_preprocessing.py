"""
Tests for enerdatasynth.resources.pjm_load.data.preprocess: pivoting, DST detection,
exclusion rules, and conditioning-vector construction. All synthetic
— no dependency on the real raw CSVs being present.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from enerdatasynth.resources.pjm_load.data.preprocess import (
    build_daily_pivot, apply_exclusions, detect_dst_days,
    build_conditioning_vector, LOAD_COLS,
)
from enerdatasynth.utils.config import COND_DIM, N_MONTHS, N_DOW


def _hourly_rows(date: str, n_hours: int) -> pd.DataFrame:
    """n_hours hourly rows for one date (23/24/25 to simulate DST)."""
    dt = pd.Timestamp(date)
    return pd.DataFrame({
        "date": [dt] * n_hours,
        "hour": list(range(n_hours)),
        "mw": np.linspace(1000, 1000 + n_hours * 10, n_hours),
    })


def test_build_daily_pivot_produces_24_hour_columns():
    zone_df = pd.concat([_hourly_rows("2023-01-01", 24), _hourly_rows("2023-01-02", 24)])
    pivot = build_daily_pivot(zone_df)
    assert len(pivot) == 2
    assert set(LOAD_COLS).issubset(pivot.columns)
    assert pivot["h00"].notna().all()


def test_detect_dst_days_flags_non_24_hour_days():
    zone_df = pd.concat([
        _hourly_rows("2023-03-12", 23),  # spring-forward
        _hourly_rows("2023-11-05", 25),  # fall-back
        _hourly_rows("2023-06-01", 24),  # normal
    ])
    result = detect_dst_days(zone_df)
    assert result["n_dst_days_detected"] == 2
    dst_dates = {pd.Timestamp(d).normalize() for d in result["dst_dates"]}
    assert pd.Timestamp("2023-03-12") in dst_dates
    assert pd.Timestamp("2023-11-05") in dst_dates
    assert pd.Timestamp("2023-06-01") not in dst_dates


def test_apply_exclusions_drops_dst_missing_load_missing_weather_and_covid():
    dates = pd.to_datetime([
        "2023-01-01",  # clean
        "2023-01-02",  # missing weather
        "2023-01-03",  # missing load (NaN hour)
        "2020-04-15",  # covid
        "2023-03-12",  # DST
    ])
    pivot = pd.DataFrame({"date": dates})
    for col in LOAD_COLS:
        pivot[col] = 1000.0
    pivot.loc[2, "h05"] = np.nan  # missing load on day 3

    daily_temp = pd.DataFrame({
        "date": dates,
        "Tmax": [70.0, np.nan, 70.0, 70.0, 70.0],
        "Tmin": [50.0, np.nan, 50.0, 50.0, 50.0],
    })

    clean, counts = apply_exclusions(pivot, daily_temp, dst_dates=[pd.Timestamp("2023-03-12")])

    assert counts["total_raw_days"] == 5
    assert counts["dst_removed"] == 1
    assert counts["missing_load_removed"] == 1
    assert counts["missing_weather_removed"] == 1
    assert counts["covid_removed"] == 1
    assert counts["final_clean_days"] == 1
    assert list(clean["date"]) == [pd.Timestamp("2023-01-01")]


def test_build_conditioning_vector_shape_and_onehot_placement():
    dates = pd.to_datetime(["2023-07-15", "2023-12-25"])  # July Saturday, December Monday
    df = pd.DataFrame({"date": dates, "month": dates.month, "Tmax": [90.0, 40.0], "Tmin": [70.0, 25.0]})
    for col in LOAD_COLS:
        df[col] = 1000.0

    Z, df_out, scaler_temp, scaler_spread = build_conditioning_vector(df, fit=True)

    assert Z.shape == (2, COND_DIM)
    # Row 0: July -> month index 6 (0-based) one-hot
    assert Z[0, 6] == 1.0
    assert Z[0, :N_MONTHS].sum() == 1.0
    # Row 1: December -> month index 11
    assert Z[1, 11] == 1.0
    # dow one-hot sums to exactly 1 per row
    dow_start = N_MONTHS
    assert np.allclose(Z[:, dow_start:dow_start + N_DOW].sum(axis=1), 1.0)


def test_build_conditioning_vector_requires_scalers_when_not_fitting():
    dates = pd.to_datetime(["2023-07-15"])
    df = pd.DataFrame({"date": dates, "month": dates.month, "Tmax": [90.0], "Tmin": [70.0]})
    for col in LOAD_COLS:
        df[col] = 1000.0
    with pytest.raises(ValueError, match="scaler_temp and scaler_spread"):
        build_conditioning_vector(df, fit=False)


def test_build_conditioning_vector_val_uses_train_fitted_scalers():
    """Validation-set conditioning must reuse the training-set scalers, not refit its own."""
    train_dates = pd.to_datetime(["2023-01-01", "2023-06-01", "2023-07-15"])
    train_df = pd.DataFrame({
        "date": train_dates, "month": train_dates.month,
        "Tmax": [30.0, 85.0, 95.0], "Tmin": [10.0, 65.0, 75.0],
    })
    for col in LOAD_COLS:
        train_df[col] = 1000.0

    _, _, scaler_temp, scaler_spread = build_conditioning_vector(train_df, fit=True)

    val_dates = pd.to_datetime(["2023-07-20"])
    val_df = pd.DataFrame({"date": val_dates, "month": val_dates.month, "Tmax": [95.0], "Tmin": [75.0]})
    for col in LOAD_COLS:
        val_df[col] = 1000.0

    Z_val, _, scaler_temp_out, scaler_spread_out = build_conditioning_vector(
        val_df, scaler_temp=scaler_temp, scaler_spread=scaler_spread, fit=False,
    )
    assert scaler_temp_out is scaler_temp
    assert scaler_spread_out is scaler_spread
    assert Z_val.shape == (1, COND_DIM)
