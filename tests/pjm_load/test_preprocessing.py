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
    build_conditioning_vector, sum_load_areas, LOAD_COLS,
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


# ---- zones made of several load areas (AE, DPL, PL) ---------------------------------

def _area_rows(area: str, start: str, hours: int, mw: float) -> pd.DataFrame:
    dt = pd.date_range(start, periods=hours, freq="h")
    return pd.DataFrame({"dt": dt, "load_area": area, "mw": mw})


def test_single_area_zone_passes_through_unchanged():
    df = _area_rows("PS", "2023-01-01", 48, 5000.0)
    out = sum_load_areas(df)
    assert len(out) == 48 and (out["mw"] == 5000.0).all()


def test_load_areas_are_summed_not_averaged():
    df = pd.concat([_area_rows("DPLCO", "2023-01-01", 48, 2000.0),
                    _area_rows("EASTON", "2023-01-01", 48, 30.0)])
    out = sum_load_areas(df)
    assert len(out) == 48
    assert (out["mw"] == 2030.0).all()


def test_hours_missing_an_area_that_year_are_dropped():
    # AE-style: a second area starts partway through the year; earlier hours would understate the zone.
    df = pd.concat([_area_rows("AECO", "2017-05-31", 48, 1000.0),
                    _area_rows("VMEU", "2017-06-01", 24, 80.0)])
    out = sum_load_areas(df)
    assert len(out) == 24
    assert out["dt"].min() == pd.Timestamp("2017-06-01") and (out["mw"] == 1080.0).all()


def test_area_sum_keeps_the_25_hour_fall_back_day():
    base = pd.date_range("2023-11-05", periods=24, freq="h")
    dt = base.append(pd.DatetimeIndex([base[1]]))      # the repeated 01:00
    rows = []
    for area, mw in (("PLCO", 4500.0), ("UGI", 120.0)):
        rows.append(pd.DataFrame({"dt": dt, "load_area": area, "mw": mw}))
    out = sum_load_areas(pd.concat(rows))
    out["date"] = out["dt"].dt.normalize()
    assert len(out) == 25 and (out["mw"] == 4620.0).all()
    assert detect_dst_days(out)["n_dst_days_detected"] == 1
