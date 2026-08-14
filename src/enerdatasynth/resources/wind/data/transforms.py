"""Wind data transforms: speed/direction <-> u/v, per-preset target-channel
derivation and decoding, standardization, wind-regime bucketing, and the
condition-vector builder.

Also hosts `WindTransformPipeline`, copied over verbatim (modulo minor
lint-only touch-ups) from the repo-root `wind_transform.py`. That module is
orphaned/dead code (not imported by any of the three training scripts) but
represents real prior work on streamwise wind rotation + deseasonalization,
so it's preserved here rather than discarded. The original `wind_transform.py`
file is left untouched at the repo root.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from enerdatasynth.resources.wind.config import PresetName

# ---------------------------------------------------------------------------
# Speed/direction <-> u/v (meteorological convention: u eastward, v northward)
# ---------------------------------------------------------------------------


def speed_dir_to_uv(speed, direction_deg):
    """Convert meteorological wind speed/direction to u (eastward) / v (northward) components."""
    direction_rad = np.deg2rad(direction_deg)
    u = -speed * np.sin(direction_rad)
    v = -speed * np.cos(direction_rad)
    return u, v


def uv_to_speed_dir(u, v):
    """Convert u/v components back to speed and direction (degrees, 0-360)."""
    speed = np.sqrt(u**2 + v**2)
    direction_rad = np.arctan2(-u, -v)
    direction_deg = np.rad2deg(direction_rad) % 360.0
    return speed, direction_deg


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------


def standardize_windows(w, mean, std):
    return (w - mean) / std


def destandardize_windows(w, mean, std):
    return w * std + mean


# ---------------------------------------------------------------------------
# Per-preset target-channel derivation / decoding
# ---------------------------------------------------------------------------


def derive_channels(df: pd.DataFrame, preset: PresetName) -> pd.DataFrame:
    """Add the preset's MODEL_CHANNELS columns to `df` (returns a copy).

    `df` must already contain the raw feature_cols (windspeed_100m,
    winddirection_100m, ...). Mirrors the per-script channel-derivation
    snippets (see repo_and_api_plan.md C6):
        1ch: u = windspeed_100m
        2ch: U = -speed*sin(dir), V = -speed*cos(dir)   (wind-vector decomposition)
        3ch: u = windspeed_100m, v_sin = sin(dir), v_cos = cos(dir)
    """
    df = df.copy()
    speed = df["windspeed_100m"]
    direction_rad = np.deg2rad(df["winddirection_100m"])

    if preset == "1ch":
        df["u"] = speed
    elif preset == "2ch":
        df["U"] = -speed * np.sin(direction_rad)
        df["V"] = -speed * np.cos(direction_rad)
    elif preset == "3ch":
        df["u"] = speed
        df["v_sin"] = np.sin(direction_rad)
        df["v_cos"] = np.cos(direction_rad)
    else:
        raise ValueError(f"Unknown preset: {preset!r}")

    return df


def decode_channels(samples: np.ndarray, preset: PresetName):
    """Inverse of `derive_channels`, applied to destandardized (physical-unit)
    samples of shape (..., n_channels) — i.e. channel is the last axis.

    Returns (speed, direction_deg) arrays with shape samples.shape[:-1].
    Mirrors each script's `reconstruct_to_dataframe` per-preset decode logic.
    """
    if preset == "1ch":
        speed = np.clip(samples[..., 0], 0.0, None)
        direction = np.zeros_like(speed)
    elif preset == "2ch":
        U = samples[..., 0]
        V = samples[..., 1]
        speed, direction = uv_to_speed_dir(U, V)
    elif preset == "3ch":
        speed = np.clip(samples[..., 0], 0.0, None)
        v_sin = samples[..., 1]
        v_cos = samples[..., 2]
        direction = np.rad2deg(np.arctan2(v_sin, v_cos)) % 360.0
    else:
        raise ValueError(f"Unknown preset: {preset!r}")

    return speed, direction


# ---------------------------------------------------------------------------
# Wind-regime bucketing
# ---------------------------------------------------------------------------

WIND_BUCKET_LABELS = ["low", "mid", "high"]

# Ordinal encoding alternative to one-hot (see build_condition_matrix's
# wind_regime_encoding param). Deliberately NOT centered on 0 (i.e. not
# {-1, 0, 1}): classifier-free guidance's "unconditional" branch zeroes out
# the regime slice, so if "mid" were encoded as exactly 0.0 it would be
# numerically indistinguishable from "no regime given at all" -- guidance
# would compute cond_pred - uncond_pred == 0 for every "mid" request by
# construction, regardless of guidance_scale, silently making guidance inert
# specifically for mid. Shifting the whole scale away from zero (1/2/3
# instead of -1/0/1) preserves the equal-spacing/midpoint property
# (mid == (low+high)/2) while keeping every real category distinguishable
# from the null vector.
ORDINAL_WIND_BUCKET_VALUES = {"low": 1.0, "mid": 2.0, "high": 3.0}


def assign_wind_bucket(wind_mean: float, q1: float, q2: float) -> str:
    if wind_mean < q1:
        return "low"
    elif wind_mean < q2:
        return "mid"
    else:
        return "high"


# ---------------------------------------------------------------------------
# Condition-vector construction
# ---------------------------------------------------------------------------

NUMERIC_COND_COLS = ["temp_max", "temp_min", "temp_spread"]
MONTHS = list(range(1, 13))
DAYS_OF_WEEK = list(range(7))


def build_condition_matrix(
    cond_df: pd.DataFrame,
    condition_on: dict[str, bool],
    numeric_mean: np.ndarray,
    numeric_std: np.ndarray,
    numeric_cond_cols: list[str] = NUMERIC_COND_COLS,
    months: list[int] = MONTHS,
    days_of_week: list[int] = DAYS_OF_WEEK,
    wind_buckets: list[str] = WIND_BUCKET_LABELS,
    wind_regime_encoding: str = "onehot",
    wind_persistence_mean: float | None = None,
    wind_persistence_std: float | None = None,
) -> np.ndarray:
    """Build the condition matrix z from a per-day condition dataframe.

    `numeric_mean`/`numeric_std` must be fit on the TRAIN split only and
    passed in explicitly (this replaces the closure-over-module-globals
    pattern used in the original scripts).

    `wind_regime_encoding`: "onehot" (default, 3 dims, one per bucket) or
    "ordinal" (1 dim, ORDINAL_WIND_BUCKET_VALUES -- see that constant's
    docstring for why "mid" steering measurably failed under one-hot and why
    ordinal is expected to fix it).

    `wind_persistence_mean`/`wind_persistence_std`: TRAIN-split standardization
    stats for the `prev_day_wind_mean` column, required when
    `condition_on["wind_persistence"]` is enabled -- see data.py's
    `prepare_datasets` for how that column is computed (yesterday's actual
    daily-mean windspeed, with days lacking a valid immediate predecessor
    imputed to the train mean, i.e. standardizing to 0.0 = "no info").
    """
    parts = []

    if condition_on.get("month"):
        month_oh = pd.get_dummies(cond_df["month"]).reindex(columns=months, fill_value=0)
        parts.append(month_oh.values.astype(np.float32))

    if condition_on.get("day_of_week"):
        dow_oh = pd.get_dummies(cond_df["day_of_week"]).reindex(columns=days_of_week, fill_value=0)
        parts.append(dow_oh.values.astype(np.float32))

    if condition_on.get("wind_regime_bucket"):
        if wind_regime_encoding == "ordinal":
            ordinal = cond_df["wind_bucket"].map(ORDINAL_WIND_BUCKET_VALUES).values.astype(np.float32)
            parts.append(ordinal.reshape(-1, 1))
        elif wind_regime_encoding == "onehot":
            bucket_oh = pd.get_dummies(cond_df["wind_bucket"]).reindex(columns=wind_buckets, fill_value=0)
            parts.append(bucket_oh.values.astype(np.float32))
        else:
            raise ValueError(f"Unknown wind_regime_encoding: {wind_regime_encoding!r}; expected 'onehot' or 'ordinal'")

    if condition_on.get("temp_min_max_spread"):
        numeric = cond_df[numeric_cond_cols].values.astype(np.float32)
        numeric = (numeric - numeric_mean) / numeric_std
        parts.append(numeric.astype(np.float32))

    if condition_on.get("wind_persistence"):
        if wind_persistence_mean is None or wind_persistence_std is None:
            raise ValueError("wind_persistence_mean/wind_persistence_std required when condition_on['wind_persistence'] is enabled")
        persistence = cond_df["prev_day_wind_mean"].values.astype(np.float32)
        persistence = (persistence - wind_persistence_mean) / wind_persistence_std
        parts.append(persistence.reshape(-1, 1).astype(np.float32))

    if not parts:
        raise ValueError("condition_on has no enabled features; cond matrix would be empty")

    return np.concatenate(parts, axis=1)


def wind_regime_slice(
    condition_on: dict[str, bool],
    months: list[int] = MONTHS,
    days_of_week: list[int] = DAYS_OF_WEEK,
    wind_buckets: list[str] = WIND_BUCKET_LABELS,
    wind_regime_encoding: str = "onehot",
) -> slice:
    """Index range of the wind-regime block within a condition vector built
    by `build_condition_matrix`/`build_condition_vector` -- 3 dims for
    "onehot", 1 dim for "ordinal" (must match whatever encoding the vector
    was actually built with).

    Mirrors those functions' fixed part order (month, day_of_week,
    wind_regime_bucket, temp_min_max_spread) exactly -- if that order ever
    changes, this must change with it. Used for regime-scoped classifier-free
    guidance (see train.py's `cond_dropout_scope` / diffusion._cfg_predict's
    `guidance_slice`), which needs to null out only the regime component of
    z rather than the whole vector.

    Raises:
        ValueError: `condition_on["wind_regime_bucket"]` is not enabled --
            there is no regime slice to compute.
    """
    if not condition_on.get("wind_regime_bucket"):
        raise ValueError("condition_on['wind_regime_bucket'] is not enabled; no wind-regime slice exists")

    offset = 0
    if condition_on.get("month"):
        offset += len(months)
    if condition_on.get("day_of_week"):
        offset += len(days_of_week)
    width = 1 if wind_regime_encoding == "ordinal" else len(wind_buckets)
    return slice(offset, offset + width)


def wind_persistence_slice(
    condition_on: dict[str, bool],
    months: list[int] = MONTHS,
    days_of_week: list[int] = DAYS_OF_WEEK,
    wind_buckets: list[str] = WIND_BUCKET_LABELS,
    wind_regime_encoding: str = "onehot",
    numeric_cond_cols: list[str] = NUMERIC_COND_COLS,
) -> slice:
    """Index range of the 1-dim wind-persistence block within a condition
    vector built by `build_condition_matrix` -- always last in the fixed part
    order (month, day_of_week, wind_regime_bucket, temp_min_max_spread,
    wind_persistence), so adding it never disturbs `wind_regime_slice`'s
    offset. Used for the persistence feature's own independent training-time
    dropout (see train.py) -- kept separate from `cond_dropout_scope`'s
    regime-CFG masking since the two serve different purposes.

    Raises:
        ValueError: `condition_on["wind_persistence"]` is not enabled.
    """
    if not condition_on.get("wind_persistence"):
        raise ValueError("condition_on['wind_persistence'] is not enabled; no wind-persistence slice exists")

    offset = 0
    if condition_on.get("month"):
        offset += len(months)
    if condition_on.get("day_of_week"):
        offset += len(days_of_week)
    if condition_on.get("wind_regime_bucket"):
        offset += 1 if wind_regime_encoding == "ordinal" else len(wind_buckets)
    if condition_on.get("temp_min_max_spread"):
        offset += len(numeric_cond_cols)
    return slice(offset, offset + 1)


# ---------------------------------------------------------------------------
# WindTransformPipeline (copied from repo-root wind_transform.py; orphaned
# prior work, preserved here for future use — not currently wired into the
# training/eval pipeline above).
# ---------------------------------------------------------------------------


class WindTransformPipeline:
    def __init__(self):
        self.alpha = None
        self.mean_u = None
        self.mean_v = None
        self.diurnal_clim_ustream = None
        self.monthly_clim_ustream = None
        self.diurnal_clim_vcross = None
        self.monthly_clim_vcross = None

    @staticmethod
    def met_wind_to_uv(speed, direction_deg):
        direction_rad = np.deg2rad(direction_deg)
        u = -speed * np.sin(direction_rad)
        v = -speed * np.cos(direction_rad)
        return u, v

    @staticmethod
    def uv_to_met_wind(u, v):
        speed = np.sqrt(u**2 + v**2)
        direction_rad = np.arctan2(-u, -v)
        direction_deg = (np.rad2deg(direction_rad)) % 360.0
        return speed, direction_deg

    @staticmethod
    def compute_rotation_angle(u, v):
        mean_u = np.nanmean(u)
        mean_v = np.nanmean(v)
        alpha = np.arctan2(mean_v, mean_u)
        return alpha, mean_u, mean_v

    @staticmethod
    def to_streamwise(u, v, alpha):
        ustream = u * np.cos(alpha) + v * np.sin(alpha)
        vcross = -u * np.sin(alpha) + v * np.cos(alpha)
        return ustream, vcross

    @staticmethod
    def from_streamwise(ustream, vcross, alpha):
        u = ustream * np.cos(alpha) - vcross * np.sin(alpha)
        v = ustream * np.sin(alpha) + vcross * np.cos(alpha)
        return u, v

    @staticmethod
    def compute_climatology(series, by="hour"):
        if by == "hour":
            key = series.index.hour
        elif by == "month":
            key = series.index.month
        elif by == "dayofyear":
            key = series.index.dayofyear
        else:
            raise ValueError("by must be 'hour', 'month', or 'dayofyear'")
        return series.groupby(key).mean()

    @staticmethod
    def remove_climatology(series, clim, by="hour"):
        if by == "hour":
            key = series.index.hour
        elif by == "month":
            key = series.index.month
        elif by == "dayofyear":
            key = series.index.dayofyear
        else:
            raise ValueError("by must be 'hour', 'month', or 'dayofyear'")
        return series - clim.reindex(key).values

    def _full_deseasonalize(self, series):
        dclim = self.compute_climatology(series, by="hour")
        after_diurnal = self.remove_climatology(series, dclim, by="hour")
        sclim = self.compute_climatology(after_diurnal, by="month")
        residual = self.remove_climatology(after_diurnal, sclim, by="month")
        return residual, dclim, sclim

    def fit(self, df, speed_col="windspeed100m", dir_col="winddirection100m"):
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        u, v = self.met_wind_to_uv(df[speed_col].values, df[dir_col].values)
        self.alpha, self.mean_u, self.mean_v = self.compute_rotation_angle(u, v)

        ustream, vcross = self.to_streamwise(u, v, self.alpha)
        ustream_s = pd.Series(ustream, index=df.index, name="ustream")
        vcross_s = pd.Series(vcross, index=df.index, name="vcross")

        _, self.diurnal_clim_ustream, self.monthly_clim_ustream = self._full_deseasonalize(ustream_s)
        _, self.diurnal_clim_vcross, self.monthly_clim_vcross = self._full_deseasonalize(vcross_s)
        return self

    def transform(self, df, speed_col="windspeed100m", dir_col="winddirection100m"):
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        if self.alpha is None:
            raise RuntimeError("Call fit() before transform().")

        u, v = self.met_wind_to_uv(df[speed_col].values, df[dir_col].values)
        ustream, vcross = self.to_streamwise(u, v, self.alpha)

        df["u"] = u
        df["v"] = v
        df["ustream"] = ustream
        df["vcross"] = vcross

        ustream_s = pd.Series(ustream, index=df.index, name="ustream")
        vcross_s = pd.Series(vcross, index=df.index, name="vcross")

        ustream_after_hour = self.remove_climatology(
            ustream_s, self.diurnal_clim_ustream, by="hour"
        )
        vcross_after_hour = self.remove_climatology(
            vcross_s, self.diurnal_clim_vcross, by="hour"
        )

        df["ustream_deseas"] = self.remove_climatology(
            ustream_after_hour, self.monthly_clim_ustream, by="month"
        )
        df["vcross_deseas"] = self.remove_climatology(
            vcross_after_hour, self.monthly_clim_vcross, by="month"
        )

        df["ustream_deseas_diff1"] = df["ustream_deseas"].diff()
        df["vcross_deseas_diff1"] = df["vcross_deseas"].diff()

        df["sinhour"] = np.sin(2 * np.pi * df.index.hour / 24.0)
        df["coshour"] = np.cos(2 * np.pi * df.index.hour / 24.0)
        df["sindoy"] = np.sin(2 * np.pi * (df.index.dayofyear - 1) / 365.25)
        df["cosdoy"] = np.cos(2 * np.pi * (df.index.dayofyear - 1) / 365.25)

        return df

    def inverse_transform(
        self,
        index,
        ustream_deseas_diff1,
        vcross_deseas_diff1,
        init_ustream_deseas,
        init_vcross_deseas,
    ):
        if self.alpha is None:
            raise RuntimeError("Call fit() before inverse_transform().")

        index = pd.DatetimeIndex(index)

        ustream_deseas_diff1 = pd.Series(ustream_deseas_diff1, index=index)
        vcross_deseas_diff1 = pd.Series(vcross_deseas_diff1, index=index)

        ustream_deseas = ustream_deseas_diff1.cumsum() + init_ustream_deseas
        vcross_deseas = vcross_deseas_diff1.cumsum() + init_vcross_deseas

        hours = index.hour
        months = index.month

        ustream = (
            ustream_deseas
            + self.monthly_clim_ustream.reindex(months).values
            + self.diurnal_clim_ustream.reindex(hours).values
        )
        vcross = (
            vcross_deseas
            + self.monthly_clim_vcross.reindex(months).values
            + self.diurnal_clim_vcross.reindex(hours).values
        )

        u, v = self.from_streamwise(ustream.values, vcross.values, self.alpha)
        speed, direction_deg = self.uv_to_met_wind(u, v)

        out = pd.DataFrame(index=index)
        out["ustream_deseas"] = ustream_deseas
        out["vcross_deseas"] = vcross_deseas
        out["ustream"] = ustream
        out["vcross"] = vcross
        out["u"] = u
        out["v"] = v
        out["windspeed100m"] = speed
        out["winddirection100m"] = direction_deg
        return out
