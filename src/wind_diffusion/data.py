"""Data loading, windowing, and the PyTorch Dataset for wind_diffusion.

The original scripts did all of this inline at module/notebook level (load
CSV -> derive channels -> window by day -> compute per-day conditions ->
split -> standardize -> build condition matrix). This module extracts that
sequence into composable functions plus the `DailyWindowDataset` class, and
`prepare_datasets()` orchestrates the full pipeline (used by train.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from wind_diffusion.config import WindDiffusionConfig
from wind_diffusion.transforms import (
    DAYS_OF_WEEK,
    MONTHS,
    NUMERIC_COND_COLS,
    WIND_BUCKET_LABELS,
    assign_wind_bucket,
    build_condition_matrix,
    decode_channels,
    derive_channels,
    standardize_windows,
)

# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------


def load_raw_dataframe(config: WindDiffusionConfig, data_path: str | None = None) -> pd.DataFrame:
    """Load, sort, and (optionally) resample the raw CSV to a regular time grid.

    Returns a dataframe indexed by timestamp with just `config.feature_cols`.
    """
    path = data_path or config.data_path
    raw_df = pd.read_csv(path)
    raw_df[config.timestamp_col] = pd.to_datetime(raw_df[config.timestamp_col], utc=True)
    raw_df = raw_df.sort_values(config.timestamp_col).reset_index(drop=True)

    df = raw_df[[config.timestamp_col] + config.feature_cols].copy()
    df = df.set_index(config.timestamp_col)

    if config.resample:
        df = df.resample(config.expected_freq).mean()
        df[config.feature_cols] = df[config.feature_cols].interpolate(limit=3)

    df = df.dropna(subset=config.feature_cols)
    return df


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def build_daily_windows(
    df: pd.DataFrame,
    model_channels: list[str],
    seq_len: int,
    drop_incomplete_days: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Group `df` (already containing MODEL_CHANNELS columns) by calendar day
    and stack complete days into a (n_days, seq_len, n_channels) array, plus
    per-day metadata dicts (raw windspeed/direction/temperature arrays).
    """
    df = df.copy()
    df["date"] = df.index.date

    windows: list[np.ndarray] = []
    window_meta: list[dict[str, Any]] = []

    for date, day_df in df.groupby("date"):
        if len(day_df) != seq_len:
            if drop_incomplete_days:
                continue
            else:
                continue  # pad/truncate not supported, same as original scripts

        day_df = day_df.sort_index()
        arr = day_df[model_channels].values.astype(np.float32)
        if np.isnan(arr).any():
            continue

        windows.append(arr)
        meta: dict[str, Any] = {
            "date": pd.Timestamp(date),
            "timestamps": day_df.index,
        }
        for col in ("windspeed_100m", "winddirection_100m", "temperature_100m"):
            if col in day_df.columns:
                meta[col] = day_df[col].values
        window_meta.append(meta)

    if windows:
        windows_arr = np.stack(windows, axis=0)
    else:
        windows_arr = np.empty((0, seq_len, len(model_channels)), dtype=np.float32)

    return windows_arr, window_meta


def build_condition_dataframe(window_meta: list[dict[str, Any]]) -> pd.DataFrame:
    """Per-day conditioning features (month, day_of_week, temp stats, wind mean)."""
    cond_rows = []
    for meta in window_meta:
        date = meta["date"]
        temp = meta["temperature_100m"]
        wind = meta["windspeed_100m"]

        cond_rows.append(
            {
                "date": date,
                "month": date.month,
                "day_of_week": date.dayofweek,
                "temp_max": float(np.max(temp)),
                "temp_min": float(np.min(temp)),
                "temp_spread": float(np.max(temp) - np.min(temp)),
                "wind_mean": float(np.mean(wind)),
            }
        )
    return pd.DataFrame(cond_rows)


def split_indices(
    cond_df: pd.DataFrame, config: WindDiffusionConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) into cond_df/windows, by date or fraction."""
    n_days = len(cond_df)
    order = np.argsort(cond_df["date"].values)

    if config.split_date is not None:
        split_date = pd.Timestamp(config.split_date, tz=cond_df["date"].iloc[0].tz)
        train_mask = cond_df["date"].values < split_date
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(~train_mask)[0]
    else:
        n_train = int(n_days * config.split_frac)
        train_idx = order[:n_train]
        val_idx = order[n_train:]

    if len(train_idx) == 0 or len(val_idx) == 0:
        raise ValueError("Split produced an empty set - adjust split_frac/split_date.")

    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Normalization stats container
# ---------------------------------------------------------------------------


@dataclass
class NormStats:
    target_mean: np.ndarray
    target_std: np.ndarray
    numeric_cond_mean: np.ndarray
    numeric_cond_std: np.ndarray
    wind_bucket_edges: tuple[float, float]
    model_channels: list[str]
    numeric_cond_cols: list[str] = field(default_factory=lambda: list(NUMERIC_COND_COLS))
    months: list[int] = field(default_factory=lambda: list(MONTHS))
    days_of_week: list[int] = field(default_factory=lambda: list(DAYS_OF_WEEK))
    wind_buckets: list[str] = field(default_factory=lambda: list(WIND_BUCKET_LABELS))
    cond_dim: int = 0
    wind_regime_encoding: str = "onehot"
    wind_persistence_mean: float = 0.0
    wind_persistence_std: float = 1.0
    persistence_by_regime: dict[str, list[float]] = field(default_factory=dict)
    diurnal_profile_by_regime: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_mean": np.asarray(self.target_mean).tolist(),
            "target_std": np.asarray(self.target_std).tolist(),
            "numeric_cond_mean": np.asarray(self.numeric_cond_mean).tolist(),
            "numeric_cond_std": np.asarray(self.numeric_cond_std).tolist(),
            "wind_bucket_edges": [float(x) for x in self.wind_bucket_edges],
            "model_channels": self.model_channels,
            "numeric_cond_cols": self.numeric_cond_cols,
            "months": self.months,
            "days_of_week": self.days_of_week,
            "wind_buckets": self.wind_buckets,
            "cond_dim": self.cond_dim,
            "wind_regime_encoding": self.wind_regime_encoding,
            "wind_persistence_mean": float(self.wind_persistence_mean),
            "wind_persistence_std": float(self.wind_persistence_std),
            "persistence_by_regime": self.persistence_by_regime,
            "diurnal_profile_by_regime": self.diurnal_profile_by_regime,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NormStats":
        return cls(
            target_mean=np.array(d["target_mean"], dtype=np.float32),
            target_std=np.array(d["target_std"], dtype=np.float32),
            numeric_cond_mean=np.array(d["numeric_cond_mean"], dtype=np.float32),
            numeric_cond_std=np.array(d["numeric_cond_std"], dtype=np.float32),
            wind_bucket_edges=tuple(d["wind_bucket_edges"]),
            model_channels=list(d["model_channels"]),
            numeric_cond_cols=list(d.get("numeric_cond_cols", NUMERIC_COND_COLS)),
            months=list(d.get("months", MONTHS)),
            days_of_week=list(d.get("days_of_week", DAYS_OF_WEEK)),
            wind_buckets=list(d.get("wind_buckets", WIND_BUCKET_LABELS)),
            cond_dim=int(d.get("cond_dim", 0)),
            wind_regime_encoding=str(d.get("wind_regime_encoding", "onehot")),
            wind_persistence_mean=float(d.get("wind_persistence_mean", 0.0)),
            wind_persistence_std=float(d.get("wind_persistence_std", 1.0)),
            persistence_by_regime=dict(d.get("persistence_by_regime", {})),
            diurnal_profile_by_regime=dict(d.get("diurnal_profile_by_regime", {})),
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class DailyWindowDataset(Dataset):
    """Each item is one standardized daily window as (channels, seq_len) plus its condition vector."""

    def __init__(self, windows_norm: np.ndarray, cond_matrix: np.ndarray):
        w = torch.tensor(windows_norm, dtype=torch.float32)  # (N, seq_len, channels)
        self.windows = w.permute(0, 2, 1).contiguous()  # (N, channels, seq_len)
        self.cond = torch.tensor(cond_matrix, dtype=torch.float32)  # (N, cond_dim)

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int):
        x = self.windows[idx]  # (channels, seq_len)
        z = self.cond[idx]  # (cond_dim,)
        return x, z


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------


@dataclass
class PreparedData:
    train_dataset: DailyWindowDataset
    val_dataset: DailyWindowDataset
    norm_stats: NormStats
    train_cond_df: pd.DataFrame
    val_cond_df: pd.DataFrame
    train_meta: list[dict[str, Any]]
    val_meta: list[dict[str, Any]]


def prepare_datasets(config: WindDiffusionConfig, data_path: str | None = None) -> PreparedData:
    """Run the full data pipeline: load -> derive channels -> window -> split
    -> standardize -> build condition matrices. Mirrors the sequence of
    operations in the original scripts' module-level code.
    """
    df = load_raw_dataframe(config, data_path=data_path)
    df = derive_channels(df, config.preset)

    model_channels = config.model_channels
    windows, window_meta = build_daily_windows(
        df, model_channels, config.seq_len, config.drop_incomplete_days
    )
    if windows.shape[0] == 0:
        raise ValueError("No complete daily windows found - check data_path/seq_len/drop_incomplete_days.")

    cond_df = build_condition_dataframe(window_meta)

    # ---- Wind persistence: yesterday's actual daily-mean windspeed, looked
    # up by calendar date (not a positional shift) so a gap in the raw data
    # never silently pairs a day with a non-adjacent "yesterday." Computed
    # on the full (pre-split) cond_df so a val day's first entry can still
    # correctly see a train day's real value as "yesterday" -- this is not
    # leakage, since real "yesterday" data would be available at generation
    # time regardless of where the train/val boundary happens to fall. Days
    # with no valid immediate predecessor (start of dataset, or right after
    # a data gap) get NaN here, imputed to the train mean below.
    wind_mean_by_date = dict(zip(cond_df["date"], cond_df["wind_mean"]))
    one_day = pd.Timedelta(days=1)
    cond_df["prev_day_wind_mean"] = cond_df["date"].apply(
        lambda d: wind_mean_by_date.get(d - one_day, np.nan)
    )

    train_idx, val_idx = split_indices(cond_df, config)

    train_windows = windows[train_idx]
    val_windows = windows[val_idx]
    train_cond_df = cond_df.iloc[train_idx].reset_index(drop=True)
    val_cond_df = cond_df.iloc[val_idx].reset_index(drop=True)
    train_meta = [window_meta[i] for i in train_idx]
    val_meta = [window_meta[i] for i in val_idx]

    # ---- Standardize target channels ----
    n_channels = len(model_channels)
    target_mean = train_windows.reshape(-1, n_channels).mean(axis=0)
    target_std = train_windows.reshape(-1, n_channels).std(axis=0)
    target_std[target_std == 0] = 1.0

    train_windows_norm = standardize_windows(train_windows, target_mean, target_std)
    val_windows_norm = standardize_windows(val_windows, target_mean, target_std)

    # ---- Wind regime buckets (25th/75th percentiles of daily mean windspeed, TRAIN only) ----
    wind_q1, wind_q2 = np.quantile(train_cond_df["wind_mean"].values, [0.25, 0.75])
    for d in (train_cond_df, val_cond_df):
        d["wind_bucket"] = d["wind_mean"].apply(lambda x: assign_wind_bucket(x, wind_q1, wind_q2))

    # ---- Reference diurnal profile per regime (TRAIN only, physical
    # windspeed units) -- the mean time-of-day shape of real days in each
    # regime. Used by api.generate_wind_profile's rejection-sampling path to
    # rank bucket-matched candidates by how closely their OWN diurnal shape
    # tracks this reference, instead of keeping matches in arbitrary
    # generation order -- rejection sampling only filters on daily-mean-in-
    # bucket, which does nothing to keep the selected days' time-of-day
    # shape realistic (measured to regress diurnal_rmse, sometimes sharply --
    # see reports/REGIME_REJECTION_SAMPLING_INVESTIGATION.md).
    train_windspeed, _ = decode_channels(train_windows, config.preset)
    diurnal_profile_by_regime = {
        regime: train_windspeed[train_cond_df["wind_bucket"].values == regime].mean(axis=0).tolist()
        for regime in WIND_BUCKET_LABELS
    }

    # ---- Wind persistence: per-regime empirical distribution of REAL
    # observed previous-day values (TRAIN only, pre-imputation) -- used by
    # api.generate_wind_profile to sample a realistic default previous-day
    # value when the caller doesn't supply one explicitly.
    persistence_by_regime = {
        regime: train_cond_df.loc[
            train_cond_df["wind_bucket"] == regime, "prev_day_wind_mean"
        ].dropna().tolist()
        for regime in WIND_BUCKET_LABELS
    }

    # ---- Wind persistence standardization (TRAIN only; NaN -> train mean, which standardizes to 0.0 = "no info") ----
    wind_persistence_mean = float(train_cond_df["prev_day_wind_mean"].mean(skipna=True))
    wind_persistence_std = float(train_cond_df["prev_day_wind_mean"].std(skipna=True))
    if wind_persistence_std == 0:
        wind_persistence_std = 1.0
    for d in (train_cond_df, val_cond_df):
        d["prev_day_wind_mean"] = d["prev_day_wind_mean"].fillna(wind_persistence_mean)

    # ---- Numeric conditioning stats (TRAIN only) ----
    numeric_mean = train_cond_df[NUMERIC_COND_COLS].values.mean(axis=0)
    numeric_std = train_cond_df[NUMERIC_COND_COLS].values.std(axis=0)
    numeric_std[numeric_std == 0] = 1.0

    train_cond_matrix = build_condition_matrix(
        train_cond_df, config.condition_on, numeric_mean, numeric_std,
        wind_regime_encoding=config.wind_regime_encoding,
        wind_persistence_mean=wind_persistence_mean,
        wind_persistence_std=wind_persistence_std,
    )
    val_cond_matrix = build_condition_matrix(
        val_cond_df, config.condition_on, numeric_mean, numeric_std,
        wind_regime_encoding=config.wind_regime_encoding,
        wind_persistence_mean=wind_persistence_mean,
        wind_persistence_std=wind_persistence_std,
    )
    cond_dim = train_cond_matrix.shape[1]

    norm_stats = NormStats(
        target_mean=target_mean,
        target_std=target_std,
        numeric_cond_mean=numeric_mean,
        numeric_cond_std=numeric_std,
        wind_bucket_edges=(float(wind_q1), float(wind_q2)),
        model_channels=model_channels,
        cond_dim=cond_dim,
        wind_regime_encoding=config.wind_regime_encoding,
        wind_persistence_mean=wind_persistence_mean,
        wind_persistence_std=wind_persistence_std,
        persistence_by_regime=persistence_by_regime,
        diurnal_profile_by_regime=diurnal_profile_by_regime,
    )

    train_dataset = DailyWindowDataset(train_windows_norm, train_cond_matrix)
    val_dataset = DailyWindowDataset(val_windows_norm, val_cond_matrix)

    return PreparedData(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        norm_stats=norm_stats,
        train_cond_df=train_cond_df,
        val_cond_df=val_cond_df,
        train_meta=train_meta,
        val_meta=val_meta,
    )
