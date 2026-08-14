"""
src/enerdatasynth/data/preprocess.py
=======================================
Fully reproducible preprocessing pipeline.

Raw PJM load CSVs + NOAA temperature data  -->  X, Z tensors ready for training.

Steps (in order):
  1. Load raw PJM load data (all years), filter to target zone.
  2. Load NOAA temperature data, compute daily Tmax / Tmin.
  3. Remove DST days (23h or 25h — incompatible with fixed 24-step output).
  4. Remove days with missing load values.
  5. Remove days with missing weather values.
  6. Remove COVID period (Mar-Jun 2020) — load behavior decoupled from weather.
  7. Build daily samples: pivot to (N_days, 24).
  8. Build conditioning vector Z:
       month (one-hot, 12) + dow (one-hot, 7) + holiday (binary, 1)
       + Tmax (z-scored, 1) + Tmin (z-scored, 1) + spread (z-scored, 1)
       = 23 dimensions total.
  9. Chronological train/val split.
  10. Normalize X with per-hour MinMax (fit on train only).
  11. Save X_daily_profiles.npy, Z_conditions.npy, metadata.json, README.md
      to data/processed/, then duplicate the entire folder to
      data/processed_backup/ so a corrupted re-run never destroys the
      only copy of the training data.

Run directly:
    python -m enerdatasynth.resources.pjm_load.data.preprocess
"""

from __future__ import annotations
import json
import shutil
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import holidays
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from pathlib import Path

from enerdatasynth.utils.config import (
    RAW_DIR, PROCESSED_DIR, PROCESSED_BACKUP, ZONE,
    N_MONTHS, N_DOW, COND_DIM, X_DIM, CONDITION_SPEC, EXCLUDED_BY_DESIGN,
    ensure_dirs,
)

LOAD_COLS = [f"h{i:02d}" for i in range(24)]


# ═════════════════════════════════════════════════════════════════════════════
# STEP-BY-STEP PREPROCESSING FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def load_raw_load_data(raw_dir: Path, zone: str) -> pd.DataFrame:
    """Load all hrl_load_metered_*.csv files and filter to one zone."""
    dfs = []
    for yr in range(2016, 2026):
        fp = raw_dir / f"hrl_load_metered_{yr}.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        df["year"] = yr
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No hrl_load_metered_*.csv files found in {raw_dir}")

    all_load = pd.concat(dfs, ignore_index=True)
    all_load["dt"]   = pd.to_datetime(all_load["datetime_beginning_ept"], format="mixed")
    all_load["hour"] = all_load["dt"].dt.hour

    zone_df = all_load[all_load["zone"] == zone].copy()
    zone_df = zone_df.sort_values("dt").reset_index(drop=True)
    zone_df["date"] = zone_df["dt"].dt.normalize()
    return zone_df


def load_temperature_data(raw_dir: Path, zone: str) -> pd.DataFrame:
    """Load NOAA temperature data, return daily Tmax/Tmin for the zone's station."""
    temp_fp = raw_dir / "pjm_hourly_temps_wide.csv"
    if not temp_fp.exists():
        raise FileNotFoundError(f"Temperature file not found: {temp_fp}")

    temp_raw = pd.read_csv(temp_fp)
    temp_raw["dt"] = (
        pd.to_datetime(temp_raw["datetime_ept"], utc=True)
          .dt.tz_convert("US/Eastern")
          .dt.tz_localize(None)
    )
    temp_raw["date"] = temp_raw["dt"].dt.normalize()

    daily_temp = (
        temp_raw.groupby("date")[zone]
        .agg(Tmax="max", Tmin="min")
        .reset_index()
    )
    daily_temp["date"] = pd.to_datetime(daily_temp["date"])
    return daily_temp


def build_daily_pivot(zone_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot hourly rows into one row per day with 24 hour-columns."""
    pivot = zone_df.pivot_table(index="date", columns="hour", values="mw")
    pivot.columns = [f"h{int(c):02d}" for c in pivot.columns]
    pivot = pivot.reset_index()
    pivot["date"] = pd.to_datetime(pivot["date"])
    return pivot


def apply_exclusions(pivot: pd.DataFrame, daily_temp: pd.DataFrame,
                      dst_dates: list) -> dict:
    """
    Apply all exclusion rules in order. Returns the cleaned dataframe
    plus a dict of exclusion counts for the README / metadata.

    dst_dates: list of Timestamps detected via the rigorous PRE-pivot
    hourly-row-count check (see `detect_dst_days`). We pass these in
    explicitly rather than re-detecting post-pivot, because a 23-hour
    day still produces 24 *columns* after pivoting (one is just NaN),
    which a post-pivot column-count check would silently miss.
    """
    total_raw = len(pivot)

    dst_dates_norm = pd.to_datetime(pd.Series(dst_dates)).dt.normalize()
    pivot["is_dst"] = pivot["date"].isin(set(dst_dates_norm))

    # 2. Missing load values (any NaN among the 24 hour columns,
    #    excluding days already flagged as DST — a 23-hour DST day
    #    will also show a NaN column, but we don't want to double-count
    #    it under "missing load")
    pivot["missing_load"] = pivot[LOAD_COLS].isnull().any(axis=1) & ~pivot["is_dst"]

    # 3. Merge weather, flag missing
    pivot = pivot.merge(daily_temp, on="date", how="left")
    pivot["missing_weather"] = pivot["Tmax"].isnull() | pivot["Tmin"].isnull()

    # 4. COVID period (currently excluded per project decision)
    pivot["year"]  = pivot["date"].dt.year
    pivot["month"] = pivot["date"].dt.month
    pivot["is_covid"] = (pivot["year"] == 2020) & (pivot["month"].isin([3, 4, 5, 6]))

    counts = {
        "total_raw_days":     int(total_raw),
        "dst_removed":        int(pivot["is_dst"].sum()),
        "missing_load_removed":    int(pivot["missing_load"].sum()),
        "missing_weather_removed": int(pivot["missing_weather"].sum()),
        "covid_removed":      int(pivot["is_covid"].sum()),
    }

    remove_mask = (
        pivot["is_dst"] | pivot["missing_load"]
        | pivot["missing_weather"] | pivot["is_covid"]
    )
    clean = pivot[~remove_mask].copy().reset_index(drop=True)
    counts["final_clean_days"] = int(len(clean))

    return clean, counts


def detect_dst_days(zone_df: pd.DataFrame) -> dict:
    """
    Rigorous DST detection done on the PRE-pivot hourly data:
    count actual hourly rows per date. Spring-forward days have 23,
    fall-back days have 25. Returns counts for cross-checking.
    """
    hours_per_day = zone_df.groupby("date")["mw"].count()
    dst_dates = hours_per_day[hours_per_day != 24]
    return {"n_dst_days_detected": int(len(dst_dates)), "dst_dates": dst_dates.index.tolist()}


def build_conditioning_vector(df: pd.DataFrame,
                               scaler_temp: StandardScaler | None = None,
                               scaler_spread: StandardScaler | None = None,
                               fit: bool = False):
    """
    Build the Z conditioning matrix.

    Z = [month_onehot(12), dow_onehot(7), is_holiday(1),
         Tmax_z(1), Tmin_z(1), spread_z(1)]   ->  23 dimensions

    If fit=True, scalers are fit on this dataframe (use only for training
    data) and returned. Otherwise the provided scalers are used to
    transform (use for validation data).
    """
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek          # 0=Mon ... 6=Sun
    us_hols   = holidays.US(years=range(2016, 2026))
    df["is_holiday"] = df["date"].dt.date.apply(lambda d: 1 if d in us_hols else 0)
    df["spread"]     = df[LOAD_COLS].max(axis=1) - df[LOAD_COLS].min(axis=1)

    # One-hot month (1-12) and dow (0-6)
    month_onehot = np.zeros((len(df), N_MONTHS), dtype=np.float32)
    month_onehot[np.arange(len(df)), df["month"].values - 1] = 1.0

    dow_onehot = np.zeros((len(df), N_DOW), dtype=np.float32)
    dow_onehot[np.arange(len(df)), df["dow"].values] = 1.0

    holiday_col = df["is_holiday"].values.reshape(-1, 1).astype(np.float32)

    if fit:
        scaler_temp = StandardScaler().fit(df[["Tmax", "Tmin"]].values)
        scaler_spread = StandardScaler().fit(df[["spread"]].values)

    if scaler_temp is None or scaler_spread is None:
        raise ValueError("scaler_temp and scaler_spread must be provided when fit=False")

    temp_z   = scaler_temp.transform(df[["Tmax", "Tmin"]].values).astype(np.float32)
    spread_z = scaler_spread.transform(df[["spread"]].values).astype(np.float32)

    Z = np.concatenate([month_onehot, dow_onehot, holiday_col, temp_z, spread_z], axis=1)
    assert Z.shape[1] == COND_DIM, f"Expected {COND_DIM} dims, got {Z.shape[1]}"

    return Z.astype(np.float32), df, scaler_temp, scaler_spread


def duplicate_processed_dir(src: Path, dst: Path) -> None:
    """
    Make a frozen backup copy of the processed data directory.
    If a run later corrupts data/processed/, this backup is untouched
    and can be restored by copying it back.
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def run_preprocessing(raw_dir: Path = RAW_DIR,
                       processed_dir: Path = PROCESSED_DIR,
                       backup_dir: Path = PROCESSED_BACKUP,
                       zone: str = ZONE) -> dict:
    ensure_dirs()
    print("=" * 70)
    print(f" PREPROCESSING PIPELINE  —  {zone} zone")
    print("=" * 70)

    # 1-2. Load raw data
    print("\n[1/6] Loading raw load + temperature data...")
    zone_df    = load_raw_load_data(raw_dir, zone)
    daily_temp = load_temperature_data(raw_dir, zone)
    dst_check  = detect_dst_days(zone_df)
    print(f"  Loaded {len(zone_df):,} hourly rows for zone {zone}")
    print(f"  DST days detected (pre-pivot check): {dst_check['n_dst_days_detected']}")

    # 3. Pivot to daily
    print("\n[2/6] Building daily pivot table...")
    pivot = build_daily_pivot(zone_df)
    print(f"  {len(pivot)} raw calendar days before exclusions")

    # 4. Apply all exclusions
    print("\n[3/6] Applying exclusion rules...")
    clean, exclusion_counts = apply_exclusions(pivot, daily_temp, dst_check["dst_dates"])
    for k, v in exclusion_counts.items():
        print(f"  {k:<28} {v:>6}")

    # 5. Chronological split
    print("\n[4/6] Chronological train/val split...")
    clean = clean.sort_values("date").reset_index(drop=True)
    train_mask = clean["year"] < 2024
    df_train = clean[train_mask].copy().reset_index(drop=True)
    df_val   = clean[~train_mask].copy().reset_index(drop=True)
    print(f"  Train: {len(df_train)} days (2016-2023)")
    print(f"  Val:   {len(df_val)} days (2024-2025)")

    # 6. Build conditioning vectors (fit scalers on train only)
    print("\n[5/6] Building conditioning vectors (Z)...")
    Z_train, df_train2, scaler_temp, scaler_spread = build_conditioning_vector(
        df_train, fit=True
    )
    Z_val, df_val2, _, _ = build_conditioning_vector(
        df_val, scaler_temp=scaler_temp, scaler_spread=scaler_spread, fit=False
    )
    print(f"  Z_train shape: {Z_train.shape}")
    print(f"  Z_val shape:   {Z_val.shape}")
    print(f"  Conditioning dim: {COND_DIM} "
          f"(month=12, dow=7, holiday=1, Tmax=1, Tmin=1, spread=1)")

    # 7. Normalize X — per-hour MinMax, fit on train only
    scaler_load = MinMaxScaler(feature_range=(0, 1))
    scaler_load.fit(df_train2[LOAD_COLS].values)
    X_train = scaler_load.transform(df_train2[LOAD_COLS].values).astype(np.float32)
    X_val   = scaler_load.transform(df_val2[LOAD_COLS].values).astype(np.float32)
    print(f"  X_train shape: {X_train.shape}")
    print(f"  X_val shape:   {X_val.shape}")

    # 8. Save everything
    print("\n[6/6] Saving processed data + backup...")
    processed_dir.mkdir(parents=True, exist_ok=True)

    np.save(processed_dir / "X_daily_profiles.npy", X_train)
    np.save(processed_dir / "X_daily_profiles_val.npy", X_val)
    np.save(processed_dir / "Z_conditions.npy", Z_train)
    np.save(processed_dir / "Z_conditions_val.npy", Z_val)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "zone": zone,
        "sample_count": {
            "train": int(len(df_train2)),
            "val": int(len(df_val2)),
            "total_raw": exclusion_counts["total_raw_days"],
        },
        "exclusion_counts": exclusion_counts,
        "excluded_by_design": EXCLUDED_BY_DESIGN,
        "X_shape_train": list(X_train.shape),
        "X_shape_val":   list(X_val.shape),
        "Z_shape_train": list(Z_train.shape),
        "Z_shape_val":   list(Z_val.shape),
        "condition_spec": CONDITION_SPEC,
        "condition_dim": COND_DIM,
        "condition_order": (
            [f"month_{i+1}" for i in range(N_MONTHS)]
            + [f"dow_{i}" for i in range(N_DOW)]
            + ["is_holiday", "Tmax_z", "Tmin_z", "spread_z"]
        ),
        "load_scaler": {
            "type": "MinMaxScaler [0,1] per-hour",
            "fitted_on": "train days only",
            "data_min_": scaler_load.data_min_.tolist(),
            "data_max_": scaler_load.data_max_.tolist(),
            "invert_formula": "MW = x_norm * (data_max_[h] - data_min_[h]) + data_min_[h]",
        },
        "temp_scaler": {
            "type": "StandardScaler (z-score)",
            "fitted_on": "train days only",
            "columns": ["Tmax", "Tmin"],
            "mean_": scaler_temp.mean_.tolist(),
            "scale_": scaler_temp.scale_.tolist(),
        },
        "spread_scaler": {
            "type": "StandardScaler (z-score)",
            "fitted_on": "train days only",
            "definition": "max(load_day) - min(load_day)",
            "mean_": scaler_spread.mean_.tolist(),
            "scale_": scaler_spread.scale_.tolist(),
        },
        "train_val_split": {
            "method": "chronological",
            "train_years": "2016-2023",
            "val_years": "2024-2025",
        },
    }
    with open(processed_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    df_train2.to_parquet(processed_dir / "df_train.parquet")
    df_val2.to_parquet(processed_dir / "df_val.parquet")

    write_readme(processed_dir, metadata)

    # 9. Duplicate to backup — frozen copy, never touched again
    duplicate_processed_dir(processed_dir, backup_dir)
    print(f"  Saved to: {processed_dir}")
    print(f"  Backed up to: {backup_dir}  (frozen copy — safe if re-runs corrupt data/processed/)")

    print("\n" + "=" * 70)
    print(" PREPROCESSING COMPLETE")
    print("=" * 70)
    return metadata


def write_readme(processed_dir: Path, metadata: dict) -> None:
    readme = f"""# Processed Dataset — PJM {metadata['zone']} Zone

Generated: {metadata['created_at_utc']}

## Sample Count
- Training days: {metadata['sample_count']['train']}
- Validation days: {metadata['sample_count']['val']}
- Total raw days before cleaning: {metadata['sample_count']['total_raw']}

## Preprocessing Decisions

Exclusions applied, in order:
| Step | Reason | Days removed |
|------|--------|---------------|
| 1 | DST days (23h or 25h — incompatible with fixed 24-step output) | {metadata['exclusion_counts']['dst_removed']} |
| 2 | Missing load readings | {metadata['exclusion_counts']['missing_load_removed']} |
| 3 | Missing weather data | {metadata['exclusion_counts']['missing_weather_removed']} |
| 4 | COVID lockdown (Mar-Jun 2020) | {metadata['exclusion_counts']['covid_removed']} |

Final clean days: {metadata['exclusion_counts']['final_clean_days']}

Train/val split: chronological. Train = 2016-2023, Val = 2024-2025.
This is NOT a forecasting task, so the split is for reproducible
development; in-sample random validation remains an option.

## Condition Definitions

Z is a {metadata['condition_dim']}-dimensional vector:

| Variable | Encoding | Dim | Notes |
|----------|----------|-----|-------|
| month | one-hot | 12 | calendar month, 1-12 |
| day of week | one-hot | 7 | 0=Mon ... 6=Sun |
| holiday | binary | 1 | US federal holiday flag |
| Tmax | z-score | 1 | daily max temperature |
| Tmin | z-score | 1 | daily min temperature |
| spread | z-score | 1 | max(load) - min(load) per day |

Explicitly excluded from this baseline (tracked as future work only):
{chr(10).join('- ' + x for x in metadata['excluded_by_design'])}

## Normalization Methods

**Load (X):** MinMaxScaler, fit per-hour, fit on training data only.
Invert with: `MW = x_norm * (max_h - min_h) + min_h`

**Temperature (Tmax, Tmin):** StandardScaler (z-score), fit on training
data only. Invert with: `T_F = z * scale + mean`

**Spread:** StandardScaler (z-score), fit on training data only.

All scaler parameters are saved in `metadata.json` under `load_scaler`,
`temp_scaler`, and `spread_scaler`.

## Files in this Directory

- `X_daily_profiles.npy` — shape {metadata['X_shape_train']}, training load profiles
- `X_daily_profiles_val.npy` — shape {metadata['X_shape_val']}, validation load profiles
- `Z_conditions.npy` — shape {metadata['Z_shape_train']}, training conditioning vectors
- `Z_conditions_val.npy` — shape {metadata['Z_shape_val']}, validation conditioning vectors
- `metadata.json` — all scaler parameters, exclusion counts, condition spec
- `df_train.parquet` / `df_val.parquet` — full feature tables (for debugging/EDA)
- `README.md` — this file

## Reproducing

```bash
python -m enerdatasynth.resources.pjm_load.data.preprocess
```

A frozen backup of this entire directory is kept at `data/processed_backup/`
immediately after generation, so that if a later step corrupts these files,
the original is recoverable by copying the backup back over this folder.
"""
    with open(processed_dir / "README.md", "w") as f:
        f.write(readme)


if __name__ == "__main__":
    run_preprocessing()
