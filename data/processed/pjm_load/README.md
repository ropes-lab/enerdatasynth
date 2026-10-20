# `data/processed/pjm_load/`

PJM load, zone PS, after `enerdatasynth preprocess`. **Only `metadata.json` is tracked.** The arrays and tables below are built from your own copy of the source files (see [`../../raw/pjm_load/README.md`](../../raw/pjm_load/README.md)) and ignored by git.

`metadata.json` holds the scaler parameters (`load_scaler`, `temp_scaler`, `spread_scaler`), the condition spec, and the exclusion counts. The shipped checkpoint needs it to build conditioning vectors and convert generated profiles back to MW, which is why it stays. It contains summary statistics, not observations.

## What `preprocess` writes

| File | Shape for the shipped run | Contents |
|---|---|---|
| `X_daily_profiles.npy` / `X_daily_profiles_val.npy` | 2785×24 / 600×24 | Training and validation load profiles, min-max scaled per hour |
| `Z_conditions.npy` / `Z_conditions_val.npy` | 2785×23 / 600×23 | Conditioning vectors |
| `df_train.parquet` / `df_val.parquet` | | Full feature tables (for debugging and EDA) |
| `metadata.json` | | Scalers, exclusion counts, condition spec (tracked) |
| `README.md` | | `preprocess` rewrites this file with the full report for your run |

Without these files, `enerdatasynth.model("pjm_load", ...)` still generates. Training, sweeping, calibrating, validation plots, and failure analysis need them, and stop with a message saying so.

## Preprocessing decisions

Exclusions applied, in order, for the shipped run:

| Step | Reason | Days removed |
|---|---|---|
| 1 | DST days (23h or 25h, incompatible with a fixed 24-step output) | 20 |
| 2 | Missing load readings | 0 |
| 3 | Missing weather data | 128 |
| 4 | COVID lockdown (Mar–Jun 2020) | 122 |

3,653 raw days leave 3,385 clean days. The train/validation split is chronological: train 2016–2023 (2,785 days), validation 2024–2025 (600 days). This is not a forecasting task, so the split exists for reproducible development.

## Condition definitions

Z is a 23-dimensional vector:

| Variable | Encoding | Dim | Notes |
|---|---|---|---|
| month | one-hot | 12 | calendar month, 1–12 |
| day of week | one-hot | 7 | 0=Mon … 6=Sun |
| holiday | binary | 1 | US federal holiday flag |
| Tmax | z-score | 1 | daily max temperature |
| Tmin | z-score | 1 | daily min temperature |
| spread | z-score | 1 | max(load) − min(load) per day |

## Normalization

- **Load (X):** MinMaxScaler, fit per hour on training data only. Invert with `MW = x_norm * (max_h - min_h) + min_h`.
- **Temperature (Tmax, Tmin) and spread:** StandardScaler (z-score), fit on training data only. Invert with `T_F = z * scale + mean`.

## Reproducing

```bash
enerdatasynth preprocess
```
