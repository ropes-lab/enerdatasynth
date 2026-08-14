# Processed Dataset — PJM PS Zone

Generated: 2026-06-22T19:57:58.608729+00:00

## Sample Count
- Training days: 2785
- Validation days: 600
- Total raw days before cleaning: 3653

## Preprocessing Decisions

Exclusions applied, in order:
| Step | Reason | Days removed |
|------|--------|---------------|
| 1 | DST days (23h or 25h — incompatible with fixed 24-step output) | 20 |
| 2 | Missing load readings | 0 |
| 3 | Missing weather data | 128 |
| 4 | COVID lockdown (Mar-Jun 2020) | 122 |

Final clean days: 3385

Train/val split: chronological. Train = 2016-2023, Val = 2024-2025.
This is NOT a forecasting task, so the split is for reproducible
development; in-sample random validation remains an option.

## Condition Definitions

Z is a 23-dimensional vector:

| Variable | Encoding | Dim | Notes |
|----------|----------|-----|-------|
| month | one-hot | 12 | calendar month, 1-12 |
| day of week | one-hot | 7 | 0=Mon ... 6=Sun |
| holiday | binary | 1 | US federal holiday flag |
| Tmax | z-score | 1 | daily max temperature |
| Tmin | z-score | 1 | daily min temperature |
| spread | z-score | 1 | max(load) - min(load) per day |

Explicitly excluded from this baseline (tracked as future work only):
- MW_anchor
- Fourier modes
- custom losses
- higher moments
- user-facing abstractions
- unified PJM model
- attention
- transformers

## Normalization Methods

**Load (X):** MinMaxScaler, fit per-hour, fit on training data only.
Invert with: `MW = x_norm * (max_h - min_h) + min_h`

**Temperature (Tmax, Tmin):** StandardScaler (z-score), fit on training
data only. Invert with: `T_F = z * scale + mean`

**Spread:** StandardScaler (z-score), fit on training data only.

All scaler parameters are saved in `metadata.json` under `load_scaler`,
`temp_scaler`, and `spread_scaler`.

## Files in this Directory

- `X_daily_profiles.npy` — shape [2785, 24], training load profiles
- `X_daily_profiles_val.npy` — shape [600, 24], validation load profiles
- `Z_conditions.npy` — shape [2785, 23], training conditioning vectors
- `Z_conditions_val.npy` — shape [600, 23], validation conditioning vectors
- `metadata.json` — all scaler parameters, exclusion counts, condition spec
- `df_train.parquet` / `df_val.parquet` — full feature tables (for debugging/EDA)
- `README.md` — this file

## Reproducing

```bash
python -m src.data.preprocess
```

A frozen backup of this entire directory is kept at `data/processed_backup/`
immediately after generation, so that if a later step corrupts these files,
the original is recoverable by copying the backup back over this folder.
