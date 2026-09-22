# Dataset Card

## Dataset Summary

Hourly electric load and temperature data for PJM Interconnection load zones, used to
train and evaluate conditional diffusion models that generate synthetic 24-hour daily
load profiles. This is not a load dataset released as a standalone product — it is the
input pipeline for the generative modeling work in this repository.

## Motivation

Collected to answer one research question (see `docs/research_status.md` §A): can a
conditional generative model reproduce the statistical, temporal, and operational
characteristics of real PJM daily load profiles, conditioned on calendar and weather
context?

## Composition

| Artifact | Location | Contents |
|---|---|---|
| Raw load CSVs | `data/raw/pjm_load/` | **Not bundled.** `data/raw/README.md` names the exact files and source; bring your own copy to reproduce preprocessing (§ below, "Reproducibility"). |
| Raw temperature CSVs | `data/raw/pjm_load/` | Same as above — not bundled. |
| Temperature cache | `data/cache/pjm_load/` | **Not bundled** (empty; `README.md` there explains it's a regenerable NOAA fetch cache, not source data). |
| Processed tensors (current package) | `data/processed/pjm_load/X_daily_profiles*.npy`, `Z_conditions*.npy`, `df_train.parquet`, `df_val.parquet`, `metadata.json` | **Bundled** — zone `PS` only, 23-dim conditioning, small (~1.3MB), required for `enerdatasynth.model("pjm_load", ...)` to construct at all |
| Processed tensors (research history) | `data/research_history/pjm_load/outputs/week3_outputs/*.npy` (Z5/Z9 conditioning), `zone_outputs/<ZONE>/` (16 additional zones) | Bundled — 6-dim/9-dim conditioning, the representation the reported Transformer results actually used; frozen provenance, not the package |
| Wind raw data | `data/raw/wind/` | **Empty — placeholder README only.** No wind dataset exists in this repository. See "Wind Data" below. |

## Source

- **Load data:** PJM Interconnection hourly metered load (`hrl_load_metered`), the
  standard PJM operational dataset format. Exact acquisition method (manual download vs.
  API) is **not documented in this repository as of this audit.**
- **Temperature data:** NOAA Integrated Surface Database (ISD), fetched per airport
  weather station and cached locally (`src/enerdatasynth/resources/pjm_load/data/temperature.py`).

## Time Coverage

2016–2025 (10 years), per the raw CSV filenames and `data/cache/pjm_load/` station-year files.

## Geographic Coverage

- **Researched (Phase 2):** 17 PJM zones — AP, DAY, PL, DEOK, CE, PN, DPL, BC, PS, RECO,
  JC, ME, DUQ, PE, AE, EKPC, DOM.
- **Packaged (installed SDK today):** zone `PS` only.

## Temporal Resolution

Hourly source data, pivoted to fixed 24-column daily profiles (one row per calendar day).

## Variables

- Hourly metered load (MW, later normalized)
- Hourly temperature per zone (equal-weighted average of that zone's constituent
  stations — see `ZONE_STATIONS` in `src/enerdatasynth/resources/pjm_load/data/temperature.py`)
- Derived: daily `Tmax`, `Tmin`, `spread = Tmax − Tmin`
- Calendar: month, day-of-week, US federal holiday flag (via the `holidays` package,
  inferred from `pyproject.toml` dependencies — exact holiday calendar/region not
  independently re-verified in this audit)

## PJM Zones

See "Geographic Coverage" above. Zone-to-station weather mapping is defined in
`ZONE_STATIONS` (`src/enerdatasynth/resources/pjm_load/data/temperature.py`); an `"RTO"` catch-all entry
(equal weight across every known station) also exists there.

## Temperature Data

NOAA ISD, resolved by USAF-WBAN station ID with automatic re-resolution across years
(stations get renumbered). On a 404 for a given station-year, the code falls back to
alternate USAF-WBAN combinations for that station code. See
`src/enerdatasynth/resources/pjm_load/data/temperature.py`'s module docstring for the full mechanism.

## Temperature-to-Zone Mapping

Equal-weighted average across each zone's constituent stations (`ZONE_STATIONS` dict in
`temperature.py`). Not distance-weighted, population-weighted, or otherwise adjusted.

## Preprocessing

Implemented in `src/enerdatasynth/resources/pjm_load/data/preprocess.py`:
1. Sort by timestamp, pivot to one row per day / 24 hour-columns.
2. Detect and flag DST days (23 or 25 hourly rows for that date).
3. Merge zone temperature, flag missing `Tmax`/`Tmin`.
4. Flag a COVID period (`is_covid`) — exact date range **not documented in this repository
   as of this audit.**
5. Remove any day flagged DST, missing-load, missing-weather, or COVID.
6. Normalize: load per-hour to `[0,1]` (MinMax), temperature via z-score, spread via
   MinMax (based on `week3.py`'s Step 7 description — **not independently re-verified
   against the current `preprocess.py` in this audit pass; treat as the historical
   procedure unless confirmed identical**).

## Daylight Saving Time Handling

Days with a non-24 hourly row count (23 or 25) are detected pre-pivot
(`detect_dst_days`) and excluded — a fixed 24-column daily representation cannot
represent them. See `docs/research_status.md` §B.

## Missing Data

- **Missing load:** any NaN among the 24 hourly load columns for a day (excluding days
  already flagged DST, to avoid double-counting) → excluded.
- **Missing weather:** NaN `Tmax` or `Tmin` after the zone temperature merge → excluded.
- No imputation is performed for either case — affected days are dropped, not filled.

## Weather Availability

Per-station-year caching (`data/cache/pjm_load/`) means weather availability is bounded by NOAA
ISD's own station coverage and uptime; a station with sparse ISD history will produce
sparser cached years. Not independently audited per-zone in this pass.

## Conditioning Variables

See `docs/research_status.md` §C for the full table and the critical distinction between
the research-history conditioning (5 conceptual fields / 6 raw columns, "Z5") and the
current package's 23-dimensional one-hot representation.

## Train/Validation Split

Chronological: **2016–2023 train / 2024–2025 validation** (`week3.py`, Step 6) — not a
random split. The current package's bundled `PS`-zone processed dataset (2,785
train / 600 validation days) is consistent with this chronological convention but was
not independently re-derived in this audit.

## Known Limitations

- Single-station-equal-weight temperature mapping may not reflect each zone's true load-
  weighted climate exposure.
- COVID-period exclusion removes a real, historically significant demand anomaly rather
  than modeling it — appropriate for this project's stated goal (typical-day generation)
  but worth knowing if using this data for anything COVID-adjacent.
- DST-day exclusion means the model has never seen (and cannot generate) the two
  irregular-length calendar days per year.
- No imputation — missing-data days are simply absent, not a statistically neutral gap.

## Intended Uses

Training and evaluating conditional generative models of PJM daily load shape, as
implemented in this repository.

## Out-of-Scope Uses

- Forecasting actual future load (see `docs/research_status.md` §A — this is explicitly
  not a forecasting dataset/model pairing).
- Any operational grid-planning, dispatch, or control decision.
- Zones or time periods outside 2016–2025 / the 17 researched zones.

## Data Quality

Not independently re-scored in this audit beyond the cleaning steps already described
(DST/missing-load/missing-weather/COVID exclusion). No outlier-detection pass beyond
that was found in the current package code.

## Wind Data

Unlike everything above (all PJM load), the wind resource has **no raw dataset in this
repository at all**, bundled or otherwise — this section documents that gap rather than
describing a dataset, and is the wind-side parallel to the "Composition"/"Source"
sections above. Full narrative: `docs/models/wind.md` §2.

- **No raw wind training data exists in this repository or its dependencies.**
  `data/raw/wind/` contains only a placeholder README.
- **The bundled checkpoints were trained elsewhere.** `data/checkpoints/wind/{1ch,2ch,3ch}/`
  ships real, working, pretrained weights — generation works today — but this
  repository cannot reproduce the preprocessing or training that produced them, because
  the source data was never vendored in.
- **A likely original source is inferable but not confirmed.** `h5py`, `h5pyd`, and
  `python-dotenv` are declared package dependencies (suggesting an HSDS/NREL WIND
  Toolkit-style original data source) but have zero actual usage anywhere in
  `src/enerdatasynth/resources/wind/` — the acquisition step itself was never vendored,
  so this is a plausible provenance clue, not a documented source.
- **What the trained checkpoints imply about the source data:** 5-minute-resolution
  `windspeed_100m` (and, for `2ch`/`3ch`, `winddirection_100m`) sequences, windowed to
  288 steps (24 hours) — see `docs/models/wind.md` §2–3 for the full representation.
- **No real-vs-generated comparison can be produced by this repository** — there is
  nothing in it to validate a generated wind profile against. `docs/models/wind.md` §9
  now shows the upstream `wind_diffusion` project's own real-vs-synthetic envelope
  plots (borrowed, clearly attributed, not reproduced here) as the closest available
  evidence.

## Reproducibility

`enerdatasynth preprocess` reproduces the current package's `PS`-zone processed dataset
from raw CSVs, **which you must supply yourself** — see
`data/raw/README.md` for exactly which files and where to
get them. `pip install enerdatasynth` does not ship raw data or trained checkpoints, by
design (`docs/packaging.md`), and — as of this update — **this repository no longer
bundles raw or cached source data either**, for the same reason.

## Licensing / Redistribution

**No raw PJM load data or NOAA ISD-derived temperature cache is distributed in this
repository.** Both were removed (previously bundled under
`data/raw/pjm_load/` and `data/cache/pjm_load/`) specifically so this project isn't
redistributing third-party-sourced data whose terms of use were never independently
verified. What remains:

- Placeholder files at both locations point to the actual public sources (PJM Data
  Miner 2, NOAA ISD) so anyone reproducing this pipeline fetches the data themselves,
  under whatever terms those sources publish it.
- `data/processed/pjm_load/` (small, derived tensors + scaler metadata) is still bundled — this
  is the package's own cleaned/normalized representation, not a redistribution of the
  raw source files, and is required for `enerdatasynth.model("pjm_load", ...)` to
  construct at all.
- The software license (`LICENSE`, MIT) covers the code in this repository only. It does
  not and cannot grant redistribution rights to third-party datasets — moot now that
  none are bundled, but the distinction still matters for `data/processed/pjm_load/`'s status if
  that's ever questioned.

This closes the redistribution-rights blocker previously flagged in
`docs/release_audit.md` for the raw/cached directories specifically.
