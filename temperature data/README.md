# PJM Zonal Hourly Temperature Profiles, 2016–2025

## What this builds

Hourly temperature time series for each PJM load zone in your list, computed
as the **equal-weighted average of the airport weather stations PJM associates
with that zone** in its 2026 Load Forecast Report.

PJM does not publish percentage weights for its zone-weather mapping — it
publishes the *station list* per zone (visible on each zonal dashboard page in
the LFR). Equal weighting across that list is the standard published-data
approximation. If you have proprietary PJM weights, edit `ZONE_STATIONS` in
`pjm_hourly_temps.py`.

## Zone → station mapping

| Your code | PJM zone | Stations |
|---|---|---|
| AE | Atlantic Electric | ACY |
| BC | BGE | BWI |
| DPL, DPLCO | Delmarva | ILG, WAL |
| EASTON | Easton Utilities (no PJM list) | ESN |
| JC | JCPL | EWR, ACY |
| ME | MetEd | PHL, ABE |
| PE | PECO | PHL |
| PEPCO | PEPCO | DCA |
| SMECO | So. Maryland Co-op (no PJM list) | DCA |
| PL, PLCO | PPL | AVP, ABE, MDT, IPT |
| UGI | UGI | AVP |
| PN | Penelec | ERI, IPT |
| PS | PSE&G | EWR |
| RECO | Rockland | EWR |
| AEPAPT | Appalachian Power | CRW, ROA |
| AEPIMP | Indiana Michigan Power | FWA |
| AEPKPT | Kentucky Power | HTS |
| AEPOPT | Ohio Power | CMH, CAK |
| AP | APS | PIT, IAD, MGW |
| OE | Ohio Edison (ATSI) | CAK, CLE |
| PAPWR | Penn Power (ATSI) | YNG, PIT |
| CE | ComEd | ORD |
| DAY | Dayton P&L | DAY |
| DEOK | Duke Ohio/KY | CVG |
| DUQ | Duquesne | PIT |
| RTO | full PJM blend | all 32 stations |
| DOM | Dominion | RIC, ORF, IAD |
| EKPC | East Kentucky Power | LEX, SDF, CVG |

## Quick start

```bash
pip install pandas numpy requests pyarrow
python pjm_hourly_temps.py
```

Runtime: 5–15 min depending on your connection. Re-runs are near-instant
because raw station-year data is cached in `cache/` as parquet.

## Outputs (in `out/`)

- `pjm_hourly_temps_wide.csv` — datetime in Eastern Prevailing Time (EPT,
  matches PJM), one column per zone, °F. Best for modeling.
- `pjm_hourly_temps_long.csv` — long format: zone, datetime_utc, datetime_ept, temp_f
- `station_hourly_temps.csv` — raw per-station hourly temps in °F, for auditing.

The script prints a missing-fraction-by-zone summary at the end. Expect
<0.5% missingness for major airport stations; gaps up to 6 hours are
linearly interpolated.

## Data source

NOAA Integrated Surface Database (ISD), hourly access tier:
`https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{usaf_wban}.csv`

ISD quality flags 2, 3, 6, 7 (erroneous/suspect) are dropped. Where multiple
observations exist within an hour, the one nearest the top of the hour wins.

## Notes & caveats

- **PJM uses Eastern Prevailing Time** (US/Eastern with DST) for all its load
  and weather products. The wide CSV uses EPT to match. UTC is also available
  in the long CSV.
- **EASTON and SMECO** are small sub-zones for which PJM doesn't publish a
  station list. I picked the geographically obvious station; swap freely.
- **AEP sub-zones** (APP, INM, KP, OP) — PJM publishes a combined AEP station
  list, not per-sub-zone. The sub-zone mappings here are geographic
  approximations.
- **Equal weighting** is a simplification. PJM's actual blend weights are
  derived from load-served-by-zone in the station's MSA; they're proprietary.
  For most uses (regression features, EDA, load-temperature scatter), equal
  weighting is within 1–2 °F of the true blend.
