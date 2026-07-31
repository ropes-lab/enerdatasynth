"""
PJM zonal hourly temperature builder, 2016-2025.

Pulls hourly temperature observations from NOAA's Integrated Surface Database
(ISD) for the airport weather stations PJM lists in its annual Load Forecast
Report, then computes each zone's hourly temperature as the equal-weighted
average of its constituent stations.

Robustness:
- Resolves USAF-WBAN station IDs from NOAA's official station inventory at
  runtime (handles station re-numbering between years).
- On a 404 for any station-year, falls back to other USAF-WBAN combinations
  that share the same airport ICAO code (e.g., KCAK).
- Caches resolved IDs and downloaded data so re-runs are near-instant.

Source for zone -> station list:
    PJM 2026 Load Forecast Report (Jan 2026), zonal dashboard pages.

Output:
    out/pjm_hourly_temps_long.csv   columns: zone, datetime_utc, datetime_ept, temp_f
    out/pjm_hourly_temps_wide.csv   datetime_ept index, one column per zone
    out/station_hourly_temps.csv    raw per-station data (for auditing)

Usage:
    pip install pandas numpy requests tqdm pyarrow
    python pjm_hourly_temps.py
"""

from __future__ import annotations

import io
import json
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

YEARS = range(2016, 2026)  # 2016 through 2025 inclusive
OUT_DIR = Path("out")
CACHE_DIR = Path("cache")
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

ISD_URL = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{sid}.csv"
ISD_INVENTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
INVENTORY_PATH = CACHE_DIR / "isd-history.csv"
RESOLVED_IDS_PATH = CACHE_DIR / "resolved_ids.json"

# Short station codes -> ICAO codes (NOAA inventory keys on ICAO)
STATION_ICAO = {
    # East / Mid-Atlantic
    "ACY": "KACY", "ABE": "KABE", "AVP": "KAVP", "BWI": "KBWI",
    "DCA": "KDCA", "ERI": "KERI", "ESN": "KESN", "EWR": "KEWR",
    "IAD": "KIAD", "ILG": "KILG", "IPT": "KIPT", "MDT": "KMDT",
    "ORF": "KORF", "PHL": "KPHL", "PIT": "KPIT", "RIC": "KRIC",
    "WAL": "KWAL",
    # West
    "CAK": "KCAK", "CLE": "KCLE", "CMH": "KCMH", "CRW": "KCRW",
    "CVG": "KCVG", "DAY": "KDAY", "FWA": "KFWA", "LEX": "KLEX",
    "ORD": "KORD", "ROA": "KROA", "SDF": "KSDF", "TOL": "KTOL",
    "YNG": "KYNG", "MGW": "KMGW", "HTS": "KHTS",
}

# Zone -> {station: weight}. Equal-weighted across the stations PJM lists.
ZONE_STATIONS: dict[str, dict[str, float]] = {
    "AE":      {"ACY": 1.0},
    "BC":      {"BWI": 1.0},                                # BGE
    "DPL":     {"ILG": 1.0, "WAL": 1.0},
    "DPLCO":   {"ILG": 1.0, "WAL": 1.0},
    "EASTON":  {"ESN": 1.0},
    "JC":      {"EWR": 1.0, "ACY": 1.0},                    # JCPL
    "ME":      {"PHL": 1.0, "ABE": 1.0},                    # MetEd
    "PE":      {"PHL": 1.0},                                # PECO
    "PEPCO":   {"DCA": 1.0},
    "SMECO":   {"DCA": 1.0},
    "PL":      {"AVP": 1.0, "ABE": 1.0, "MDT": 1.0, "IPT": 1.0},
    "PLCO":    {"AVP": 1.0, "ABE": 1.0, "MDT": 1.0, "IPT": 1.0},
    "UGI":     {"AVP": 1.0},
    "PN":      {"ERI": 1.0, "IPT": 1.0},                    # PENLC
    "PS":      {"EWR": 1.0},
    "RECO":    {"EWR": 1.0},
    "AEPAPT":  {"CRW": 1.0, "ROA": 1.0},
    "AEPIMP":  {"FWA": 1.0},
    "AEPKPT":  {"HTS": 1.0},
    "AEPOPT":  {"CMH": 1.0, "CAK": 1.0},
    "AP":      {"PIT": 1.0, "IAD": 1.0, "MGW": 1.0},
    "OE":      {"CAK": 1.0, "CLE": 1.0},
    "PAPWR":   {"YNG": 1.0, "PIT": 1.0},
    "CE":      {"ORD": 1.0},
    "DAY":     {"DAY": 1.0},
    "DEOK":    {"CVG": 1.0},
    "DUQ":     {"PIT": 1.0},
    "DOM":     {"RIC": 1.0, "ORF": 1.0, "IAD": 1.0},
    "EKPC":    {"LEX": 1.0, "SDF": 1.0, "CVG": 1.0},
    "RTO": {sid: 1.0 for sid in STATION_ICAO},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pjm_temps")


# --------------------------------------------------------------------------- #
# NOAA ISD station inventory
# --------------------------------------------------------------------------- #

def load_inventory() -> pd.DataFrame:
    if not INVENTORY_PATH.exists():
        log.info("downloading NOAA ISD station inventory...")
        r = requests.get(ISD_INVENTORY_URL, timeout=120)
        r.raise_for_status()
        INVENTORY_PATH.write_bytes(r.content)

    inv = pd.read_csv(INVENTORY_PATH, dtype=str, na_filter=False)
    inv["USAF"] = inv["USAF"].str.strip()
    inv["WBAN"] = inv["WBAN"].str.strip()
    inv["ICAO"] = inv["ICAO"].str.strip()
    inv["SID"] = inv["USAF"].str.zfill(6) + inv["WBAN"].str.zfill(5)
    inv["BEGIN"] = pd.to_datetime(inv["BEGIN"], format="%Y%m%d", errors="coerce")
    inv["END"] = pd.to_datetime(inv["END"], format="%Y%m%d", errors="coerce")
    return inv


def candidate_ids(inv: pd.DataFrame, icao: str, year: int) -> list[str]:
    """All USAF-WBAN combos for an ICAO that were active during `year`.

    Sorted with US primary ASOS (USAF starts with '72', valid WBAN) first.
    """
    yr_start = pd.Timestamp(f"{year}-01-01")
    yr_end = pd.Timestamp(f"{year}-12-31")
    hits = inv[
        (inv["ICAO"] == icao)
        & (inv["BEGIN"] <= yr_end)
        & (inv["END"] >= yr_start)
    ].copy()
    if hits.empty:
        return []
    # Prefer USAF starting with '72' and WBAN != '99999' (placeholder)
    hits["pref"] = (
        hits["USAF"].str.startswith("72").map({True: 0, False: 1})
        + (hits["WBAN"] == "99999").map({True: 1, False: 0})
    )
    hits = hits.sort_values(["pref", "USAF", "WBAN"])
    return hits["SID"].tolist()


# --------------------------------------------------------------------------- #
# Resolved-ID cache
# --------------------------------------------------------------------------- #

def load_resolved_ids() -> dict[str, str]:
    if RESOLVED_IDS_PATH.exists():
        return json.loads(RESOLVED_IDS_PATH.read_text())
    return {}


def save_resolved_ids(d: dict[str, str]) -> None:
    RESOLVED_IDS_PATH.write_text(json.dumps(d, indent=2, sort_keys=True))


# --------------------------------------------------------------------------- #
# Download & parse one station-year
# --------------------------------------------------------------------------- #

def _cache_path(code: str, year: int) -> Path:
    return CACHE_DIR / f"{code}_{year}.parquet"


def _parse_isd_csv(content: bytes) -> pd.DataFrame:
    raw = pd.read_csv(
        io.BytesIO(content),
        usecols=["DATE", "TMP"],
        dtype={"TMP": "string"},
    )
    raw["datetime_utc"] = pd.to_datetime(raw["DATE"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["datetime_utc"])

    parts = raw["TMP"].str.split(",", n=1, expand=True)
    val = pd.to_numeric(parts[0], errors="coerce")
    qual = parts[1] if 1 in parts.columns else pd.Series(["1"] * len(raw))
    val = val.where(~qual.isin(["2", "3", "6", "7"]), np.nan)
    val = val.where(val != 9999, np.nan)
    raw["temp_c"] = val / 10.0

    raw["hour_utc"] = raw["datetime_utc"].dt.floor("h")
    raw["dist"] = (raw["datetime_utc"] - raw["hour_utc"]).dt.total_seconds().abs()
    raw = raw.sort_values(["hour_utc", "dist"]).drop_duplicates("hour_utc", keep="first")
    return (
        raw[["hour_utc", "temp_c"]]
        .rename(columns={"hour_utc": "datetime_utc"})
        .reset_index(drop=True)
    )


def fetch_station_year(
    code: str,
    year: int,
    inv: pd.DataFrame,
    resolved: dict[str, str],
    retries: int = 3,
) -> pd.DataFrame:
    cache = _cache_path(code, year)
    if cache.exists():
        return pd.read_parquet(cache)

    icao = STATION_ICAO[code]
    key = f"{code}_{year}"

    candidates = candidate_ids(inv, icao, year)
    primary = resolved.get(key)
    if primary:
        candidates = [primary] + [c for c in candidates if c != primary]
    if not candidates:
        raise RuntimeError(f"no ISD candidates for {code} ({icao}) in {year}")

    last_err: Exception | None = None
    for sid in candidates:
        url = ISD_URL.format(year=year, sid=sid)
        for attempt in range(1, retries + 1):
            try:
                r = requests.get(url, timeout=120)
                if r.status_code == 404:
                    log.warning("  %s %s: ID %s not in archive, trying next candidate", code, year, sid)
                    break
                r.raise_for_status()
                df = _parse_isd_csv(r.content)
                if df.empty or df["temp_c"].notna().sum() < 100:
                    log.warning("  %s %s: ID %s returned <100 valid obs, trying next candidate", code, year, sid)
                    break
                df.to_parquet(cache, index=False)
                resolved[key] = sid
                save_resolved_ids(resolved)
                return df
            except requests.HTTPError as e:
                last_err = e
                log.warning("  %s %s HTTP error (try %d): %s", code, year, attempt, e)
                if attempt < retries:
                    time.sleep(2 * attempt)
            except requests.RequestException as e:
                last_err = e
                log.warning("  %s %s network error (try %d): %s", code, year, attempt, e)
                if attempt < retries:
                    time.sleep(2 * attempt)

    raise RuntimeError(
        f"all {len(candidates)} ISD candidates failed for {code} ({icao}) in {year}; "
        f"last error: {last_err}"
    )


# --------------------------------------------------------------------------- #
# Build station panel
# --------------------------------------------------------------------------- #

def build_station_panel(codes: list[str], inv: pd.DataFrame, resolved: dict[str, str]) -> pd.DataFrame:
    frames: dict[str, pd.Series] = {}
    failed: list[str] = []
    total = len(codes) * len(list(YEARS))
    cached = sum(_cache_path(c, y).exists() for c in codes for y in YEARS)
    log.info("station-years: %d total, %d cached, %d to download", total, cached, total - cached)

    with tqdm(total=total, desc="station-years", unit="file", smoothing=0.1) as pbar:
        for code in codes:
            chunks = []
            for y in YEARS:
                is_cached = _cache_path(code, y).exists()
                pbar.set_postfix_str(f"{code} {y}{' (cached)' if is_cached else ''}")
                try:
                    chunks.append(fetch_station_year(code, y, inv, resolved))
                except Exception as e:
                    log.error("  %s %s permanently failed: %s", code, y, e)
                    failed.append(f"{code} {y}")
                pbar.update(1)
            if chunks:
                s = pd.concat(chunks, ignore_index=True).set_index("datetime_utc")["temp_c"]
                s = s[~s.index.duplicated(keep="first")].sort_index()
                frames[code] = s

    if failed:
        log.warning("permanently failed station-years (%d): %s", len(failed), ", ".join(failed))

    panel = pd.DataFrame(frames)
    full_index = pd.date_range(
        f"{min(YEARS)}-01-01 00:00",
        f"{max(YEARS)}-12-31 23:00",
        freq="h",
        tz="UTC",
    )
    panel = panel.reindex(full_index)
    panel.index.name = "datetime_utc"
    # Coerce every column to float64 — parquet round-trip can produce nullable
    # dtypes (Float64/Int64) that don't play with np.isnan downstream.
    panel = panel.apply(pd.to_numeric, errors="coerce").astype("float64")
    panel = panel.interpolate(method="time", limit=6, limit_direction="both")
    return panel


# --------------------------------------------------------------------------- #
# Zone weighting
# --------------------------------------------------------------------------- #

def build_zone_panel(station_panel: pd.DataFrame) -> pd.DataFrame:
    zones = {}
    for zone, weights in ZONE_STATIONS.items():
        cols = [c for c in weights if c in station_panel.columns]
        if not cols:
            log.warning("zone %s: no available stations, skipping", zone)
            continue
        w = np.array([weights[c] for c in cols], dtype=np.float64)
        w = w / w.sum()
        # Coerce to clean float64 (pd.to_numeric handles object dtype, Float64,
        # mixed strings, pd.NA, etc. uniformly)
        sub_df = station_panel[cols].apply(pd.to_numeric, errors="coerce").astype("float64")
        sub = sub_df.to_numpy()
        mask = ~np.isnan(sub)
        weighted = np.where(mask, sub * w, 0.0).sum(axis=1)
        weight_sum = np.where(mask, w, 0.0).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            zones[zone] = np.where(weight_sum > 0, weighted / weight_sum, np.nan)
    return pd.DataFrame(zones, index=station_panel.index)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    inv = load_inventory()
    resolved = load_resolved_ids()

    needed = sorted({s for w in ZONE_STATIONS.values() for s in w})
    log.info("building panel for %d stations across %d years", len(needed), len(list(YEARS)))

    station_panel_c = build_station_panel(needed, inv, resolved)
    station_panel_f = station_panel_c * 9 / 5 + 32

    zone_panel_c = build_zone_panel(station_panel_c)
    zone_panel_f = zone_panel_c * 9 / 5 + 32

    zone_panel_f_ept = zone_panel_f.copy()
    zone_panel_f_ept.index = zone_panel_f_ept.index.tz_convert("US/Eastern")
    zone_panel_f_ept.index.name = "datetime_ept"
    zone_panel_f_ept.round(2).to_csv(OUT_DIR / "pjm_hourly_temps_wide.csv")

    long_df = (
        zone_panel_f.stack()
        .rename("temp_f")
        .reset_index()
        .rename(columns={"level_1": "zone"})
    )
    long_df["datetime_ept"] = long_df["datetime_utc"].dt.tz_convert("US/Eastern")
    long_df = long_df[["zone", "datetime_utc", "datetime_ept", "temp_f"]]
    long_df["temp_f"] = long_df["temp_f"].round(2)
    long_df.to_csv(OUT_DIR / "pjm_hourly_temps_long.csv", index=False)

    station_panel_f.round(2).to_csv(OUT_DIR / "station_hourly_temps.csv")

    log.info("done. wrote %s", OUT_DIR.resolve())
    print("\nMissing-fraction by zone (%):")
    print((zone_panel_f.isna().mean() * 100).round(2).sort_values().to_string())


if __name__ == "__main__":
    main()
