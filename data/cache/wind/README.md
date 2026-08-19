# `data/cache/wind/`

Empty, and has no reason to be populated today. [`../pjm_load/`](../pjm_load/README.md)'s cache exists because `resources/pjm_load/data/temperature.py` fetches NOAA ISD data on demand and caches it station-year by station-year to avoid re-downloading; wind has no equivalent fetch step at all (see [`../raw/wind/README.md`](../../raw/wind/README.md)) — there's nothing to cache the result of.

If the original NREL WIND Toolkit / HSDS acquisition step (implied by the unused `h5pyd` dependency — see `../raw/wind/README.md`) is ever vendored into this package, a cache layer for it would plausibly live here, mirroring `pjm_load`'s pattern: keyed by whatever the source's natural unit is (station-year for NOAA; presumably grid-point/year for WIND Toolkit), safe to delete and rebuild given network access.

Exists today purely as the resource-nested sibling of `../pjm_load/`, for structural consistency.
