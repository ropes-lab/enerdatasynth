# `data/cache/wind/`

Empty. [`../pjm_load/`](../pjm_load/README.md) has a cache because
`resources/pjm_load/data/temperature.py` downloads NOAA ISD weather data on demand and
saves it by station and year. Wind has no download step (see
[`../../raw/wind/README.md`](../../raw/wind/README.md)), so there's nothing to cache.

If the original NREL WIND Toolkit download code is ever added, its cache would go here,
following `pjm_load`'s pattern: keyed by the source's natural unit (probably grid point
and year), and safe to delete and rebuild with network access.
