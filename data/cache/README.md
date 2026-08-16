# `data/cache/`

Regenerable fetch caches, one folder per resource:

- [`pjm_load/`](pjm_load/README.md) — NOAA temperature fetch cache.
- [`wind/`](wind/README.md) — placeholder; wind has no fetch-and-cache step in this package.

Nothing here is source data. Both are safe to delete — `pjm_load/` rebuilds automatically since a real fetch pipeline exists for it, and `wind/` just stays empty.
