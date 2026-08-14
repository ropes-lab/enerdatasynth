# `data/cache/`

Regenerable fetch caches, one folder per resource:

- [`pjm_load/`](pjm_load/README.md) — NOAA temperature fetch cache.
- [`wind/`](wind/README.md) — placeholder; wind has no fetch-and-cache step in this package.

Nothing here is source data — both are safe to delete; they rebuild automatically where
a fetch pipeline exists (`pjm_load/`) or simply stay empty (`wind/`).
