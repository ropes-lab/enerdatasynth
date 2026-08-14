# `docs/assets/`

Polished, README/report-quality figures — distinct from `data/reports/pjm_load/figures/` and
`data/research_history/pjm_load/reports/` (raw evaluation-run output, not curated for display).

- **`pjm_load_real_vs_generated.png`** — six representative validation days, real vs.
  generated, used in the [root README](../../README.md) and
  [`docs/models/pjm_load.md`](../models/pjm_load.md) §9. Generated directly from real
  validation data and the package's actual current default checkpoint (not a fabricated
  or cherry-picked figure) — see that model report section for the generating script's
  intent and an honest read of what it shows.

No wind equivalent exists yet — there's no real wind data anywhere in this repository to
generate a comparable figure from (see `docs/models/wind.md` §9). Add one here, using
the same methodology, once that changes.

## What should not be placed here

Raw/unlabeled evaluation output — figures here should be captioned, sourced, and
referenced from at least one doc page, not just dropped in for later use.
