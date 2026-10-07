# `docs/assets/`

Polished, README/report-quality figures — different from `data/reports/pjm_load/figures/` and `data/research_history/pjm_load/reports/`, which hold raw evaluation-run output that hasn't been curated for display.

- **`pjm_load_real_vs_generated.png`** — six representative validation days, real vs. generated, used in [`docs/models/pjm_load.md`](../models/pjm_load.md) §9. Generated from real validation data and the shipped default checkpoint by `examples/plot_real_vs_generated.py`, with a fixed seed and days chosen by a rule on `Tmax`, not fabricated or cherry-picked. See that section for an honest read of what it shows.

No wind equivalent exists yet, since there's no real wind data anywhere in this repository to generate a comparable figure from (see `docs/models/wind.md` §9). Add one here, using the same methodology, once that changes.

## What should not be placed here

Raw or unlabeled evaluation output. Figures here should be captioned, sourced, and referenced from at least one doc page, not just dropped in for later use.
