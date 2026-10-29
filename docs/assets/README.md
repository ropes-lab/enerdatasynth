# `docs/assets/`

Polished, README/report-quality figures — different from `data/reports/pjm_load/figures/` and `data/research_history/pjm_load/reports/`, which hold raw evaluation-run output that hasn't been curated for display.

- **`pjm_load_real_vs_generated.png`** — six representative validation days, real vs. generated, used in [`docs/models/pjm_load.md`](../models/pjm_load.md) §9. Generated from real validation data and the shipped default checkpoint by `examples/pjm_load/plot_real_vs_generated.py`, with a fixed seed and days chosen by a rule on `Tmax`, not fabricated or cherry-picked. See that section for an honest read of what it shows.
- **`pjm_load_all_zones_generated.png`**: used in [`docs/models/pjm_load.md`](../models/pjm_load.md) §11. One generated July weekday (86/70 °F) for each of the 17 PJM zone models, mean and 5th–95th percentile of 100 days, made by `examples/pjm_load/plot_all_zones.py` with a fixed seed. Generated output only, each panel in its own zone's megawatts; it needs no data and shows that every zone has a working model. Scores against real days are in `data/reports/pjm_load/zone_results.csv`.
- **`wind_upstream_real_vs_synthetic_{low,mid,high}_envelope.png`**: used in [`docs/models/wind.md`](../models/wind.md) §9. Copied without changes from the upstream `wind_diffusion` project's `presentation_plots/real_vs_synthetic/` output, because this repository has no real wind data to make its own. The generation settings are unknown, and the shaded bands sit at the sampler's output limits, so only the mean lines are useful (§9 explains). Replace them with a figure made the same way as the load figure above once real wind data is available.

- **`wind_generated_examples.png`**: used in [`docs/models/wind.md`](../models/wind.md) §9 and [`examples/wind/README.md`](../../examples/wind/README.md). Five generated July days for each requested wind regime from the bundled `1ch` checkpoint, made by `examples/wind/plot_generated_samples.py` with a fixed seed. It shows generated output only, because there is no real wind data here to compare with. A copy is in `data/reports/wind/figures/`.

## What should not be placed here

Raw or unlabeled evaluation output. Figures here should be captioned, sourced, and referenced from at least one doc page, not just dropped in for later use.
