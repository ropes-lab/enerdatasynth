# `data/research_history/pjm_load/`

The frozen research record behind PJM load, in three folders:

- [`scripts/`](scripts/README.md): the eleven weekly research scripts, from `eda_exploration.py` through `week7_calibrate.py` and `zone_expansion.py`.
- [`reports/`](reports/README.md): the reports and figures those scripts produced (the Week 5 benchmark before and after the sampler fix, the Week 6 architecture, noise and conditional studies, Week 7 results, and the Phase 2 zone study).
- [`outputs/`](outputs/README.md): the trained weights, scalers and calibration files from those runs, in `week3_outputs/` and `zone_outputs/<ZONE>/` for all 17 zones. The datasets are not included.

The [category README](../README.md#provenance-table) has the table that maps each script to its status and to whatever replaced it. Wind has no counterpart here; see [`../wind/README.md`](../wind/README.md).
