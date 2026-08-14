# `examples/`

Runnable demonstrations of the package's public API, in increasing order of scope. Each
script's own header explains exactly what it does — this file is the map.

| Script | What it demonstrates | Runtime |
|---|---|---|
| `exploratory_data_analysis.py` | Inspecting the processed dataset (`metadata.json`, mean daily shape) — a lightweight companion to the full historical EDA in `data/research_history/scripts/eda_exploration.py` | Seconds |
| `generate_samples.py` | Loading a trained checkpoint via `LoadDiffusionModel` and generating profiles — the minimal Tier-2 usage pattern | Seconds |
| `run_full_pipeline.py` | End-to-end walkthrough (preprocess → train one small model → generate → validate) in a couple of minutes, not the full sweep | ~Minutes |
| `run_all.py` | The real top-level orchestrator: preprocess → full hyperparameter sweep → generate for the best config → validation plots → failure analysis → results summary | Hours (full sweep) |
| `run_benchmark.py` | Classical-vs-diffusion comparison (AR/ARMA/ARIMA/SARIMA vs. a trained checkpoint), built entirely from library code — no duplicated logic | Depends on data size |

**Naming note found during this audit:** `run_benchmark.py`'s own module docstring
calls itself `benchmarks/run_benchmark.py` and describes itself as "the 'move to
benchmarks/' successor" to the research-history benchmark scripts — but there is no
top-level `benchmarks/` directory in this repository; the file lives here in
`examples/`. Not moved during this audit (moving it would change every path any external
doc/script references); flagged in `docs/release_audit.md` as a low-priority structural
inconsistency to resolve deliberately, not accidentally.

## What a researcher should look at first

`generate_samples.py` if you just want to see output; `run_full_pipeline.py` if you want
to see the whole pipeline connect without waiting hours.

## Assumptions these make

All of these assume raw data is already present (`data/raw/`) or that
preprocessing has already been run — none of them ship or download data themselves.
