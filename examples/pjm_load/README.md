# `examples/pjm_load/`

Runnable demonstrations of the PJM load pipeline, smallest first. Each script's docstring ends with a **Reproduce** block giving the exact commands.

The repository ships no real data. Every script below except where noted reads the processed arrays, so first get the PJM and NOAA source files yourself (sources in [`data/README.md`](../../data/README.md#data-sources)), put them in `data/raw/pjm_load/`, and run `enerdatasynth preprocess`. Generating from the shipped checkpoint with `enerdatasynth.model("pjm_load")` needs none of this.

| Script | What it does | Output | Runtime |
|---|---|---|---|
| `exploratory_data_analysis.py` | Inspects the processed dataset: mean daily shape, per-hour spread, split sizes | printed | Seconds |
| `generate_samples.py` | Loads the trained checkpoint through `LoadDiffusionModel` and generates profiles for the real validation conditions | printed | Seconds |
| `plot_real_vs_generated.py` | Six validation days chosen by a fixed rule on `Tmax`, real profile next to a generated one for the same conditions | `docs/assets/pjm_load_real_vs_generated.png` | Under a minute |
| `compare_checkpoints.py` | Compares trained checkpoints over several seeds on the validation set; the evidence for the shipped default | `data/reports/pjm_load/checkpoint_comparison.csv` | A few minutes |
| `run_benchmark.py` | AR, ARMA, ARIMA and SARIMA against a trained checkpoint, using library code only | `data/reports/pjm_load/benchmark_comparison.csv` | Depends on data size |
| `run_full_pipeline.py` | A quick end-to-end run: preprocess, train one small model, generate, validate | a small run under `data/checkpoints/pjm_load/` | Minutes |
| `run_all.py` | The full orchestrator: preprocess, sweep, generate, plots, failure analysis, results summary | `data/processed/`, `data/checkpoints/`, `data/reports/` | Hours (full sweep) |

Start with `generate_samples.py` to see output, or `run_full_pipeline.py` to watch the whole pipeline connect without waiting hours. Wind's equivalent folder is [`../wind/`](../wind/README.md).
