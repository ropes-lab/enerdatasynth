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
| `plot_all_zones.py` | One generated July weekday for each of the 17 zone models, in a grid. Needs no data | `docs/assets/pjm_load_all_zones_generated.png` | About a minute |
| `plot_real_vs_generated_all_zones.py` | One validation day per zone (the day nearest the zone's median `Tmax`), real profile next to a generated one for the same conditions, all 17 zones | `docs/assets/pjm_load_real_vs_generated_all_zones.png` | Needs the preprocessed arrays for every zone; a few minutes |
| `train_all_zones.py` | Runs `preprocess`, `train` and `calibrate` for each of the 17 PJM zones with the shipped Transformer settings, in a scratch folder; this is how the bundled zone models were built | `zones/<ZONE>/` folders in the scratch folder | About 5 minutes per zone |
| `evaluate_all_zones.py` | Scores every zone's model on its validation days (7 metrics, with and without calibration) | `data/reports/pjm_load/zone_results.csv` | A few minutes |
| `compare_diffusion_zones.py` | Transformer against an MLP denoiser given the same training and a small per-zone tuning, on all 17 zones, 10 metrics | `data/reports/pjm_load/diffusion_zone_comparison.csv` | About 2 hours |
| `run_full_pipeline.py` | A quick end-to-end run: preprocess, train one small model, generate, validate | a small run under `data/checkpoints/pjm_load/` | Minutes |
| `run_all.py` | The full orchestrator: preprocess, sweep, generate, plots, failure analysis, results summary | `data/processed/`, `data/checkpoints/`, `data/reports/` | Hours (full sweep) |

Both `*_all_zones.py` scripts need your own copy of the raw files; point `--raw-dir` at them. Start with `generate_samples.py` to see output, or `run_full_pipeline.py` to watch the whole pipeline connect without waiting hours. Wind's equivalent folder is [`../wind/`](../wind/README.md).
