# Examples

Examples live in `examples/`, one folder per resource: [`examples/pjm_load/`](../examples/pjm_load/README.md) and [`examples/wind/`](../examples/wind/README.md). Each script's docstring ends with a **Reproduce** block: the exact commands, what you have to provide, and where the output goes. The PJM scripts need your own copy of the source data, because this repository ships none (sources in [`data/README.md`](../data/README.md#data-sources)); the wind scripts need nothing.

## PJM load

All runnable from the project root after `pip install -e .` and `enerdatasynth preprocess` on your own data.

### `examples/pjm_load/generate_samples.py`

Loads a trained checkpoint via `LoadDiffusionModel.from_run_id(...)` and generates synthetic profiles conditioned on the real validation set's condition vectors. Demonstrates the lower-level API (raw `z` arrays) — see [`docs/api_reference.md`](api_reference.md) for how this relates to the fluent `enerdatasynth.model(...)` interface.

```bash
python examples/pjm_load/generate_samples.py
```

### `examples/pjm_load/plot_all_zones.py`

Generates one July weekday from each of the 17 zone models and draws them in a grid (`docs/assets/pjm_load_all_zones_generated.png`). It needs no data, only the installed package, so it is a quick way to see that every zone works.

```bash
python examples/pjm_load/plot_all_zones.py
```

### `examples/pjm_load/train_all_zones.py` and `evaluate_all_zones.py`

`train_all_zones.py` runs `enerdatasynth preprocess`, `train` and `calibrate` for each of the 17 PJM zones with the shipped Transformer settings (`transformer_T1000_H64_L2`), inside a scratch folder so the processed data never lands in your repository. This is how the bundled per-zone models were built. `evaluate_all_zones.py` then scores each zone's model on its validation days and writes `data/reports/pjm_load/zone_results.csv`. Both need your own copy of the raw PJM and NOAA files.

```bash
python examples/pjm_load/train_all_zones.py --raw-dir /path/to/raw --workdir /tmp/eds_zones
python examples/pjm_load/evaluate_all_zones.py --workdir /tmp/eds_zones
```

### `examples/pjm_load/compare_diffusion_zones.py`

Compares the shipped Transformer with an MLP denoiser trained with the same protocol and tuned over three configurations per zone, on all 17 PJM zones and ten metrics, with and without calibration. It needs your own raw data (`--raw-dir`) and the repository's checkpoints, and writes `data/reports/pjm_load/diffusion_zone_comparison.csv`. The result is summarized in [`docs/models/pjm_load.md`](models/pjm_load.md) §11.

```bash
python examples/pjm_load/compare_diffusion_zones.py --stage prep  --raw-dir /path/to/raw --workdir /tmp/eds_cmp
python examples/pjm_load/compare_diffusion_zones.py --stage train --raw-dir /path/to/raw --workdir /tmp/eds_cmp
python examples/pjm_load/compare_diffusion_zones.py --stage eval  --raw-dir /path/to/raw --workdir /tmp/eds_cmp
```

### `examples/pjm_load/run_full_pipeline.py`

A fast, small walkthrough of every pipeline stage — preprocess (if needed) → train a tiny model → generate → validate — in a couple of minutes rather than a full research-grade run. Not the production entry point (that's `examples/pjm_load/run_all.py` / `enerdatasynth pipeline`); this is for seeing every stage connect quickly.

```bash
python examples/pjm_load/run_full_pipeline.py
```

### `examples/pjm_load/exploratory_data_analysis.py`

A small demonstration of inspecting the processed dataset (`data/processed/pjm_load/`, built by `enerdatasynth preprocess` from your own data, since none ships): mean daily load shape, per-hour spread, train/val split sizes. Intentionally lightweight — the original, more exhaustive EDA (zone correlations, load-duration curves, COVID-period visualization) is preserved as historical research in `data/research_history/pjm_load/scripts/eda_exploration.py`.

```bash
python examples/pjm_load/exploratory_data_analysis.py
```

### The fluent SDK (not yet a standalone example script, shown inline)

```python
import enerdatasynth

load = enerdatasynth.model("pjm_load", zone="PS")
load.set(month=7, Tmax=86, Tmin=70, weekday=True, holiday=False)
samples = load.sample(n=100, seed=0)   # (100, 24), MW

# Different conditions, same object, incremental .set():
load.set(month=1, Tmax=30, Tmin=15)
winter_samples = load.sample(n=50, seed=1)

# Raw normalized output instead of MW:
normalized = load.sample(n=10, units="normalized")
```

See [`docs/api_reference.md`](api_reference.md) for the full `.set()`/`.sample()` contract, including the documented UX decisions (weekday sampling, the `spread` default) behind it.

### `examples/pjm_load/run_benchmark.py`

Reproduces the classical-vs-diffusion comparison table shown in the README, built entirely from library calls — see [`docs/evaluation.md`](evaluation.md).

```bash
python examples/pjm_load/run_benchmark.py --run-id transformer_T1000_H64_L2
python examples/pjm_load/run_benchmark.py --run-id transformer_T1000_H64_L2 --n-days 50   # faster, subsampled
```

## Wind

Both wind examples use the checkpoints that ship in the package, so they need no data. See [`examples/wind/README.md`](../examples/wind/README.md) for what isn't available yet (a real-vs-generated example, and training examples).

### `examples/wind/generate_samples.py`

Generates wind days for a preset, regime and month and prints the first as a 24-hour profile.

```bash
python examples/wind/generate_samples.py
python examples/wind/generate_samples.py --preset 3ch --regime high --month 1 --n 5
```

### `examples/wind/plot_generated_samples.py`

Five generated days for each regime from the `1ch` preset, saved as `docs/assets/wind_generated_examples.png`. Generated output only: there is no real wind data here to compare with.

```bash
python examples/wind/plot_generated_samples.py
```
