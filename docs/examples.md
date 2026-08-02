# Examples

All runnable from the project root after `pip install -e .` (or a real install — see [`docs/installation.md`](installation.md)).

## `examples/generate_samples.py`

Loads a trained checkpoint via `LoadDiffusionModel.from_run_id(...)` and generates synthetic profiles conditioned on the real validation set's condition vectors. Demonstrates the lower-level API (raw `z` arrays) — see [`docs/api_reference.md`](api_reference.md) for how this relates to the fluent `enerdatasynth.model(...)` interface.

```bash
python examples/generate_samples.py
```

## `examples/run_full_pipeline.py`

A fast, small walkthrough of every pipeline stage — preprocess (if needed) → train a tiny model → generate → validate — in a couple of minutes rather than a full research-grade run. Not the production entry point (that's `examples/run_all.py` / `enerdatasynth pipeline`); this is for seeing every stage connect quickly.

```bash
python examples/run_full_pipeline.py
```

## `examples/exploratory_data_analysis.py`

A small demonstration of inspecting the processed dataset (`data/processed/`): mean daily load shape, per-hour spread, train/val split sizes. Intentionally lightweight — the original, more exhaustive EDA (zone correlations, load-duration curves, COVID-period visualization) is preserved as historical research in `data/research_history/scripts/eda_exploration.py`.

```bash
python examples/exploratory_data_analysis.py
```

## The fluent SDK (not yet a standalone example script, shown inline)

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

## `examples/run_benchmark.py`

Reproduces the classical-vs-diffusion comparison table shown in the README, built entirely from library calls — see [`docs/evaluation.md`](evaluation.md).

```bash
python examples/run_benchmark.py --run-id transformer_T1000_H64_L2
python examples/run_benchmark.py --run-id transformer_T1000_H64_L2 --n-days 50   # faster, subsampled
```
