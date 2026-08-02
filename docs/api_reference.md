# API Reference

## `enerdatasynth.model(name, zone=None) -> ResourceModel`

Construct a ready-to-configure generator.

```python
load = enerdatasynth.model("pjm_load", zone="PS")
```

- `name` — resource type. Currently only `"pjm_load"` exists.
- `zone` — defaults to the first zone in the registry for `name` (currently `"PS"`, the only one). Raises `ValueError` for any unsupported zone.

Raises `ValueError` for an unknown `name`. Raises `FileNotFoundError` if `data/processed/metadata.json` doesn't exist yet (run `enerdatasynth preprocess`) or if no trained checkpoint exists yet (run `enerdatasynth train`).

Internally, `model()` auto-discovers the best available checkpoint: it prefers any `architecture="transformer"` run (lowest validation loss among those — the validated-best architecture), falling back to the best `"mlp"` run if no transformer checkpoint has been trained yet against the current preprocessing pipeline. This resolution is automatic and requires no code changes when a better checkpoint becomes available later.

## `ResourceModel.set(**kwargs) -> ResourceModel`

Set conditioning variables. Chainable (`load.set(month=7).set(Tmax=86, Tmin=70)` — returns `self`); only the keys you pass are updated, so you can build up conditions incrementally.

| Parameter | Type | Required | Default | Notes |
|---|---|---|---|---|
| `month` | `int` (1–12) | **Yes** | — | No sensible default exists; must be set before `.sample()`. |
| `Tmax` | `float` | **Yes** | — | Degrees **Fahrenheit** — matches the training data's units. Not silently converted from Celsius; see "Why Fahrenheit?" below. |
| `Tmin` | `float` | **Yes** | — | Same units as `Tmax`. |
| `weekday` | `bool` | No | `True` | See "Weekday semantics" below. |
| `day_of_week` | `int` (0=Mon..6=Sun) or `str` (`"Monday"`, `"Mon"`, ...) | No | — | Exact override; takes precedence over `weekday` when both are set. |
| `holiday` | `bool` | No | `False` | US federal holiday flag. |
| `spread` | `float` | No | training-set mean | See "Why does `spread` default to the mean?" below. |

Calling `.sample()` before setting `month`, `Tmax`, and `Tmin` raises `ValueError` listing exactly which fields are missing.

### Why Fahrenheit?

The training data's temperature pipeline (`src/enerdatasynth/data/temperature.py`) converts NOAA observations to °F. `.set(Tmax=..., Tmin=...)` uses the same units deliberately — silently guessing a conversion (e.g. assuming Celsius input) risks generating a profile for a physically different day than intended, with no error to catch the mistake. State units explicitly rather than guess.

### Weekday semantics

The model's actual conditioning is a *specific* day of week (one-hot, 7 classes), not a "weekday" category. There's no single correct choice for "an unspecified weekday" — Monday and Friday can have meaningfully different load shapes. `.set(weekday=True)` therefore samples a uniformly random weekday (Mon–Fri) **independently for each generated profile** rather than arbitrarily fixing one day for every sample; `weekday=False` samples uniformly from Sat/Sun. If you need one exact day, use `day_of_week`.

### Why does `spread` default to the mean?

`spread` (daily load range, max MW − min MW) was part of the conditioning the model was trained with, but it's a property of the load curve being generated — not something a caller predicting from weather conditions alone would actually know in advance. There is no validated model here for estimating spread from `Tmax`/`Tmin`. Defaulting to the training-set mean (z-scored value of 0.0) represents "a typical-variability day" — a documented simplification, not an estimate to be trusted for spread-sensitive analysis. Pass `spread=` explicitly if you have a real value in mind.

## `ResourceModel.sample(n=1, seed=None, units="MW") -> np.ndarray`

Generate `n` synthetic 24-hour profiles under the currently-set conditions.

- `n` — number of profiles to generate.
- `seed` — if given, reseeds every RNG (`random`, `numpy`, `torch`) immediately before sampling, making the result reproducible. If omitted, uses whatever global RNG state the process is already in.
- `units` — `"MW"` (default): denormalized to real load units using the scaler fit during preprocessing (`data/processed/metadata.json`'s `load_scaler`). `"normalized"`: raw `[0, 1]`-ish model output. Any other value raises `ValueError`.

Returns an `(n, 24)` `numpy.ndarray`.

## `enerdatasynth.generators.BaseGenerator`

The common interface `LoadDiffusionModel` implements, and any future resource-type generator (e.g. wind) would implement too. See [`docs/extending.md`](extending.md).

```python
class BaseGenerator(ABC):
    domain: str
    def sample(self, z, seed=None, **kwargs) -> np.ndarray: ...
    def count_parameters(self) -> int: ...  # optional
```

## Lower-level API (research/development use)

For training new checkpoints, computing raw evaluation metrics, or working directly with conditioning vectors instead of `.set()`'s human-readable interface, the top-level package also re-exports:

```python
from enerdatasynth import RunConfig, train_one_config, LoadDiffusionModel, compute_metrics, calibrate_checkpoint
```

`LoadDiffusionModel` (in `enerdatasynth.generators`) is the class `ResourceModel` wraps internally — it takes a raw `(N, cond_dim)` array via `.sample(z, seed=...)` rather than `.set()`'s human-readable conditions, and exposes `.from_run_id(run_id)`/`.from checkpoint_path` construction. Use it directly if you need conditioning vectors your own preprocessing already produces, rather than `.set()`'s calendar/weather interface. See [`docs/training.md`](training.md) for the full research/development workflow.
