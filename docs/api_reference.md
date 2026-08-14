# API Reference

## `enerdatasynth.model(name, zone=None) -> ResourceModel`

Construct a ready-to-configure generator.

```python
load = enerdatasynth.model("pjm_load", zone="PS")
wind = enerdatasynth.model("wind", zone="1ch")
```

- `name` — resource type: `"pjm_load"` or `"wind"`. Use `enerdatasynth.list_resources()` to see everything currently registered, with each one's supported zones/presets and checkpoint availability, without reading source.
- `zone` — defaults to the first entry in the registry for `name` (`"PS"` for load; `"1ch"` for wind). Raises `ValueError` for any unsupported zone. **For wind, `zone` is repurposed as the channel preset** (`"1ch"`/`"2ch"`/`"3ch"`) — there's no geographic-zone concept the way load has `"PS"`; see [Wind](#wind) below.

Raises `ValueError` for an unknown `name`. For `"pjm_load"`: raises `FileNotFoundError` if `data/processed/pjm_load/metadata.json` doesn't exist yet (run `enerdatasynth preprocess`) or if no trained checkpoint exists yet (run `enerdatasynth train`). For `"wind"`: raises `enerdatasynth.resources.wind.exceptions.CheckpointNotFoundError` if no checkpoint is available anywhere for the requested preset (all three of `"1ch"`/`"2ch"`/`"3ch"` have one bundled in this repo today, so this only fires if you've pointed `WIND_DIFFUSION_CHECKPOINTS` somewhere that doesn't have the preset you asked for).

Internally, `model()` delegates to a per-resource construction hook (`registry.get_registry()[name].construct`) rather than one shared checkpoint-discovery mechanism — load's and wind's checkpoints live in completely different places, resolved completely differently (see [Wind](#wind)). For load specifically: it prefers any `architecture="transformer"` run (lowest validation loss among those — the validated-best architecture), falling back to the best `"mlp"` run if no transformer checkpoint has been trained yet. This resolution is automatic and requires no code changes when a better checkpoint becomes available later.

## `ResourceModel.set(**kwargs) -> ResourceModel`

Set conditioning variables. Chainable (`load.set(month=7).set(Tmax=86, Tmin=70)` — returns `self`); only the keys you pass are updated, so you can build up conditions incrementally. Accepted keys depend on which resource you constructed (`load` vs. `wind` — see [Wind](#wind) for its own table below); passing an unrecognized key raises `ValueError` immediately, listing the valid keys for that resource.

### `pjm_load` parameters

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

The training data's temperature pipeline (`src/enerdatasynth/resources/pjm_load/data/temperature.py`) converts NOAA observations to °F. `.set(Tmax=..., Tmin=...)` uses the same units deliberately — silently guessing a conversion (e.g. assuming Celsius input) risks generating a profile for a physically different day than intended, with no error to catch the mistake. State units explicitly rather than guess.

### Weekday semantics

The model's actual conditioning is a *specific* day of week (one-hot, 7 classes), not a "weekday" category. There's no single correct choice for "an unspecified weekday" — Monday and Friday can have meaningfully different load shapes. `.set(weekday=True)` therefore samples a uniformly random weekday (Mon–Fri) **independently for each generated profile** rather than arbitrarily fixing one day for every sample; `weekday=False` samples uniformly from Sat/Sun. If you need one exact day, use `day_of_week`.

### Why does `spread` default to the mean?

`spread` (daily load range, max MW − min MW) was part of the conditioning the model was trained with, but it's a property of the load curve being generated — not something a caller predicting from weather conditions alone would actually know in advance. There is no validated model here for estimating spread from `Tmax`/`Tmin`. Defaulting to the training-set mean (z-scored value of 0.0) represents "a typical-variability day" — a documented simplification, not an estimate to be trusted for spread-sensitive analysis. Pass `spread=` explicitly if you have a real value in mind.

## `ResourceModel.sample(n=1, seed=None, units=None, with_metadata=False)`

Generate `n` synthetic profiles under the currently-set conditions.

- `n` — number of profiles to generate.
- `seed` — if given, reseeds every RNG (`random`, `numpy`, `torch`) immediately before sampling, making the result reproducible. If omitted, uses whatever global RNG state the process is already in.
- `units` — defaults to the resource's first valid unit if omitted (`"MW"` for `pjm_load`, `"m/s"` for `wind`). For `pjm_load`: `"MW"` (denormalized to real load units using the scaler fit during preprocessing) or `"normalized"` (raw `[0, 1]`-ish model output). For `wind`: only `"m/s"` is supported today — see [Wind](#wind). Any other value raises `ValueError`.
- `with_metadata` — `False` (default) returns a bare array. `True` returns `(array, metadata)` instead, where `metadata` is `{"name", "zone", "n", "seed", "units", "conditions"}` (the conditions actually in effect at this call). Either way, `.last_metadata()` recovers the same dict afterward — provenance is never silently lost, just not forced into the default return shape.

Returns an `(n, 24)` `numpy.ndarray` for `pjm_load`; an `(n, 288, 2)` array (`[..., 0]` = windspeed m/s, `[..., 1]` = direction degrees) for `wind`.

## `ResourceModel.reset() -> ResourceModel`

Clears every condition set via `.set()` so far. Chainable. `.set()` mutates state on the instance and accumulates silently across calls — a real footgun for a long-lived `ResourceModel` reused across requests (a value set several calls ago can still be in effect without anything visibly wrong at the call site that forgot it). Call `.reset()` before reconfiguring from scratch instead of constructing a new `enerdatasynth.model(...)` instance just to clear state.

## `ResourceModel.last_metadata() -> dict | None`

Provenance for the most recent `.sample()` call on this instance (`None` if `.sample()` hasn't been called yet) — see `with_metadata` above.

## Wind

```python
wind = enerdatasynth.model("wind", zone="1ch")
wind.set(wind_regime="mid", month=7, temp_min=10, temp_max=22)
samples = wind.sample(n=100, seed=0)   # (100, 288, 2), m/s + degrees
```

Wraps a separately-developed diffusion pipeline (`wind_diffusion`, vendored from a Data-For-Good research project — see `src/enerdatasynth/resources/wind/`) through the same `BaseGenerator`/`ResourceModel` contract `pjm_load` uses. `zone` selects the channel preset (`"1ch"`/`"2ch"`/`"3ch"` — windspeed only, U/V vector, or speed+direction respectively; see `wind_diffusion`'s own `README.md`/`MODEL_CARD.md` for what each actually measures).

### `wind` `.set()` parameters

| Parameter | Type | Required | Default | Notes |
|---|---|---|---|---|
| `wind_regime` | `str` (`"low"`/`"mid"`/`"high"`) | No | `"mid"` | Raises `ValueError` at `.sample()` time if not one of the checkpoint's known buckets. |
| `month` | `int` (1–12) | No | `6` | wind_diffusion's own default if unset — **not** the same "must set explicitly" contract load's `month` has. |
| `day_of_week` | `int` (0=Mon..6=Sun) | No | `0` | No name-string parsing (unlike load's `day_of_week`) and no `weekday` convenience — pass the exact integer. |
| `temp_min` / `temp_max` | `float` | No | `10.0` / `22.0` | Degrees Celsius (not Fahrenheit — a real, deliberate difference from load's `Tmax`/`Tmin`, since these come from a different training pipeline with different source units). |
| `previous_day_windspeed` | `float` (m/s) | No | wind_diffusion's own neutral default | Only meaningful against a checkpoint trained with wind-persistence conditioning; none of the three bundled checkpoints were, so this is a no-op against any of them today. |

None of wind's fields are hard-required — every one has a real default from `enerdatasynth.resources.wind.evaluation.evaluate.build_condition_vector`, unlike load's `month`/`Tmax`/`Tmin`.

### Wind units: only "m/s" is supported

`wind.sample(units=...)` only accepts `"m/s"` — `"normalized"` isn't offered for any preset (`1ch`/`2ch`/`3ch` alike). `enerdatasynth.resources.wind.evaluation.evaluate.generate_multiple_for_condition` bakes denormalization into its output rather than exposing a raw-model-output option at that layer, so there's no raw value to return even for a future `"normalized"` option without deeper changes to `wind_diffusion` itself. Accepted as a real, permanent gap rather than something to fix — not prioritized.

### Checkpoints

`model("wind", zone=...)` works with no setup for all three presets: this repo bundles real trained checkpoints at `data/checkpoints/wind/{1ch,2ch,3ch}/` — `wind_diffusion`'s own measured-best recipe for each (`1ch`/`3ch`: `cond_dropout_prob=0.3`/`cond_dropout_scope="wind_regime"`/`hidden_dim=128`; `2ch`: the same, plus `joint_consistency_channels=[0,1]`/`joint_consistency_weight=1.0`). `resources/wind/manifest.json` itself still says `"pending"` for every preset — these bundled checkpoints are this repo's own addition, not something `wind_diffusion` publishes or discovers on its own.

An explicit `WIND_DIFFUSION_CHECKPOINTS` env var always takes precedence over the bundled checkpoint (see `enerdatasynth.resources.wind.registry.resolve_weights`'s own explicit-first precedence) — point it at your own trained checkpoint's directory if you have one, rather than the bundled default being silently used instead.

**Each preset's generation settings differ, and are resolved automatically.** `WindDiffusionModel` reads its own preset's `recommended_generation` from `wind_diffusion`'s bundled `manifest.json` at construction time and uses it as `.sample()`'s per-call defaults (Tier 2 only — `ResourceModel.sample()` doesn't expose these knobs at all). Concretely: `1ch`/`3ch` default to `eta=1.0, guidance_scale=40.0, guidance_scope="wind_regime"`; `2ch` defaults to `eta=0.0, guidance_scale=1.0` instead — `eta=1.0` was measured *mixed* for `2ch` (a clear win for `1ch`/`3ch`) and guidance measurably breaks `2ch` even with dynamic thresholding on. This only matters if you're calling `WindDiffusionModel.sample()` directly (Tier 2); `ResourceModel.sample()` (Tier 1, what `.set()`/`.sample()` gives you) always uses the right preset-specific defaults automatically, with no way to override them at that layer.

### Why `ResourceModel._build_conditioning` needed a `condition_builder` override for wind

Wind's conditioning shape doesn't fit the `ConditionField`-based system `pjm_load` uses (see [`docs/extending.md`](extending.md) for the full explanation): which fields exist depends on the checkpoint's own config, one field is a derived+jointly-scaled transform rather than one column per input, and the regime encoding (onehot vs. ordinal) is itself a checkpoint-level choice. `ResourceModel` supports an optional `condition_builder` override for exactly this case — wind is the first (and so far only) resource using it, wrapping `enerdatasynth.resources.wind.evaluation.evaluate.build_condition_vector` directly rather than reimplementing it.

## `enerdatasynth.generators.BaseGenerator`

The common interface `LoadDiffusionModel` and `WindDiffusionModel` both implement. See [`docs/extending.md`](extending.md) for what implementing a third one would involve.

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
from enerdatasynth.generators import WindDiffusionModel
```

`LoadDiffusionModel`/`WindDiffusionModel` (in `enerdatasynth.generators`) are the classes `ResourceModel` wraps internally — they take a raw conditioning array via `.sample(z, seed=...)` rather than `.set()`'s human-readable conditions. `LoadDiffusionModel` exposes `.from_run_id(run_id)`/direct checkpoint-path construction; `WindDiffusionModel(preset=..., checkpoint=..., norm_stats=..., device=...)` delegates entirely to `enerdatasynth.resources.wind.registry.resolve_weights` and `enerdatasynth.resources.wind.evaluation.evaluate.generate_multiple_for_condition` rather than reimplementing either — see `src/enerdatasynth/generators/wind.py`. Use either directly if you need conditioning vectors your own preprocessing already produces, or want Tier-2 controls (`sampler`/`steps`/`eta`/`guidance_scale`/`dynamic_threshold` for wind) that `.set()`/`.sample()` deliberately don't expose. See [`docs/training.md`](training.md) for load's research/development workflow.
