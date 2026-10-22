# Extending: Adding a New Generator (e.g. Solar)

This package's public interface, `enerdatasynth.generators.BaseGenerator`,
is deliberately thin so a different PJM resource type could be added as
a first-class citizen of the SDK — callable through the same
`.sample(z, seed=...)` contract as `LoadDiffusionModel` — without
requiring downstream tooling (CLIs, dashboards, benchmarking harnesses)
to special-case it.

**Wind (`WindDiffusionModel`) is now a real, working second example** —
not hypothetical anymore. It wraps a diffusion pipeline developed
separately (`wind_diffusion`, vendored from a Data-For-Good research
project — see `src/enerdatasynth/resources/wind/`), reachable via
`enerdatasynth.model("wind", zone=...)` with real trained checkpoints
bundled in this repo for all three channel presets
(`data/checkpoints/wind/{1ch,2ch,3ch}/`). See
[`docs/api_reference.md`](api_reference.md#wind) for its full parameter
reference. This page now describes both what `BaseGenerator` requires in
the abstract *and*, concretely, how wind actually satisfies it — useful
as a template for a third resource type (solar, say), and as an honest
record of which parts of the original "thin, resource-agnostic" design
held up and which needed extending.

## What `BaseGenerator` requires

```python
from enerdatasynth.generators.base import BaseGenerator

class WindDiffusionModel(BaseGenerator):
    domain = "wind"

    def sample(self, z, seed=None, **kwargs) -> np.ndarray:
        ...
```

That's the entire hard requirement: a `domain` string and a `sample`
method returning one synthetic profile per row of `z`. `count_parameters()`
is optional (override it if there's an underlying `nn.Module` — wind's
does).

**How wind's actually looks** (`src/enerdatasynth/generators/wind.py`):
`WindDiffusionModel.__init__` resolves a checkpoint via
`enerdatasynth.resources.wind.registry.resolve_weights` and loads the model/schedule/
norm_stats; `.sample(z, seed=None, sampler=..., steps=..., eta=...,
guidance_scale=..., dynamic_threshold=...)` delegates the actual
generation to `enerdatasynth.resources.wind.evaluation.evaluate.generate_multiple_for_condition`
and returns `(N, seq_len, 2)` — windspeed and direction stacked on a new
last axis, since collapsing to windspeed-only would silently drop
direction for `2ch`/`3ch`. Nothing about the diffusion sampling itself is
reimplemented here; the adapter's job is purely resolving checkpoints and
translating between `BaseGenerator`'s raw-array contract and
`wind_diffusion`'s own richer (human-readable-argument) API underneath.

## What `BaseGenerator` does NOT prescribe — confirmed by wind

Deliberately, `BaseGenerator` says nothing about:

- **Conditioning schema.** Predicted correctly: wind's conditioning
  (month/day-of-week/wind-regime/temp-min/temp-max/wind-persistence,
  `cond_dim` varying by channel preset) is genuinely different from
  load's (month/dow/holiday/Tmax/Tmin/spread, fixed `cond_dim=23`) — see
  "Conditioning: the part that needed real new plumbing" below for what
  that actually required.
- **Temporal resolution.** Confirmed: wind profiles are `seq_len=288`
  (5-minute cadence over a day), not load's `x_dim=24` (hourly). Nothing
  in `BaseGenerator` assumed otherwise, and nothing needed to change for
  this.
- **Constructor signature.** Confirmed: `WindDiffusionModel.__init__`
  takes `preset`/`checkpoint`/`norm_stats`/`device` — nothing like
  `LoadDiffusionModel`'s `run_id`/`result.json` convention.
  `WindDiffusionModel.from_run_id(run_id, **kwargs)` exists only for
  superficial symmetry (`run_id` *is* the preset for wind); it doesn't
  read a `result.json` the way load's does, because `wind_diffusion` has
  no such file.
- **Underlying architecture.** Confirmed: wind reuses none of
  `TransformerDenoiser`/`diffusion/ddpm.py` — it wraps an entirely
  separate, separately-validated model (`wind_diffusion`'s own
  Conv1D+Transformer denoiser) rather than fitting into this package's
  existing training/model code.

## Conditioning: the part that needed real new plumbing

This is the one place the original version of this doc under-predicted
the work. It's not just "a different schema" — wind's conditioning is
structurally different from load's in ways `ResourceModel`'s original
(load-only) `_build_conditioning()` couldn't express even after
generalizing it into the `conditioning.ConditionField`/`build_condition_vector`
system:

- **Field presence is config-dependent.** Which fields exist in the
  vector at all is driven by the trained checkpoint's own `condition_on`
  dict — not a fixed schema the way load's 6 fields always are.
- **A derived, jointly-scaled field.** `temp_min_max_spread` is a single
  3-dim z-scored transform of `[temp_max, temp_min, temp_max-temp_min]`
  — one *computed* value sharing a scaler, not one column per raw input
  the way `ConditionField` assumes.
- **Checkpoint-level encoding choice.** `wind_regime` is either one-hot
  or ordinal depending on `NormStats.wind_regime_encoding` — the encoding
  itself is a per-checkpoint decision, not a fixed property of the field.

Rather than extend `ConditionField` to cover this shape (not justified
before a real second field-modeled resource exists to validate the
extension against), `ResourceModel` gained an optional `condition_builder`
override: a resource can provide a full custom
`(conditions, metadata, n, rng) -> np.ndarray` callable instead of a
`condition_spec`. Wind's (`generators/wind_conditioning.py`'s
`wind_condition_builder`) wraps `enerdatasynth.resources.wind.evaluation.evaluate.build_condition_vector`
directly — no new logic, just translation. `ResourceModel._missing_required()`
also had to special-case this: a `condition_builder` resource reports
nothing as hard-required, since wind's own defaults (unlike load's
`month`) mean there's nothing that must be set before
`.sample()`.

**If you're adding a third resource**: try the `condition_spec`
(`ConditionField`) path first — it's less code and it's what load
actually uses. Only reach for `condition_builder` if your resource's
conditioning genuinely doesn't fit "one field, one column, one scaler"
the way wind's doesn't.

## Units and denormalization also needed a per-resource hook

Load's `.sample(units="MW"|"normalized")` used a hardcoded
`_denormalize()` against `metadata["load_scaler"]`. `ResourceModel` now
takes an optional `postprocess: Callable[[np.ndarray, str], np.ndarray]`
and `valid_units` tuple per resource — wind's `postprocess` is a no-op
(the underlying `generate_multiple_for_condition` already returns
physical units) and its `valid_units` is `("m/s",)` only; see
[`docs/api_reference.md`](api_reference.md#wind-units-only-ms-is-supported) for why
`"normalized"` isn't offered for wind.

## Wiring into `enerdatasynth.model(...)`

Registering a resource is additive and self-contained — a resource
module calls `registry.register_resource(...)` on itself at import time;
`src/enerdatasynth/sdk.py` never needs editing to add one. Load's and
wind's registrations (in `generators/load_conditioning.py` and
`generators/wind_conditioning.py` respectively) are the two reference
implementations:

```python
# bottom of generators/wind_conditioning.py
register_resource(ResourceEntry(
    name="wind", generator_cls=WindDiffusionModel, zones=["1ch", "2ch", "3ch"],
    condition_spec=[], extra_set_keys=WIND_EXTRA_SET_KEYS,
    construct=wind_construct, checkpoint_status=wind_checkpoint_status,
))
```

`ResourceEntry`/`ConstructResult` (`src/enerdatasynth/registry.py`) are
frozen dataclasses, not raw dicts — a missing/misspelled field is a
`TypeError` at the registration call site, not a silent `None` three
calls later. `register_resource` also rejects a duplicate `name` unless
you pass `replace=True`, so a copy-pasted resource module that forgot to
rename itself fails loudly at import time.

`construct(zone) -> ConstructResult` is the one thing that genuinely
varies enough between resources that `ResourceModel` can't build it
generically — where checkpoints/metadata live and how they're resolved.
Load's (`_load_construct`, in `generators/load_conditioning.py`) reads
`PROCESSED_DIR/metadata.json` and scans `CHECKPOINT_DIR` for
`result.json` files. Wind's (`wind_construct`, in
`generators/wind_conditioning.py`) checks a bundled
`data/checkpoints/wind/<preset>/` first, then falls through to
`enerdatasynth.resources.wind.registry.resolve_weights`'s own chain (explicit path →
env dir → manifest download → error) — an entirely different mechanism,
already better than load's (sha256-verified, supports remote download)
before any of this wind work started. `ConstructResult`'s fields
(`generator`, `metadata`, `condition_spec`, `extra_set_keys`,
`condition_builder`, `valid_units`, `postprocess`) are everything
`ResourceModel` needs; `describe_resource(name)`'s `checkpoint_status()`
hook is the same idea for cheap introspection without constructing a full
`ResourceModel` (wind's delegates to `wind_diffusion.list_models()`
rather than load's irrelevant `CHECKPOINT_DIR` scan).

**Adding a third resource, concretely**: two things, at two different layers.

1. **The bridge** (always required, small): a new `generators/<name>.py`
   (a `BaseGenerator` subclass — you get `count_parameters()` and the
   `sample()`-array-normalization helper `_ensure_batch()` for free as long
   as you store your model at `self.model`) and a new
   `generators/<name>_conditioning.py` (a `condition_spec` or
   `condition_builder`, a `construct()`/`checkpoint_status()` pair, and the
   `register_resource(...)` call), plus one import line in
   `generators/__init__.py` so that registration side effect actually runs.
   Nothing in `sdk.py` changes. This part is genuinely resource-agnostic —
   both load's and wind's bridge files follow this exact same shape.

2. **The implementation** (the actual model, and only if you're building it
   from scratch rather than wrapping an existing pipeline the way wind
   does): lives under `src/enerdatasynth/resources/<name>/`, mirroring
   `resources/pjm_load/`'s category taxonomy wherever a category genuinely
   applies to your resource — `data/` (raw → cleaned tensors), `models/`
   (denoiser/architecture code), `diffusion/` (or whatever your generative
   process actually is), `training/`, `evaluation/`, `sampling/`,
   `visualization/`. Not every category needs to exist for every resource
   — wind, for example, has no `sampling/` folder of its own and keeps
   `config.py`/`registry.py`/`exceptions.py` at its resource root instead of
   forcing them into one of the shared categories, since load has no
   equivalent for those. Match the categories that make sense; don't invent
   empty ones just for symmetry. See [`docs/architecture.md`](architecture.md)
   for the full reasoning behind this layout, and the `README.md` inside
   each of `resources/pjm_load/{data,models,diffusion,training,evaluation,
   sampling,visualization}/` for what each category actually contains in a
   real, from-scratch resource implementation.

   If you're instead *wrapping* an existing external pipeline (wind's
   situation), the same `resources/<name>/` location applies, but its
   internal layout can just mirror whatever that pipeline's own module
   structure already is — see `resources/wind/README.md` for how that
   vendored package's original files map onto the shared category names.

For example: a hypothetical `water` resource generating synthetic reservoir
or usage profiles would get `generators/water.py` +
`generators/water_conditioning.py` (the bridge) and
`resources/water/{data,models,...}/` (the implementation) — nothing about
`sdk.py` or any other resource's code would need to change for either
piece to exist, and `enerdatasynth.model("water", ...).set(...).sample(...)`
would work immediately once `register_resource(...)` runs.

**One real exception, added since the above was first written**: the CLI's
`generate`/`evaluate`/`report` subcommands take an explicit
`--resource {pjm_load,wind}` flag (`cli.py`'s `build_parser`) rather than
reading the registry dynamically — a third resource needs its `choices=[...]`
list extended to include `"water"` and a matching `if args.resource ==
"water":` branch added to each of `cmd_generate`/`cmd_evaluate`/`cmd_report`
before it's reachable from those three CLI subcommands. This is a small,
disclosed limitation of the CLI *convenience* layer specifically, not the
core registration mechanism — `enerdatasynth.model("water", ...)` from
Python needs none of this.

**One more thing that needed to be per-resource, discovered only once
multiple wind presets were actually exercised together**: generation-time
settings (`eta`, `guidance_scale`, ...) aren't safely shareable across
even a *single* resource's own presets, let alone across resources —
`1ch`/`3ch`'s recommended `eta=1.0` is measurably wrong for `2ch`
(`eta=0.0`). `WindDiffusionModel` now reads each preset's own
`recommended_generation` from `wind_diffusion`'s bundled `manifest.json`
at construction time rather than hardcoding one set of Tier-2 defaults —
see [`docs/api_reference.md`](api_reference.md#checkpoints) for the
concrete numbers. Worth checking for in a third resource too if it ends
up with multiple presets/variants of its own.

## What's still genuinely unfinished for wind

Concrete, not hand-wavy — these are the actual open items, in rough
priority order:

- **Checkpoints are bundled ad hoc, not properly published.** All three
  presets' weights are committed directly under `data/checkpoints/wind/`
  (copied from the source Data-For-Good project's local `outputs/`,
  gitignored there and never published anywhere else) rather than going
  through `wind_diffusion.manifest.json`'s intended mechanism (a declared
  download URL + sha256, resolved and cached by
  `enerdatasynth.resources.wind.registry.resolve_weights` on demand). That mechanism
  already exists and works — it's just not wired up for any preset yet.
  Committing ~14MB of checkpoint binaries directly into this repo's git
  history was the fast path, not the durable one; moving to real
  published weights (GitHub Releases or similar) is the concrete next
  step, most valuable once checkpoint updates need to happen more than
  once.
- **`units="normalized"` isn't offered for wind at all** — see
  [`docs/api_reference.md`](api_reference.md#wind-units-only-ms-is-supported).
- **`previous_day_windspeed`'s smarter default isn't replicated.**
  `wind_diffusion.generate_wind_profile` samples a realistic per-realization
  value from the regime-matched empirical distribution when this isn't
  given; `wind_condition_builder` here falls back to the plain neutral
  default instead (see that function's docstring). Only matters for a
  checkpoint trained with wind-persistence conditioning, which none of
  the three bundled checkpoints are.
- **No multi-zone equivalent for wind** — `wind_diffusion` was trained on
  a single WTK grid point (offshore NJ); `zone` for wind means channel
  preset, not geography, so there's no "add more zones" story the way
  load's `research_history/scripts/zone_expansion.py` has one.

## Suggested real first step for a genuinely new (e.g. solar) resource

Before writing any model code: get real data, do the EDA, and figure out
what a sensible conditioning vector and evaluation metric suite look
like — mirroring how both load's `data/research_history/pjm_load/` and wind's
source project arrived at their pipelines empirically, not by assumption.
`enerdatasynth.resources.pjm_load.evaluation.metrics.compute_metrics` is load-specific
(hourly/first-diff/ACF/RMSE/peak/energy/ramp against a 24-value profile);
wind uses its own separate fidelity-gate suite
(`enerdatasynth.resources.wind.evaluation.metrics`, Wasserstein/ACF/spectral/regime-accuracy
against a 288-value profile) rather than reusing load's — check what's
actually appropriate before reusing either uncritically.
