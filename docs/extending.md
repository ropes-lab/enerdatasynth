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
project — see `src/wind_diffusion/`), reachable via
`enerdatasynth.model("wind", zone="1ch")` with a real trained checkpoint
bundled in this repo (`data/checkpoints/wind/1ch/`); `"2ch"`/`"3ch"` have
working code but no published weights yet. See
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
`wind_diffusion.registry.resolve_weights` and loads the model/schedule/
norm_stats; `.sample(z, seed=None, sampler=..., steps=..., eta=...,
guidance_scale=..., dynamic_threshold=...)` delegates the actual
generation to `wind_diffusion.evaluate.generate_multiple_for_condition`
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
`wind_condition_builder`) wraps `wind_diffusion.evaluate.build_condition_vector`
directly — no new logic, just translation. `ResourceModel._missing_required()`
also had to special-case this: a `condition_builder` resource reports
nothing as hard-required, since wind's own defaults (unlike load's
`month`/`Tmax`/`Tmin`) mean there's nothing that must be set before
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

Registering a resource is additive — a new `_REGISTRY` entry in
`src/enerdatasynth/sdk.py`:

```python
_REGISTRY = {
    "pjm_load": {
        "generator_cls": LoadDiffusionModel, "zones": ["PS"],
        "condition_spec": LOAD_CONDITION_SPEC, "extra_set_keys": LOAD_EXTRA_SET_KEYS,
        "construct": _load_construct, "checkpoint_status": _load_checkpoint_status,
    },
    "wind": {
        "generator_cls": WindDiffusionModel, "zones": ["1ch", "2ch", "3ch"],
        "condition_spec": [], "extra_set_keys": WIND_EXTRA_SET_KEYS,
        "construct": wind_construct, "checkpoint_status": wind_checkpoint_status,
    },
}
```

`construct(zone) -> dict` is the one thing that genuinely varies enough
between resources that `ResourceModel` can't build it generically — where
checkpoints/metadata live and how they're resolved. Load's
(`_load_construct`, in `sdk.py`) reads `PROCESSED_DIR/metadata.json` and
scans `CHECKPOINT_DIR` for `result.json` files. Wind's (`wind_construct`,
in `generators/wind_conditioning.py`) checks a bundled
`data/checkpoints/wind/<preset>/` first, then falls through to
`wind_diffusion.registry.resolve_weights`'s own chain (explicit path →
env dir → manifest download → error) — an entirely different mechanism,
already better than load's (sha256-verified, supports remote download)
before any of this wind work started. `construct()`'s return dict
(`generator`, `metadata`, `condition_spec`, `extra_set_keys`,
`condition_builder`, `valid_units`, `postprocess`) is everything
`ResourceModel` needs; `describe_resource(name)`'s `checkpoint_status()`
hook is the same idea for cheap introspection without constructing a full
`ResourceModel` (wind's delegates to `wind_diffusion.list_models()`
rather than load's irrelevant `CHECKPOINT_DIR` scan).

## What's still genuinely unfinished for wind

Concrete, not hand-wavy — these are the actual open items, in rough
priority order:

- **`"2ch"`/`"3ch"` have no published checkpoint.** Trained weights exist
  (in the source Data-For-Good project's local `outputs/`, gitignored
  there and never published), but only `"1ch"`'s were copied into this
  repo. Copying the others in (or setting up `wind_diffusion.manifest.json`'s
  proper download mechanism instead of ad hoc bundling) is the most
  concrete next step.
- **`units="normalized"` isn't offered for wind at all** — see
  [`docs/api_reference.md`](api_reference.md#wind-units-only-ms-is-supported).
- **`previous_day_windspeed`'s smarter default isn't replicated.**
  `wind_diffusion.generate_wind_profile` samples a realistic per-realization
  value from the regime-matched empirical distribution when this isn't
  given; `wind_condition_builder` here falls back to the plain neutral
  default instead (see that function's docstring). Only matters for a
  checkpoint trained with wind-persistence conditioning, which the
  bundled `1ch` one isn't.
- **No multi-zone equivalent for wind** — `wind_diffusion` was trained on
  a single WTK grid point (offshore NJ); `zone` for wind means channel
  preset, not geography, so there's no "add more zones" story the way
  load's `research_history/scripts/zone_expansion.py` has one.

## Suggested real first step for a genuinely new (e.g. solar) resource

Before writing any model code: get real data, do the EDA, and figure out
what a sensible conditioning vector and evaluation metric suite look
like — mirroring how both load's `data/research_history/` and wind's
source project arrived at their pipelines empirically, not by assumption.
`enerdatasynth.evaluation.metrics.compute_metrics` is load-specific
(hourly/first-diff/ACF/RMSE/peak/energy/ramp against a 24-value profile);
wind uses its own separate fidelity-gate suite
(`wind_diffusion.metrics`, Wasserstein/ACF/spectral/regime-accuracy
against a 288-value profile) rather than reusing load's — check what's
actually appropriate before reusing either uncritically.
