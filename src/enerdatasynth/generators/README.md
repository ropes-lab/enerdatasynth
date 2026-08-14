# `enerdatasynth.generators`

**This is where PJM load and wind actually become one product.** Everywhere else in
`src/enerdatasynth/` (`models/`, `diffusion/`, `training/`, `sampling/`, `evaluation/`,
`data/`) is load-only, and wind's equivalent code lives entirely in the separate
`src/wind_diffusion/` package — this subpackage is the shared layer that makes both
resources drivable identically through one interface despite that split. `BaseGenerator`
is the common contract every resource-type generator implements, so `sdk.py`'s
`ResourceModel` can drive any of them the same way, regardless of what's underneath.

## Files

- **`base.py`** — `BaseGenerator`, an actually-enforced ABC (`domain` attribute +
  `.sample()`). Subclassing without implementing both raises `TypeError` at
  instantiation, not silently.
- **`load.py`** — `LoadDiffusionModel`, the electric-load implementation. Wraps
  checkpoint loading, architecture selection, the DDPM sampler, and calibration behind
  one object.
- **`load_conditioning.py`** — `LOAD_CONDITION_SPEC` (load's conditioning schema as
  `ConditionField` entries) + `_load_construct`/`_load_checkpoint_status` (the
  `model("pjm_load", ...)` construction hooks) + the `register_resource(...)` call that
  actually adds `"pjm_load"` to the shared registry. **This is the reference
  implementation** a third resource type would copy.
- **`wind.py`** — `WindDiffusionModel`, wrapping the vendored `wind_diffusion` package
  (see `src/wind_diffusion/`). Real trained checkpoints for all three presets (1ch/2ch/
  3ch) are bundled under `data/checkpoints/wind/`.
- **`wind_conditioning.py`** — wind's registration + its `condition_builder` (an
  escape hatch from the generic `ConditionField` model — see the module docstring for
  exactly why wind's conditioning shape doesn't fit it).

## Public API vs. internal

`BaseGenerator` and `LoadDiffusionModel` are re-exported at the top level (Tier 2,
research/development use). `WindDiffusionModel` is reached via
`enerdatasynth.generators.WindDiffusionModel` or, for everyday use, simply
`enerdatasynth.model("wind", zone=<preset>)` (Tier 1).

## How a new resource type gets added

Copy `load_conditioning.py`'s pattern: define a `ConditionField` list (or a
`condition_builder` if the shape doesn't fit that model, like wind's), write a
`_construct(zone)` hook returning a `ConstructResult`, call `register_resource(...)` at
import time, and add one import line to this directory's `__init__.py`. `sdk.py` itself
never needs editing. See `docs/extending.md` for the full walkthrough.

## What should not be changed casually

`load_conditioning.py`'s `LOAD_CONDITION_SPEC` has a runtime assertion tying its implied
width to `utils.config.COND_DIM` — they must stay in sync, since `LoadDiffusionModel`'s
default `cond_dim` is `COND_DIM`. Changing one without the other breaks silently at
generation time (a shape mismatch inside the model, not a clear error at the boundary).

## Where the scientific provenance lives

Wind's checkpoints are validated only in the sense that they load and produce
plausible-shaped output — see `docs/research_status.md` and `docs/model_report_card.md`
for the honest distinction between "code-complete" and "scientifically validated" for
wind specifically. No wind result belongs in `docs/results.md`, which is load-only.
