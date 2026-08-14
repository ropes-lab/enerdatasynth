# `enerdatasynth.generators`

**Where PJM load and wind become one product.** Everywhere else under
`src/enerdatasynth/resources/` is per-resource; this subpackage is the shared layer
that drives both identically. `BaseGenerator` is the common contract every
resource-type generator implements, so `sdk.py`'s `ResourceModel` can drive any of them
the same way regardless of what's underneath.

## Files

- **`base.py`** — `BaseGenerator`, an enforced ABC (`domain` + `.sample()`).
  Subclassing without implementing both raises `TypeError` at instantiation.
- **`load.py`** — `LoadDiffusionModel`. Wraps checkpoint loading, architecture
  selection, the DDPM sampler, and calibration behind one object.
- **`load_conditioning.py`** — `LOAD_CONDITION_SPEC`, `_load_construct`/
  `_load_checkpoint_status`, and the `register_resource(...)` call that adds
  `"pjm_load"`. **The reference implementation a third resource copies.**
- **`wind.py`** — `WindDiffusionModel`, wrapping `resources/wind/`. Real checkpoints
  for all three presets are bundled under `data/checkpoints/wind/`.
- **`wind_conditioning.py`** — wind's registration + its `condition_builder` (an
  escape hatch from `ConditionField` — see the module docstring for why wind needs it).

## Public API vs. internal

`BaseGenerator` and `LoadDiffusionModel` are re-exported at the top level (Tier 2).
`WindDiffusionModel` is reached the same way, or via
`enerdatasynth.model("wind", zone=<preset>)` (Tier 1, everyday use).

## Adding a new resource type

Copy `load_conditioning.py`'s pattern: a `ConditionField` list (or `condition_builder`
if the shape doesn't fit, like wind's), a `_construct(zone)` hook returning a
`ConstructResult`, a `register_resource(...)` call at import time, and one import line
in this folder's `__init__.py`. `sdk.py` never needs editing. Full walkthrough:
`docs/extending.md`.

## What should not be changed casually

`load_conditioning.py`'s `LOAD_CONDITION_SPEC` has a runtime assertion tying its width
to `utils.config.COND_DIM` — changing one without the other breaks generation silently
(a shape mismatch inside the model, not a clear error at the boundary).

## Where the scientific provenance lives

Wind's checkpoints are validated only in that they load and produce plausible-shaped
output — see `docs/research_status.md` for the "code-complete" vs. "scientifically
validated" distinction. No wind result belongs in `docs/results.md`, which is load-only.
