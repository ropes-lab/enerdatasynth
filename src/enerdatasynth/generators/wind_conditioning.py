"""
src/enerdatasynth/generators/wind_conditioning.py
=====================================================
Wind's `.set()`-accepted keys and the custom raw-conditioning-vector
builder `wind_diffusion` needs, plus the `model("wind", zone=...)`
construction hook (`wind_construct`).

Unlike load's fixed 6-field schema (`load_conditioning.py`'s
`LOAD_CONDITION_SPEC`, built from `conditioning.ConditionField`), wind's
conditioning shape is genuinely config-dependent: which fields exist at
all is driven by the trained checkpoint's own `condition_on` dict, one
field (`temp_min_max_spread`) is a joint 3-dim z-scored transform of a
*derived* value (`spread = temp_max - temp_min`) rather than one column
per field, and `wind_regime`'s encoding (onehot vs. ordinal) is itself a
checkpoint-level choice (`NormStats.wind_regime_encoding`). Forcing this
through `ConditionField`'s one-field-per-entry model would mean
meaningfully extending it for a shape only wind needs today -- not
justified before a real second field-modeled resource exists to validate
the extension against. Instead, `ResourceModel` supports an optional
`condition_builder` override (see `sdk.py`) that a resource with a
genuinely different conditioning shape can provide instead of
`condition_spec` -- wind is the first to use it.
"""

from __future__ import annotations

import numpy as np

from wind_diffusion.data import NormStats
from wind_diffusion.evaluate import build_condition_vector as _wind_build_condition_vector

from enerdatasynth.generators.wind import WindDiffusionModel
from enerdatasynth.utils.config import CHECKPOINT_DIR

# enerdatasynth's own convention for a locally-bundled wind checkpoint --
# checked before falling back to wind_diffusion's own resolution chain
# (explicit path > WIND_DIFFUSION_CHECKPOINTS env dir > manifest-declared
# download > CheckpointNotFoundError). Mirrors this repo's existing
# practice of committing load's checkpoints directly under
# data/checkpoints/ (unlike the source wind_diffusion project, which
# gitignores its own outputs/ and has never published weights anywhere).
# Currently populated for "1ch" only -- see README.md/docs/api_reference.md.
_WIND_CHECKPOINT_ROOT = CHECKPOINT_DIR / "wind"

# .set()-accepted keys, mirroring generate_wind_profile's human-facing
# conditioning args (see the original repo's API_REVIEW.md "core" group)
# -- NOT its Tier-2 concerns (checkpoint/sampler/eta/guidance_scale/
# rejection_sampling stay hidden at this layer, same as load hides
# architecture/checkpoint paths from ResourceModel.set()).
WIND_EXTRA_SET_KEYS = [
    "wind_regime", "month", "day_of_week", "temp_min", "temp_max",
    "previous_day_windspeed",
]


def wind_condition_builder(conditions: dict, metadata: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """`ResourceModel`'s `condition_builder` override for the `"wind"`
    resource. `metadata` here is `{"norm_stats": <NormStats.to_dict()>,
    "condition_on": <dict>}` (set by `wind_construct` below) --
    deliberately not the same shape load's metadata dict has, since
    `ResourceModel` treats `metadata` as opaque, resource-owned state
    passed straight through to whichever builder/spec that resource
    provides.

    Simplification vs. `wind_diffusion.generate_wind_profile` directly:
    when `previous_day_windspeed` isn't set, this falls back to
    `build_condition_vector`'s own neutral "no info" default (standardized
    0.0) rather than `generate_wind_profile`'s fancier per-realization
    sampling from the regime-matched empirical distribution
    (`NormStats.persistence_by_regime`) -- that logic lives in
    `wind_diffusion.api._build_cond_matrix_for_batch`, one layer above
    `build_condition_vector`, and isn't replicated here. Worth revisiting
    if wind persistence conditioning becomes the shipped default recipe.
    """
    norm_stats = NormStats.from_dict(metadata["norm_stats"])
    condition_on = metadata["condition_on"]

    kwargs = {
        key: conditions[key]
        for key in ("month", "day_of_week", "temp_min", "temp_max", "previous_day_windspeed")
        if key in conditions
    }
    wind_regime = conditions.get("wind_regime") or (
        "mid" if "mid" in norm_stats.wind_buckets else norm_stats.wind_buckets[0]
    )
    if wind_regime not in norm_stats.wind_buckets:
        raise ValueError(f"wind_regime={wind_regime!r} not recognized; expected one of {norm_stats.wind_buckets}.")

    vec = _wind_build_condition_vector(norm_stats, condition_on, wind_regime=wind_regime, **kwargs)
    return np.tile(vec[None, :], (n, 1)).astype(np.float32)


def wind_postprocess(samples: np.ndarray, units: str) -> np.ndarray:
    """`ResourceModel.sample()`'s postprocess override for wind.
    `WindDiffusionModel.sample()` already returns physical units
    (windspeed m/s, direction degrees) via `wind_diffusion`'s own
    denormalization -- no further transform needed here. Only `"m/s"` is
    a valid `units` value for wind today (see `wind_construct`'s
    `valid_units`); `"normalized"` isn't offered since
    `generate_multiple_for_condition` bakes denormalization into its
    output rather than exposing a raw-model-output option at that layer --
    a real, documented gap vs. load's `units="normalized"`, not an
    oversight.
    """
    return samples


def wind_construct(zone: str) -> dict:
    """`model("wind", zone=...)`'s construction hook (see `sdk.py`).

    `zone` is repurposed as the channel preset (`"1ch"`/`"2ch"`/`"3ch"`)
    for wind -- there's no geographic-zone concept the way load has `"PS"`;
    reusing the existing zone-selection mechanism avoids inventing a
    second construction-time-selector parameter just for wind.

    Raises `wind_diffusion.exceptions.CheckpointNotFoundError` if no
    trained checkpoint has been published for the requested preset yet
    (expected for "2ch"/"3ch" today -- see `_WIND_CHECKPOINT_ROOT`'s note;
    "1ch" resolves to the bundled checkpoint below without needing any
    env var or explicit path).
    """
    bundled_dir = _WIND_CHECKPOINT_ROOT / zone
    bundled_checkpoint = bundled_dir / "model_checkpoint.pt"
    bundled_norm_stats = bundled_dir / "norm_stats.json"
    if bundled_checkpoint.exists() and bundled_norm_stats.exists():
        generator = WindDiffusionModel(preset=zone, checkpoint=bundled_checkpoint, norm_stats=bundled_norm_stats)
    else:
        generator = WindDiffusionModel(preset=zone)  # falls through to wind_diffusion's own resolution chain
    return {
        "generator": generator,
        "metadata": {
            "norm_stats": generator.norm_stats.to_dict(),
            "condition_on": generator.config.condition_on,
        },
        "condition_spec": [],
        "extra_set_keys": WIND_EXTRA_SET_KEYS,
        "condition_builder": wind_condition_builder,
        "valid_units": ("m/s",),
        "postprocess": wind_postprocess,
    }


def wind_checkpoint_status() -> dict:
    """`describe_resource("wind")`'s checkpoint-availability hook (see
    `sdk.py`) -- delegates to `wind_diffusion`'s own registry
    introspection (`list_models()`) rather than reusing load's
    `CHECKPOINT_DIR`-scanning `_discover_best_checkpoint`, which knows
    nothing about wind's manifest-based resolution.
    """
    from wind_diffusion import list_models
    return {"per_preset": list_models()}
