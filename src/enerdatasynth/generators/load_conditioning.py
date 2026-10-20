"""
src/enerdatasynth/generators/load_conditioning.py
====================================================
`LOAD_CONDITION_SPEC` -- electric load's conditioning schema, expressed as
`conditioning.ConditionField` entries instead of hand-typed index
arithmetic. This is the reference implementation a second resource type
(e.g. wind) would mirror with its own `<domain>_conditioning.py`.

Day-of-week is the one field that needs custom per-row resolution rather
than a single shared value: `weekday=True` (default) samples a *different*
uniformly-random weekday independently for each generated profile rather
than fixing one day for the whole batch (see `_resolve_dow`'s docstring
for why -- this is a genuine UX decision, not a validated finding, kept
exactly as it was before this file existed).

Also owns load's `construct(zone) -> registry.ConstructResult` /
`checkpoint_status()` hooks and registers `"pjm_load"` with
`registry.register_resource(...)` at the bottom of this file -- moved
here from `sdk.py` so load's file layout mirrors wind's
(`generators/wind_conditioning.py` already worked this way), which is
what makes "add a resource by copying the pattern" actually true.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np

from enerdatasynth.conditioning import ConditionField, condition_dim
from enerdatasynth.generators.load import LoadDiffusionModel
from enerdatasynth.registry import ConstructResult, ResourceEntry, register_resource
from enerdatasynth.resources.pjm_load.exceptions import CheckpointNotFoundError, InvalidConditionError
from enerdatasynth.utils.config import (
    N_DOW, N_MONTHS, COND_DIM, checkpoint_search_dirs, processed_metadata_path,
)

_DOW_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WEEKDAY_INDICES = [0, 1, 2, 3, 4]
_WEEKEND_INDICES = [5, 6]


def _parse_day_of_week(value: Union[int, str]) -> int:
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _DOW_NAMES:
            return _DOW_NAMES.index(key)
        abbrevs = {n[:3]: i for i, n in enumerate(_DOW_NAMES)}
        if key in abbrevs:
            return abbrevs[key]
        raise InvalidConditionError(f"Unrecognized day_of_week: {value!r}")
    if isinstance(value, int) and 0 <= value <= 6:
        return value
    raise InvalidConditionError(f"day_of_week must be 0-6 or a day name, got {value!r}")


def _resolve_dow(conditions: dict, n: int, rng: np.random.Generator) -> np.ndarray:
    """Exact `day_of_week` override takes precedence; otherwise samples
    from the weekday/weekend pool implied by `weekday` (default True),
    independently per row -- there's no single "correct" weekday to pick
    for an unspecified one, so this avoids arbitrarily fixing e.g.
    Wednesday for every realization in the batch."""
    if "day_of_week" in conditions:
        return np.full(n, _parse_day_of_week(conditions["day_of_week"]))
    is_weekday = conditions.get("weekday", True)
    pool = _WEEKDAY_INDICES if is_weekday else _WEEKEND_INDICES
    return rng.choice(pool, size=n)


LOAD_CONDITION_SPEC = [
    ConditionField(name="month", dim=N_MONTHS, encoding="onehot", required=True, values=list(range(1, 13))),
    ConditionField(name="day_of_week", dim=N_DOW, encoding="onehot", values=list(range(7)), resolve=_resolve_dow),
    ConditionField(name="holiday", dim=1, encoding="binary", default=False),
    ConditionField(name="Tmax", dim=1, encoding="zscore", required=True, scaler_key="temp_scaler", scaler_index=0),
    ConditionField(name="Tmin", dim=1, encoding="zscore", required=True, scaler_key="temp_scaler", scaler_index=1),
    ConditionField(name="spread", dim=1, encoding="zscore", scaler_key="spread_scaler", scaler_index=0, default=0.0),
]

# `weekday` is a valid .set() input but not a ConditionField of its own --
# it's one of two inputs _resolve_dow reads from `conditions` (the other,
# `day_of_week`, IS a declared field name above). ResourceModel.set()
# validates against LOAD_CONDITION_SPEC's field names plus this list.
LOAD_EXTRA_SET_KEYS = ["weekday"]

assert condition_dim(LOAD_CONDITION_SPEC) == COND_DIM, (
    f"LOAD_CONDITION_SPEC's implied width ({condition_dim(LOAD_CONDITION_SPEC)}) "
    f"no longer matches utils.config.COND_DIM ({COND_DIM}) -- these must stay in "
    f"sync since LoadDiffusionModel's default cond_dim is COND_DIM."
)


def _discover_best_checkpoint(checkpoint_root: Path | None = None) -> str:
    """
    Find the best available trained checkpoint: prefers any
    architecture="transformer" run (lowest best_val_loss among those),
    since that's the validated-best architecture (see MIGRATION.md);
    falls back to the best "mlp" run if no transformer checkpoint has
    been trained against the current preprocessing pipeline. The shipped
    default (`transformer_T1000_H64_L2`) is a transformer run, so this
    resolves to it; the seven legacy MLP runs only matter if it is
    removed. Training another transformer (`enerdatasynth train`) is
    picked up automatically, with no code changes needed here.

    Load-specific: scans CHECKPOINT_DIR for result.json files, a
    convention only enerdatasynth's own training pipeline produces. Wind's
    equivalent (`generators/wind_conditioning.py`'s `wind_construct`)
    resolves checkpoints through `enerdatasynth.resources.wind.registry` instead --
    a different mechanism entirely, not something this function generalizes.
    """
    # An explicit root searches only that folder. By default, search the
    # project's data/checkpoints/pjm_load/ first, then the checkpoints
    # bundled in the package; a run id in both is taken from the project.
    roots = [Path(checkpoint_root)] if checkpoint_root is not None else checkpoint_search_dirs()
    by_run = {}
    for root in reversed(roots):
        if root.exists():
            for run_dir in sorted(root.iterdir()):
                result_path = run_dir / "result.json"
                if result_path.exists():
                    with open(result_path) as f:
                        by_run[run_dir.name] = json.load(f)
    candidates = list(by_run.values())

    if not candidates:
        raise CheckpointNotFoundError(
            f"No trained checkpoints found under {roots[0]} or in the installed package. "
            f"Run `enerdatasynth train` first (see docs/quickstart.md)."
        )

    transformer_runs = [r for r in candidates if r.get("architecture") == "transformer"]
    pool = transformer_runs if transformer_runs else candidates
    best = min(pool, key=lambda r: r["best_val_loss"])
    return best["run_id"]


def _load_construct(zone: str) -> ConstructResult:
    """`model("pjm_load", zone=...)`'s construction hook -- the exact
    logic `model()` used to run inline before construction became
    resource-delegated, moved here (from `sdk.py`) so `sdk.py` itself
    needs no per-resource edit; mirrors `wind_conditioning.wind_construct`."""
    metadata_path = processed_metadata_path()
    if not metadata_path.exists():
        raise CheckpointNotFoundError(
            f"No scaler file (metadata.json) found at {metadata_path}. "
            f"Run `enerdatasynth preprocess` first (see docs/quickstart.md)."
        )
    with open(metadata_path) as f:
        metadata = json.load(f)

    run_id = _discover_best_checkpoint()
    generator = LoadDiffusionModel.from_run_id(run_id)
    return ConstructResult(
        generator=generator,
        metadata=metadata,
        condition_spec=LOAD_CONDITION_SPEC,
        extra_set_keys=LOAD_EXTRA_SET_KEYS,
    )


def _load_checkpoint_status() -> dict:
    """`describe_resource("pjm_load")`'s checkpoint-availability hook."""
    status = {"processed_dataset_available": processed_metadata_path().exists()}
    try:
        status["best_run_id"] = _discover_best_checkpoint()
        status["checkpoint_available"] = True
    except FileNotFoundError:
        status["best_run_id"] = None
        status["checkpoint_available"] = False
    return status


register_resource(ResourceEntry(
    name="pjm_load",
    generator_cls=LoadDiffusionModel,
    zones=["PS"],
    condition_spec=LOAD_CONDITION_SPEC,
    extra_set_keys=LOAD_EXTRA_SET_KEYS,
    construct=_load_construct,
    checkpoint_status=_load_checkpoint_status,
))
