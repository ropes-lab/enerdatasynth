"""
src/enerdatasynth/conditioning.py
====================================
Generic, resource-agnostic machinery for turning a `ResourceModel`'s
human-readable `.set()` conditions into the raw `(n, dim)` array a
`BaseGenerator.sample()` expects.

Extracted from what was previously load-only logic hand-written directly
in `sdk.ResourceModel._build_conditioning()` -- every field name, column
offset, and scaler lookup was load's exact 6-field schema, encoded as
positional index arithmetic (`Z[:, N_MONTHS+N_DOW+1] = Tmax_z`), even
though `utils.config.CONDITION_SPEC` already *described* that schema
generically and was never actually consulted by that code (only used to
get logged into `metadata.json` during preprocessing).

This module makes a spec load-bearing: a resource type declares a list of
`ConditionField` entries (see `generators/load.py`'s `LOAD_CONDITION_SPEC`
for the reference implementation) and `ResourceModel` builds the vector
generically from whichever spec `_REGISTRY[name]["condition_spec"]`
provides. This is the concrete prerequisite for a second resource type
(e.g. wind) to share `ResourceModel` instead of forking it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


@dataclass(frozen=True)
class ConditionField:
    """One field in a resource's conditioning vector.

    name         : key this field is read from in `ResourceModel._conditions`.
    dim          : how many vector columns this field occupies.
    encoding     : "onehot" | "binary" | "zscore".
    required     : must be set before `.sample()`, or `missing_required()` flags it.
    values       : onehot only -- legal raw values, in the exact order they
                   map to vector columns (e.g. month: `[1, 2, ..., 12]`).
    scaler_key / scaler_index : zscore only -- where to find `(mean_, scale_)`
                   in `ResourceModel._metadata`. `scaler_index` handles a
                   scaler shared across multiple fields (e.g. load's
                   `temp_scaler` covers both Tmax and Tmin as a 2-element
                   array, in that order).
    default      : used when the field is NOT required and not set. For
                   `zscore` fields this is an ALREADY-standardized value
                   (e.g. `0.0` = "dataset mean", matching the exact
                   semantics `spread` always had); for `binary` fields, a
                   raw value passed through `float()` same as an explicit
                   one would be.
    resolve      : optional custom per-row resolver for fields whose value
                   isn't a single shared scalar (e.g. load's day-of-week:
                   "weekday" samples a *different* day per realization,
                   see `generators/load.py`'s `_resolve_dow`). Signature:
                   `(conditions, n, rng) -> np.ndarray` of `n` raw values.
                   When absent, the field's value is looked up once from
                   `conditions[name]` (falling back to `default`) and
                   applied to every row.
    """

    name: str
    dim: int
    encoding: str
    required: bool = False
    values: Optional[list] = None
    scaler_key: Optional[str] = None
    scaler_index: int = 0
    default: Any = None
    resolve: Optional[Callable[[dict, int, np.random.Generator], np.ndarray]] = None


def missing_required(spec: list[ConditionField], conditions: dict) -> list[str]:
    """Names of required fields not yet present in `conditions`."""
    return [f.name for f in spec if f.required and f.name not in conditions]


def condition_dim(spec: list[ConditionField]) -> int:
    """Total vector width implied by `spec` -- derived, not hand-duplicated."""
    return sum(f.dim for f in spec)


def build_condition_vector(
    spec: list[ConditionField],
    conditions: dict,
    metadata: dict,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build an `(n, condition_dim(spec))` array from a resource's spec.

    Raises `ValueError` if a required field is missing, an onehot field's
    value isn't one of its declared `values`, or an unknown encoding is
    declared -- mirrors the exact error conditions/messages the original
    load-only `_build_conditioning()` raised.
    """
    missing = missing_required(spec, conditions)
    if missing:
        raise ValueError(
            f"Missing required condition(s) before sample(): {missing}. "
            f"Call .set({', '.join(f'{m}=...' for m in missing)}) first."
        )

    Z = np.zeros((n, condition_dim(spec)), dtype=np.float32)
    offset = 0
    for f in spec:
        if f.encoding == "onehot":
            raw_per_row = (
                f.resolve(conditions, n, rng) if f.resolve is not None
                else np.full(n, conditions[f.name])
            )
            for i, raw in enumerate(raw_per_row):
                if raw not in f.values:
                    raise ValueError(f"{f.name} must be one of {f.values}, got {raw!r}")
                Z[i, offset + f.values.index(raw)] = 1.0

        elif f.encoding == "binary":
            value = conditions.get(f.name, f.default)
            Z[:, offset] = float(value)

        elif f.encoding == "zscore":
            if f.name in conditions:
                scaler = metadata[f.scaler_key]
                mean = scaler["mean_"][f.scaler_index]
                scale = scaler["scale_"][f.scaler_index]
                Z[:, offset] = (conditions[f.name] - mean) / scale
            else:
                Z[:, offset] = f.default  # already-standardized default

        else:
            raise ValueError(f"Unknown encoding {f.encoding!r} for field {f.name!r}")

        offset += f.dim

    return Z
