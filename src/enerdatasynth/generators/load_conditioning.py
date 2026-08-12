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
"""

from __future__ import annotations

from typing import Union

import numpy as np

from enerdatasynth.conditioning import ConditionField, condition_dim
from enerdatasynth.utils.config import N_DOW, N_MONTHS, COND_DIM

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
        raise ValueError(f"Unrecognized day_of_week: {value!r}")
    if isinstance(value, int) and 0 <= value <= 6:
        return value
    raise ValueError(f"day_of_week must be 0-6 or a day name, got {value!r}")


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
