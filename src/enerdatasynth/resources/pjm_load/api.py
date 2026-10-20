"""Standalone public API for the PJM load resource: `generate_load_profile` +
`LoadProfile`, plus `list_models`/`describe_model` manifest lookups — the resource-level
equivalent of `resources/wind/api.py`, added for parity (wind had a standalone,
resource-independent API; load didn't). This wires together the existing
`generators.load`/`generators.load_conditioning` machinery behind one ergonomic
function, exactly the way `resources/wind/api.py` wires together wind's registry/
evaluate/diffusion modules — no new generation logic lives here, only the same
composition wind's file already demonstrated.

Unlike wind's `generate_wind_profile`, this does not need a separate checkpoint/
norm_stats resolution step: `_discover_best_checkpoint()` + `LoadDiffusionModel.from_run_id`
already do that (see `generators/load_conditioning.py`).
"""

from __future__ import annotations

import importlib.resources
import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from enerdatasynth.conditioning import build_condition_vector
from enerdatasynth.generators.load import LoadDiffusionModel
from enerdatasynth.generators.load_conditioning import (
    LOAD_CONDITION_SPEC,
    _discover_best_checkpoint,
)
from enerdatasynth.resources.pjm_load.exceptions import CheckpointNotFoundError, InvalidConditionError
from enerdatasynth.utils.config import ZONE, processed_metadata_path


# ---------------------------------------------------------------------------
# LoadProfile
# ---------------------------------------------------------------------------


@dataclass
class LoadProfile:
    """A batch of generated synthetic 24-hour load-profile realizations for one
    set of conditions (month/day-of-week/holiday/temperature). Mirrors
    `resources.wind.api.WindProfile`'s shape so both resources' standalone results
    are used the same way.

    Attributes:
        hours: (24,) array, hour-of-day 0-23.
        load: (n_realizations, 24) array, MW or normalized per `units` at generation
            time.
        condition: the resolved condition (month, day-of-week, holiday, Tmax, Tmin).
        metadata: generation provenance (run_id, architecture, checkpoint path,
            device, wall-clock seconds, calibration applied).
    """

    hours: np.ndarray
    load: np.ndarray
    condition: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Tidy dataframe, one row per (realization, hour)."""
        n_realizations, n_hours = self.load.shape
        frames = [
            pd.DataFrame({"hour": self.hours, "realization": i, "load": self.load[i]})
            for i in range(n_realizations)
        ]
        return pd.concat(frames, ignore_index=True).set_index("hour")

    def mean(self) -> np.ndarray:
        """Load mean across realizations, shape (24,)."""
        return self.load.mean(axis=0)

    def percentile_envelope(self, percentiles: tuple[float, float] = (2.5, 97.5)) -> tuple[np.ndarray, np.ndarray]:
        """Load (lower, upper) percentile envelope across realizations, each shape (24,)."""
        lo, hi = percentiles
        return np.percentile(self.load, lo, axis=0), np.percentile(self.load, hi, axis=0)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_load_profile(
    zone: str = ZONE,
    *,
    month: int,
    Tmax: float,
    Tmin: float,
    day_of_week: int | str | None = None,
    weekday: bool = True,
    holiday: bool = False,
    spread: float | None = None,
    n_realizations: int = 100,
    run_id: str | None = None,
    seed: int | None = None,
    units: str = "MW",
    apply_calibration: bool | None = None,
    device: str | None = None,
) -> LoadProfile:
    """Generate `n_realizations` synthetic 24-hour PJM load profiles for one set of
    conditions, from a trained checkpoint — the standalone equivalent of
    `enerdatasynth.model("pjm_load", zone=zone).set(...).sample(...)`, usable directly
    without the fluent SDK layer (same relationship `generate_wind_profile` has to
    `enerdatasynth.model("wind", ...)`).

    `month`/`Tmax`/`Tmin` are required, matching `LOAD_CONDITION_SPEC`'s own
    `required=True` fields — there's no defensible default for them (see
    `docs/api_reference.md`'s "Not silently resolved" notes).

    `run_id`: which bundled checkpoint to use. `None` (default) resolves to
    `_discover_best_checkpoint()` — the same "best transformer, else best mlp"
    checkpoint `enerdatasynth.model("pjm_load", ...)` picks today (see
    `docs/research_status.md` §D for what that currently resolves to).

    Raises:
        CheckpointNotFoundError: no processed dataset or no trained checkpoint found.
        InvalidConditionError: an out-of-range `month`/`day_of_week`, or an
            unrecognized `units` value.
    """
    if units not in ("MW", "normalized"):
        raise InvalidConditionError(f"units must be 'MW' or 'normalized', got {units!r}")

    metadata_path = processed_metadata_path()
    if not metadata_path.exists():
        raise CheckpointNotFoundError(
            f"No scaler file (metadata.json) found at {metadata_path}. "
            f"Run `enerdatasynth preprocess` first (see docs/quickstart.md)."
        )
    with open(metadata_path) as f:
        metadata = json.load(f)

    start = time.time()
    resolved_run_id = run_id or _discover_best_checkpoint()
    model = LoadDiffusionModel.from_run_id(resolved_run_id, device=device)

    conditions: dict[str, Any] = {"month": month, "Tmax": Tmax, "Tmin": Tmin, "weekday": weekday, "holiday": holiday}
    if day_of_week is not None:
        conditions["day_of_week"] = day_of_week
    if spread is not None:
        conditions["spread"] = spread

    rng = np.random.default_rng(seed)
    try:
        z = build_condition_vector(LOAD_CONDITION_SPEC, conditions, metadata, n_realizations, rng)
    except ValueError as e:
        raise InvalidConditionError(str(e)) from e

    load = model.sample(z, seed=seed, apply_calibration=apply_calibration)
    if units == "MW":
        scaler = metadata["load_scaler"]
        data_min = np.array(scaler["data_min_"])
        data_max = np.array(scaler["data_max_"])
        load = load * (data_max - data_min) + data_min

    wall_time = time.time() - start
    result_meta = {
        "run_id": resolved_run_id,
        "architecture": model.architecture,
        "device": str(model.device),
        "wall_time_seconds": wall_time,
        "n_realizations": n_realizations,
        "seed": seed,
        "units": units,
        "calibration_applied": model.calibration is not None if apply_calibration is not False else False,
    }

    return LoadProfile(
        hours=np.arange(24),
        load=load,
        condition={"zone": zone, **conditions},
        metadata=result_meta,
    )


# ---------------------------------------------------------------------------
# Manifest lookups
# ---------------------------------------------------------------------------


def list_models() -> dict:
    """List every bundled pjm_load checkpoint with its architecture/hyperparameters/
    validation loss, from `resources/pjm_load/manifest.json` — real values aggregated
    from each checkpoint's own `result.json`, not fabricated (see that file's header).
    Uses `importlib.resources` (not a relative file path) so this resolves correctly
    from an installed wheel, same mechanism as `resources/wind/registry.py`."""
    text = importlib.resources.files("enerdatasynth.resources.pjm_load").joinpath("manifest.json").read_text(encoding="utf-8")
    return json.loads(text)


def describe_model(run_id: str) -> dict:
    """Describe a single checkpoint's manifest entry."""
    models = list_models()["checkpoints"]
    if run_id not in models:
        raise CheckpointNotFoundError(f"No checkpoint named {run_id!r}; known: {sorted(models)}")
    return models[run_id]
