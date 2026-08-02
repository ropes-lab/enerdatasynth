"""
src/enerdatasynth/sdk.py
==========================
The package's primary, implementation-hiding public interface:

    import enerdatasynth

    load = enerdatasynth.model("pjm_load", zone="PS")
    load.set(month=7, Tmax=86, Tmin=70, weekday=True, holiday=False)
    samples = load.sample(n=100)

`ResourceModel` never exposes tensors, architecture classes, DDPM
internals, or checkpoint paths — callers work entirely in terms of
calendar/weather conditions and get back MW-scale profiles. See
`docs/api_reference.md` for the full contract, and the "Not silently
resolved" notes on individual `set()` parameters below for the
specific UX decisions this required where no single "correct" answer
exists (weekday sampling semantics, the unavoidable `spread` default).

Adding a new resource type (e.g. wind) means registering a new entry
in `_REGISTRY` pointing at a `BaseGenerator` subclass and its own
conditioning-building logic — `model("wind")` would work through this
exact same interface once that exists. Nothing wind-specific is
implemented here; see `docs/extending.md`.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from enerdatasynth.utils.config import PROCESSED_DIR, CHECKPOINT_DIR, N_MONTHS, N_DOW
from enerdatasynth.generators import LoadDiffusionModel, BaseGenerator

_DOW_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_WEEKDAY_INDICES = [0, 1, 2, 3, 4]
_WEEKEND_INDICES = [5, 6]

# Resource types this SDK currently knows how to serve. Adding a new
# one (e.g. "wind") means adding an entry here once a real generator
# for it exists — see docs/extending.md. Zones are constrained to what
# the underlying preprocessing pipeline actually supports; see
# MIGRATION.md for why this is PS-only today (config.ZONE) even though
# research_history/scripts/zone_expansion.py explored 17 zones.
_REGISTRY = {
    "pjm_load": {
        "generator_cls": LoadDiffusionModel,
        "zones": ["PS"],
    },
}


def _discover_best_checkpoint(checkpoint_root: Path = CHECKPOINT_DIR) -> str:
    """
    Find the best available trained checkpoint: prefers any
    architecture="transformer" run (lowest best_val_loss among those),
    since that's the validated-best architecture (see MIGRATION.md);
    falls back to the best "mlp" run if no transformer checkpoint has
    been trained yet against the current preprocessing pipeline. As of
    this package's last audit, no transformer checkpoint compatible
    with the current one-hot conditioning existed — so today this
    resolves to the best legacy MLP run, and will silently start
    resolving to a transformer run automatically once one is trained
    (`enerdatasynth train`), with no code changes needed here.
    """
    checkpoint_root = Path(checkpoint_root)
    candidates = []
    if checkpoint_root.exists():
        for run_dir in sorted(checkpoint_root.iterdir()):
            result_path = run_dir / "result.json"
            if not result_path.exists():
                continue
            with open(result_path) as f:
                candidates.append(json.load(f))

    if not candidates:
        raise FileNotFoundError(
            f"No trained checkpoints found under {checkpoint_root}. "
            f"Run `enerdatasynth train` first (see docs/quickstart.md)."
        )

    transformer_runs = [r for r in candidates if r.get("architecture") == "transformer"]
    pool = transformer_runs if transformer_runs else candidates
    best = min(pool, key=lambda r: r["best_val_loss"])
    return best["run_id"]


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


class ResourceModel:
    """
    A configured, ready-to-sample generator for one resource type and
    zone. Returned by `enerdatasynth.model(...)` — not constructed
    directly.
    """

    def __init__(self, name: str, zone: str, generator: BaseGenerator, metadata: dict):
        self._name = name
        self._zone = zone
        self._generator = generator
        self._metadata = metadata
        self._conditions: dict = {}

    def __repr__(self) -> str:
        return f"<ResourceModel {self._name!r} zone={self._zone!r} domain={self._generator.domain!r}>"

    def set(self, *, month: Optional[int] = None,
            Tmax: Optional[float] = None, Tmin: Optional[float] = None,
            weekday: Optional[bool] = None, holiday: Optional[bool] = None,
            day_of_week: Optional[Union[int, str]] = None,
            spread: Optional[float] = None) -> "ResourceModel":
        """
        Set conditioning variables. Chainable — returns self. Only
        the keys you pass are updated; previously-set values persist
        across calls, so you can build up conditions incrementally.

        month       : 1-12 (calendar month). Required before sample().
        Tmax, Tmin  : daily max/min temperature, in degrees Fahrenheit
                      — the same units as the training data
                      (`research_history/`'s NOAA temperature pipeline
                      converts to °F; this SDK does not silently
                      convert units, to avoid guessing wrong). Required
                      before sample().
        weekday     : True samples a uniformly random weekday (Mon-Fri)
                      independently for each generated profile; False
                      samples a random weekend day (Sat-Sun). Default:
                      True. This is a genuine UX decision, not a
                      validated finding — the model's actual
                      conditioning is a specific day-of-week, not a
                      "weekday" category, and there's no single
                      "correct" weekday to pick, so this samples one
                      per profile instead of arbitrarily fixing e.g.
                      Wednesday for every sample. Use `day_of_week`
                      for exact control.
        day_of_week : Exact override, 0=Monday..6=Sunday, or a name
                      ("Monday", "Mon", ...). Takes precedence over
                      `weekday` when both are set.
        holiday     : US federal holiday flag. Default: False.
        spread      : Daily load range (max MW - min MW), which the
                      model was also conditioned on during training.
                      Not something a caller predicting from weather
                      alone would actually know in advance — defaults
                      to the training set's mean spread (a "typical
                      variability" day) if not given. This is a
                      documented simplification, not a validated
                      estimate of what spread a given weather condition
                      implies; see docs/api_reference.md.
        """
        updates = dict(month=month, Tmax=Tmax, Tmin=Tmin, weekday=weekday,
                        holiday=holiday, day_of_week=day_of_week, spread=spread)
        for key, value in updates.items():
            if value is not None:
                self._conditions[key] = value
        return self

    def _missing_required(self) -> list:
        return [k for k in ("month", "Tmax", "Tmin") if k not in self._conditions]

    def _build_conditioning(self, n: int, rng: np.random.Generator) -> np.ndarray:
        missing = self._missing_required()
        if missing:
            raise ValueError(
                f"Missing required condition(s) before sample(): {missing}. "
                f"Call .set({', '.join(f'{m}=...' for m in missing)}) first."
            )

        temp_scaler = self._metadata["temp_scaler"]
        spread_scaler = self._metadata["spread_scaler"]
        tmax_mean, tmin_mean = temp_scaler["mean_"]
        tmax_scale, tmin_scale = temp_scaler["scale_"]
        spread_mean, spread_scale = spread_scaler["mean_"][0], spread_scaler["scale_"][0]

        month = self._conditions["month"]
        if not (1 <= month <= 12):
            raise ValueError(f"month must be 1-12, got {month}")
        Tmax_z = (self._conditions["Tmax"] - tmax_mean) / tmax_scale
        Tmin_z = (self._conditions["Tmin"] - tmin_mean) / tmin_scale
        holiday_flag = float(self._conditions.get("holiday", False))

        if "spread" in self._conditions:
            spread_z = (self._conditions["spread"] - spread_mean) / spread_scale
        else:
            spread_z = 0.0  # dataset-mean spread; see set()'s docstring

        if "day_of_week" in self._conditions:
            dow_choices = np.full(n, _parse_day_of_week(self._conditions["day_of_week"]))
        else:
            is_weekday = self._conditions.get("weekday", True)
            pool = _WEEKDAY_INDICES if is_weekday else _WEEKEND_INDICES
            dow_choices = rng.choice(pool, size=n)

        Z = np.zeros((n, N_MONTHS + N_DOW + 1 + 1 + 1 + 1), dtype=np.float32)
        Z[:, month - 1] = 1.0
        for i, dow in enumerate(dow_choices):
            Z[i, N_MONTHS + dow] = 1.0
        Z[:, N_MONTHS + N_DOW] = holiday_flag
        Z[:, N_MONTHS + N_DOW + 1] = Tmax_z
        Z[:, N_MONTHS + N_DOW + 2] = Tmin_z
        Z[:, N_MONTHS + N_DOW + 3] = spread_z
        return Z

    def _denormalize(self, X_normalized: np.ndarray) -> np.ndarray:
        data_min = np.asarray(self._metadata["load_scaler"]["data_min_"])
        data_max = np.asarray(self._metadata["load_scaler"]["data_max_"])
        return X_normalized * (data_max - data_min) + data_min

    def sample(self, n: int = 1, seed: Optional[int] = None, units: str = "MW") -> np.ndarray:
        """
        Generate `n` synthetic daily profiles under the currently-set
        conditions (call `.set()` first).

        units : "MW" (default) — denormalized to real load units using
                the scaler fit during preprocessing — or "normalized"
                for the raw [0,1] model output.
        """
        if units not in ("MW", "normalized"):
            raise ValueError(f"units must be 'MW' or 'normalized', got {units!r}")

        rng = np.random.default_rng(seed)
        z = self._build_conditioning(n, rng)
        samples = self._generator.sample(z, seed=seed)

        if units == "normalized":
            return samples
        return self._denormalize(samples)


def model(name: str, zone: Optional[str] = None) -> ResourceModel:
    """
    Construct a ready-to-configure generator for a resource type.

        load = enerdatasynth.model("pjm_load", zone="PS")

    Currently supported: `name="pjm_load"`, `zone="PS"` (the only zone
    the shipped preprocessing/training pipeline currently produces
    data for — see MIGRATION.md for what multi-zone support would
    require). Automatically uses the best available trained
    checkpoint; no architecture, checkpoint path, or run_id is exposed.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(_REGISTRY)}")

    entry = _REGISTRY[name]
    supported_zones = entry["zones"]
    if zone is None:
        zone = supported_zones[0]
    if zone not in supported_zones:
        raise ValueError(
            f"Zone {zone!r} is not supported for model {name!r}. "
            f"Supported zones: {supported_zones}. "
            f"See research_history/scripts/zone_expansion.py for exploratory "
            f"multi-zone work not yet promoted into this package."
        )

    metadata_path = PROCESSED_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No processed dataset found at {metadata_path}. "
            f"Run `enerdatasynth preprocess` first (see docs/quickstart.md)."
        )
    with open(metadata_path) as f:
        metadata = json.load(f)

    run_id = _discover_best_checkpoint()
    generator = entry["generator_cls"].from_run_id(run_id)

    return ResourceModel(name=name, zone=zone, generator=generator, metadata=metadata)
