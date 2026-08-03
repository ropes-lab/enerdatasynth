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
from typing import Optional

import numpy as np

from enerdatasynth import conditioning
from enerdatasynth.utils.config import PROCESSED_DIR, CHECKPOINT_DIR
from enerdatasynth.generators import LoadDiffusionModel, BaseGenerator
from enerdatasynth.generators.load_conditioning import LOAD_CONDITION_SPEC, LOAD_EXTRA_SET_KEYS

# Resource types this SDK currently knows how to serve. Adding a new one
# (e.g. "wind") means adding an entry here once a real generator for it
# exists -- see docs/extending.md -- plus its own `<domain>_conditioning.py`
# module providing `condition_spec` (see `generators/load_conditioning.py`
# for the reference implementation). `ResourceModel` reads `condition_spec`
# generically (via `conditioning.py`) rather than hardcoding any one
# resource's field names, so a second entry here is genuinely additive --
# no changes needed to ResourceModel itself. Zones are constrained to what
# the underlying preprocessing pipeline actually supports; see
# MIGRATION.md for why this is PS-only today (config.ZONE) even though
# research_history/scripts/zone_expansion.py explored 17 zones.
_REGISTRY = {
    "pjm_load": {
        "generator_cls": LoadDiffusionModel,
        "zones": ["PS"],
        "condition_spec": LOAD_CONDITION_SPEC,
        "extra_set_keys": LOAD_EXTRA_SET_KEYS,
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


class ResourceModel:
    """
    A configured, ready-to-sample generator for one resource type and
    zone. Returned by `enerdatasynth.model(...)` — not constructed
    directly.

    Generic across resource types: `.set()`/`.sample()` build the raw
    conditioning vector via `conditioning.build_condition_vector()` against
    whichever `condition_spec` `_REGISTRY[name]` provides (see
    `generators/load_conditioning.py`'s `LOAD_CONDITION_SPEC` for the
    reference implementation and the per-field UX notes -- e.g. why
    `weekday` samples a different day per realization -- that used to live
    directly on this class's `.set()` docstring). This class itself no
    longer hardcodes any single resource's field names.
    """

    def __init__(
        self, name: str, zone: str, generator: BaseGenerator, metadata: dict,
        condition_spec: list[conditioning.ConditionField], extra_set_keys: Optional[list[str]] = None,
    ):
        self._name = name
        self._zone = zone
        self._generator = generator
        self._metadata = metadata
        self._condition_spec = condition_spec
        self._valid_set_keys = {f.name for f in condition_spec} | set(extra_set_keys or [])
        self._conditions: dict = {}

    def __repr__(self) -> str:
        return f"<ResourceModel {self._name!r} zone={self._zone!r} domain={self._generator.domain!r}>"

    def set(self, **kwargs) -> "ResourceModel":
        """
        Set conditioning variables. Chainable — returns self. Only the
        keys you pass are updated; previously-set values persist across
        calls, so you can build up conditions incrementally.

        Accepted keys depend on the resource (`enerdatasynth.model(name,
        ...)`'s `name`) — see `docs/api_reference.md` for `"pjm_load"`'s
        full parameter reference (`month`, `Tmax`, `Tmin`, `weekday`,
        `day_of_week`, `holiday`, `spread`) including the UX notes behind
        each default. Unknown keys for the current resource raise
        `ValueError` immediately rather than being silently ignored.
        """
        unknown = set(kwargs) - self._valid_set_keys
        if unknown:
            raise ValueError(
                f"Unknown condition(s) for {self._name!r}: {sorted(unknown)}. "
                f"Valid keys: {sorted(self._valid_set_keys)}."
            )
        for key, value in kwargs.items():
            if value is not None:
                self._conditions[key] = value
        return self

    def _missing_required(self) -> list:
        return conditioning.missing_required(self._condition_spec, self._conditions)

    def _build_conditioning(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return conditioning.build_condition_vector(
            self._condition_spec, self._conditions, self._metadata, n, rng,
        )

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

    return ResourceModel(
        name=name, zone=zone, generator=generator, metadata=metadata,
        condition_spec=entry["condition_spec"], extra_set_keys=entry.get("extra_set_keys"),
    )
