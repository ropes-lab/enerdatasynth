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

A second resource type, wind, is registered the same way:

    wind = enerdatasynth.model("wind", zone="3ch")   # zone = channel preset
    wind.set(wind_regime="mid", month=7, temp_min=10, temp_max=22)
    samples = wind.sample(n=100, units="m/s")        # (100, 288, 2): [...,0]=speed, [...,1]=direction

No trained wind checkpoint has been published yet (see
`generators/wind.py`), so `model("wind", ...)` raises
`wind_diffusion.exceptions.CheckpointNotFoundError` until one exists --
expected, not a wiring bug. Its conditioning shape didn't fit the generic
`ConditionField`-based mechanism `_REGISTRY["pjm_load"]` uses (dynamic
field presence, a derived+jointly-scaled field, checkpoint-level onehot-
vs-ordinal encoding) -- `ResourceModel` supports an optional
`condition_builder` escape hatch for exactly this case; see
`generators/wind_conditioning.py`.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np

from enerdatasynth import conditioning
from enerdatasynth.utils.config import PROCESSED_DIR, CHECKPOINT_DIR
from enerdatasynth.generators import LoadDiffusionModel, WindDiffusionModel, BaseGenerator
from enerdatasynth.generators.load_conditioning import LOAD_CONDITION_SPEC, LOAD_EXTRA_SET_KEYS
from enerdatasynth.generators.wind_conditioning import WIND_EXTRA_SET_KEYS, wind_checkpoint_status, wind_construct


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

    Load-specific: scans CHECKPOINT_DIR for result.json files, a
    convention only enerdatasynth's own training pipeline produces. Wind's
    equivalent (`generators/wind_conditioning.py`'s `wind_construct`)
    resolves checkpoints through `wind_diffusion.registry` instead --
    a different mechanism entirely, not something this function generalizes.
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


def _load_construct(zone: str) -> dict:
    """`model("pjm_load", zone=...)`'s construction hook (see `_REGISTRY`
    below) -- the exact logic `model()` used to run inline before
    construction became resource-delegated, moved here unchanged so wind
    (`wind_conditioning.wind_construct`) can provide its own without
    `model()` needing an if/else per resource."""
    metadata_path = PROCESSED_DIR / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No processed dataset found at {metadata_path}. "
            f"Run `enerdatasynth preprocess` first (see docs/quickstart.md)."
        )
    with open(metadata_path) as f:
        metadata = json.load(f)

    run_id = _discover_best_checkpoint()
    generator = LoadDiffusionModel.from_run_id(run_id)
    return {
        "generator": generator,
        "metadata": metadata,
        "condition_spec": LOAD_CONDITION_SPEC,
        "extra_set_keys": LOAD_EXTRA_SET_KEYS,
    }


def _load_checkpoint_status() -> dict:
    """`describe_resource("pjm_load")`'s checkpoint-availability hook."""
    status = {"processed_dataset_available": (PROCESSED_DIR / "metadata.json").exists()}
    try:
        status["best_run_id"] = _discover_best_checkpoint()
        status["checkpoint_available"] = True
    except FileNotFoundError:
        status["best_run_id"] = None
        status["checkpoint_available"] = False
    return status


# Resource types this SDK currently knows how to serve. Each entry's
# `construct(zone) -> dict` is the one thing that actually varies enough
# between resources that ResourceModel can't build it generically --
# where checkpoints/metadata live and how they're resolved (load: a local
# CHECKPOINT_DIR scan + PROCESSED_DIR/metadata.json; wind: wind_diffusion's
# own sha256-verified registry, keyed by preset) -- everything downstream
# of that (conditioning vector construction, units/denormalization) is
# generic, driven by whatever `construct` returns. See `_load_construct`
# above and `generators/wind_conditioning.py`'s `wind_construct` for the
# two reference implementations. Zones are constrained to what each
# resource's pipeline actually supports; see MIGRATION.md for why
# "pjm_load" is PS-only today even though
# research_history/scripts/zone_expansion.py explored 17 zones. Wind
# repurposes "zone" as its channel preset ("1ch"/"2ch"/"3ch") -- see
# wind_construct's docstring for why.
_REGISTRY = {
    "pjm_load": {
        "generator_cls": LoadDiffusionModel,
        "zones": ["PS"],
        "condition_spec": LOAD_CONDITION_SPEC,
        "extra_set_keys": LOAD_EXTRA_SET_KEYS,
        "construct": _load_construct,
        "checkpoint_status": _load_checkpoint_status,
    },
    "wind": {
        "generator_cls": WindDiffusionModel,
        "zones": ["1ch", "2ch", "3ch"],
        "condition_spec": [],
        "extra_set_keys": WIND_EXTRA_SET_KEYS,
        "construct": wind_construct,
        "checkpoint_status": wind_checkpoint_status,
    },
}


class ResourceModel:
    """
    A configured, ready-to-sample generator for one resource type and
    zone. Returned by `enerdatasynth.model(...)` — not constructed
    directly.

    Generic across resource types: `.set()`/`.sample()` build the raw
    conditioning vector via whichever `condition_spec` (or, for a resource
    whose conditioning shape doesn't fit that model, `condition_builder`
    override -- see `generators/wind_conditioning.py`) `_REGISTRY[name]`
    provides. This class itself no longer hardcodes any single resource's
    field names.

    `.set()` mutates state on the instance and accumulates silently across
    calls -- a real footgun for a long-lived instance reused across
    requests. `.reset()` clears it. `.sample()`'s default return is a bare
    array (matching this SDK's existing minimalism), but `with_metadata=
    True` (or `.last_metadata()` afterward, regardless of that flag)
    recovers what actually produced it -- the resolved `n`/`seed`/`units`
    and the conditions in effect at that call.
    """

    def __init__(
        self, name: str, zone: str, generator: BaseGenerator, metadata: dict,
        condition_spec: list[conditioning.ConditionField], extra_set_keys: Optional[list[str]] = None,
        condition_builder=None, valid_units: tuple = ("MW", "normalized"), postprocess=None,
    ):
        self._name = name
        self._zone = zone
        self._generator = generator
        self._metadata = metadata
        self._condition_spec = condition_spec
        self._condition_builder = condition_builder
        self._valid_set_keys = {f.name for f in condition_spec} | set(extra_set_keys or [])
        self._valid_units = tuple(valid_units)
        self._postprocess = postprocess
        self._conditions: dict = {}
        self._last_metadata: Optional[dict] = None

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

    def reset(self) -> "ResourceModel":
        """
        Clear every condition set via `.set()` so far. Chainable — returns
        self.

        `.set()` mutates state on the instance and accumulates silently
        across calls, which is a real footgun for a long-lived
        `ResourceModel` reused across requests (a value set three calls
        ago can still be in effect without anything visibly wrong at the
        call site that forgot it). `.reset()` is the escape hatch: call it
        before reconfiguring from scratch instead of constructing a new
        `enerdatasynth.model(...)` instance just to clear state.
        """
        self._conditions.clear()
        return self

    def last_metadata(self) -> Optional[dict]:
        """
        Provenance for the most recent `.sample()` call on this instance
        (`None` if `.sample()` hasn't been called yet): the resolved `n`,
        `seed`, `units`, and a snapshot of the conditions in effect at
        that call. Recoverable regardless of whether `with_metadata=True`
        was passed to that particular `.sample()` call -- provenance is
        never silently lost, it's just not forced into `.sample()`'s
        default return shape (a bare array, matching this SDK's existing
        minimalism) unless you ask for it inline.
        """
        return self._last_metadata

    def _missing_required(self) -> list:
        if self._condition_builder is not None:
            return []  # wind (and any condition_builder resource): nothing hard-required today
        return conditioning.missing_required(self._condition_spec, self._conditions)

    def _build_conditioning(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self._condition_builder is not None:
            return self._condition_builder(self._conditions, self._metadata, n, rng)
        return conditioning.build_condition_vector(
            self._condition_spec, self._conditions, self._metadata, n, rng,
        )

    def _denormalize(self, X_normalized: np.ndarray) -> np.ndarray:
        """Load's default `postprocess`-equivalent when a resource doesn't
        provide its own (see `sample()`): a MinMax denormalization against
        `self._metadata["load_scaler"]` -- load-specific, only ever
        reached when `self._postprocess is None`, which today is only
        true for `"pjm_load"`."""
        data_min = np.asarray(self._metadata["load_scaler"]["data_min_"])
        data_max = np.asarray(self._metadata["load_scaler"]["data_max_"])
        return X_normalized * (data_max - data_min) + data_min

    def sample(
        self, n: int = 1, seed: Optional[int] = None, units: Optional[str] = None,
        with_metadata: bool = False,
    ):
        """
        Generate `n` synthetic daily profiles under the currently-set
        conditions (call `.set()` first).

        units : defaults to the resource's first valid unit if omitted
                (`"MW"` for `"pjm_load"`, `"m/s"` for `"wind"`) -- see
                `enerdatasynth.describe_resource(name)` to check which
                units a given resource actually supports; raises
                `ValueError` for anything else.
        with_metadata : False (default) returns a bare array, matching
                this SDK's existing minimalism. True returns
                `(array, metadata)` instead. Either way, `.last_metadata()`
                recovers the same dict afterward -- provenance (the
                resolved `n`/`seed`/`units` and the conditions actually in
                effect) is never silently lost, just not forced into the
                default return shape.
        """
        units = self._valid_units[0] if units is None else units
        if units not in self._valid_units:
            raise ValueError(f"units must be one of {self._valid_units}, got {units!r}")

        rng = np.random.default_rng(seed)
        z = self._build_conditioning(n, rng)
        samples = self._generator.sample(z, seed=seed)

        if self._postprocess is not None:
            result = self._postprocess(samples, units)
        elif units == "normalized":
            result = samples
        else:
            result = self._denormalize(samples)

        self._last_metadata = {
            "name": self._name, "zone": self._zone,
            "n": n, "seed": seed, "units": units,
            "conditions": dict(self._conditions),
        }
        return (result, self._last_metadata) if with_metadata else result


def list_resources() -> dict:
    """List every resource type this SDK knows how to serve, with its
    supported zones and current checkpoint availability -- lets a caller
    discover what's registered without reading `_REGISTRY` in source.
    Mirrors `wind_diffusion`'s `list_models()`/`describe_model()` pattern.
    """
    return {name: describe_resource(name) for name in _REGISTRY}


def describe_resource(name: str) -> dict:
    """Describe one registered resource type: supported zones, its
    conditioning fields (the keys `.set()` accepts), and whether a trained
    checkpoint is currently available. Unlike `model(name)`, never raises
    for a missing checkpoint/processed dataset -- reports their absence
    instead, so this is safe to call for discovery before attempting to
    actually construct a `ResourceModel`.
    """
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(_REGISTRY)}")

    entry = _REGISTRY[name]
    condition_fields = sorted(
        {f.name for f in entry["condition_spec"]} | set(entry.get("extra_set_keys") or [])
    )
    info: dict = {
        "zones": entry["zones"],
        "domain": entry["generator_cls"].domain,
        "condition_fields": condition_fields,
    }
    info.update(entry["checkpoint_status"]())
    return info


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

    built = entry["construct"](zone)
    return ResourceModel(
        name=name, zone=zone,
        generator=built["generator"], metadata=built["metadata"],
        condition_spec=built["condition_spec"], extra_set_keys=built.get("extra_set_keys"),
        condition_builder=built.get("condition_builder"),
        valid_units=built.get("valid_units", ("MW", "normalized")),
        postprocess=built.get("postprocess"),
    )
