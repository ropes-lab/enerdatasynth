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

Real trained checkpoints (1ch/2ch/3ch) are bundled and used
automatically -- see `generators/wind_conditioning.py`'s `wind_construct`.
Wind's conditioning shape didn't fit the generic `ConditionField`-based
mechanism `"pjm_load"` uses (dynamic field presence, a derived+jointly-
scaled field, checkpoint-level onehot-vs-ordinal encoding) --
`ResourceModel` supports an optional `condition_builder` escape hatch
for exactly this case; see `generators/wind_conditioning.py`.

Neither resource is wired in here directly: this module only knows about
`registry.get_registry()`, populated by each resource's own conditioning
module (`generators/load_conditioning.py`, `generators/wind_conditioning.py`)
calling `registry.register_resource(...)` at import time. Adding a third
resource never requires editing this file -- see `registry.py`'s module
docstring.
"""

from __future__ import annotations
from typing import Optional

import numpy as np

from enerdatasynth import conditioning, registry
from enerdatasynth.generators import BaseGenerator


class ResourceModel:
    """
    A configured, ready-to-sample generator for one resource type and
    zone. Returned by `enerdatasynth.model(...)` — not constructed
    directly.

    Generic across resource types: `.set()`/`.sample()` build the raw
    conditioning vector via whichever `condition_spec` (or, for a resource
    whose conditioning shape doesn't fit that model, `condition_builder`
    override -- see `generators/wind_conditioning.py`) that resource's
    `registry.ResourceEntry` provides. This class itself no longer
    hardcodes any single resource's field names.

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
    discover what's registered without reading `registry.py` in source.
    Mirrors `wind_diffusion`'s `list_models()`/`describe_model()` pattern.
    """
    reg = registry.get_registry()
    return {name: describe_resource(name) for name in reg}


def describe_resource(name: str) -> dict:
    """Describe one registered resource type: supported zones, its
    conditioning fields (the keys `.set()` accepts), and whether a trained
    checkpoint is currently available. Unlike `model(name)`, never raises
    for a missing checkpoint/processed dataset -- reports their absence
    instead, so this is safe to call for discovery before attempting to
    actually construct a `ResourceModel`.
    """
    reg = registry.get_registry()
    if name not in reg:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(reg)}")

    entry = reg[name]
    condition_fields = sorted(
        {f.name for f in entry.condition_spec} | set(entry.extra_set_keys)
    )
    info: dict = {
        "zones": entry.zones,
        "domain": entry.generator_cls.domain,
        "condition_fields": condition_fields,
    }
    info.update(entry.checkpoint_status())
    return info


def model(name: str, zone: Optional[str] = None) -> ResourceModel:
    """
    Construct a ready-to-configure generator for a resource type.

        load = enerdatasynth.model("pjm_load", zone="PS")

    Currently supported: `name="pjm_load"` for any of the 17 PJM zones
    (`zone="PS"` is the default; the others are AE, AP, BC, CE, DAY, DEOK,
    DOM, DPL, DUQ, EKPC, JC, ME, PE, PL, PN and RECO, each with its own trained
    model); `name="wind"`, `zone="1ch"|"2ch"|"3ch"`. Zone names are not
    case-sensitive. Automatically uses the best available trained checkpoint;
    no architecture, checkpoint path, or run_id is exposed. See
    `list_resources()` / `describe_resource(name)` to discover what's registered.
    """
    reg = registry.get_registry()
    if name not in reg:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(reg)}")

    entry = reg[name]
    supported_zones = entry.zones
    if zone is None:
        zone = supported_zones[0]
    matches = [z for z in supported_zones if z.lower() == str(zone).lower()]
    if not matches:
        raise ValueError(
            f"Zone {zone!r} is not supported for model {name!r}. "
            f"Supported zones: {supported_zones}."
        )
    zone = matches[0]

    built = entry.construct(zone)  # registry.ConstructResult
    return ResourceModel(
        name=name, zone=zone,
        generator=built.generator, metadata=built.metadata,
        condition_spec=built.condition_spec, extra_set_keys=built.extra_set_keys,
        condition_builder=built.condition_builder,
        valid_units=built.valid_units,
        postprocess=built.postprocess,
    )
