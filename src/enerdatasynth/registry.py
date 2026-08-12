"""
src/enerdatasynth/registry.py
================================
The typed contract a resource type (electric load, wind, ...) uses to
register itself with `enerdatasynth.model(...)` / `list_resources()` /
`describe_resource()` (`sdk.py`).

Previously this was a single `_REGISTRY = {...}` dict literal hardcoded
in `sdk.py`, with both resources' entries written inline there -- adding
a resource meant editing `sdk.py` itself, and each entry's required keys
were documented only in prose (a missing/misspelled key surfaced as a
silent `None` from `.get(...)`, not an error at registration time).

Now: a resource module calls `register_resource(ResourceEntry(...))`
once, at import time, to add itself here. `sdk.py` only ever reads from
`get_registry()` -- it doesn't know, and doesn't need to know, how many
resource types exist or what they're called. See
`generators/load_conditioning.py` and `generators/wind_conditioning.py`
for the two reference registrations, and `generators/__init__.py` for
where those modules get imported (the one place a new resource still
needs a line added, so its registration side effect actually runs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

from enerdatasynth.conditioning import ConditionField

if TYPE_CHECKING:
    # Only needed for type hints below (`from __future__ import
    # annotations` already defers their evaluation) -- a real runtime
    # import here would be circular: `generators/__init__.py` imports
    # `generators.load_conditioning`/`generators.wind_conditioning`,
    # which import THIS module, so `enerdatasynth.generators` can't be
    # a prerequisite of importing `enerdatasynth.registry`.
    from enerdatasynth.generators.base import BaseGenerator


@dataclass(frozen=True)
class ConstructResult:
    """What a resource's `construct(zone) -> ConstructResult` hook
    returns -- everything `ResourceModel.__init__` needs. Replaces the
    previous raw `dict` return, where a missing/misspelled key (e.g.
    `built.get("condition_builder")`) failed silently or late instead of
    at construction time.

    condition_spec / extra_set_keys / condition_builder / valid_units /
    postprocess mirror `ResourceModel.__init__`'s own parameters exactly
    -- see `sdk.py` for what each one means.
    """

    generator: BaseGenerator
    metadata: dict
    condition_spec: list[ConditionField] = field(default_factory=list)
    extra_set_keys: list[str] = field(default_factory=list)
    condition_builder: Optional[Callable[[dict, dict, int, np.random.Generator], np.ndarray]] = None
    valid_units: tuple = ("MW", "normalized")
    postprocess: Optional[Callable[[np.ndarray, str], np.ndarray]] = None


@dataclass(frozen=True)
class ResourceEntry:
    """One resource type's registration: everything `enerdatasynth.
    model()` / `list_resources()` / `describe_resource()` need,
    independent of any particular zone. `condition_spec`/`extra_set_keys`
    are duplicated here (rather than only inside `ConstructResult`) so
    `describe_resource()` can list a resource's `.set()`-accepted keys
    without constructing a full generator or touching a checkpoint.
    """

    name: str
    generator_cls: type[BaseGenerator]
    zones: list[str]
    condition_spec: list[ConditionField]
    extra_set_keys: list[str]
    construct: Callable[[str], ConstructResult]
    checkpoint_status: Callable[[], dict]


_REGISTRY: dict[str, ResourceEntry] = {}


def register_resource(entry: ResourceEntry, *, replace: bool = False) -> None:
    """Add a resource type to the shared registry `sdk.py` reads from.
    Called once, at import time, by each resource's own conditioning
    module -- never by `sdk.py` itself, which needs no per-resource edit.

    Raises `ValueError` on a duplicate name unless `replace=True`, so a
    copy-pasted resource module that forgot to change its registered
    name fails loudly at import time instead of silently shadowing an
    existing resource depending on import order.
    """
    if entry.name in _REGISTRY and not replace:
        raise ValueError(
            f"Resource {entry.name!r} is already registered "
            f"(by {_REGISTRY[entry.name].generator_cls.__name__}); "
            f"pass replace=True if this is intentional."
        )
    _REGISTRY[entry.name] = entry


def get_registry() -> dict[str, ResourceEntry]:
    """The live, shared registry -- `sdk.py` reads from this instead of
    owning a `_REGISTRY` literal itself."""
    return _REGISTRY
