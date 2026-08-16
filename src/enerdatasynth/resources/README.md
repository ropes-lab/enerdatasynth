# `enerdatasynth.resources`

Every resource this package generates lives here, one subpackage each: [`pjm_load/`](pjm_load/README.md) and [`wind/`](wind/README.md). Both follow the same category taxonomy (`data/`, `models/`, `diffusion/`, `training/`, `evaluation/`, `sampling/`, `visualization/`) — see [`docs/architecture.md`](../../../docs/architecture.md) for why, and each subfolder's own README for what's actually shared versus independently implemented.

This package has no code of its own; it's just the container. The shared layer both resources go through (`registry.py`, `conditioning.py`, `sdk.py`, `cli.py`, `generators/`) lives one level up, in `src/enerdatasynth/` directly. See [`../README.md`](../README.md).

## Adding a third resource

A new resource, water usage say, nests here the same way: `resources/<name>/` with whichever of the seven categories apply, plus its own `README.md`. `docs/extending.md` has the full checklist — registering it with `registry.register_resource(...)` is what makes `enerdatasynth.model("<name>", ...)` work, without touching `sdk.py` at all.
