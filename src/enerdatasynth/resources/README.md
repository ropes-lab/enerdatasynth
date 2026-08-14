# `enerdatasynth.resources`

Every resource this package generates lives here, one subpackage each:
[`pjm_load/`](pjm_load/README.md) and [`wind/`](wind/README.md). Both nest under the
same category taxonomy (`data/`, `models/`, `diffusion/`, `training/`, `evaluation/`,
`sampling/`, `visualization/`) — see [`docs/architecture.md`](../../../docs/architecture.md)
for why, and each subfolder's own README for what's genuinely shared in shape but
independently implemented underneath.

This package itself has no code — it's purely the container. The shared layer both
resources go through (`registry.py`, `conditioning.py`, `sdk.py`, `cli.py`,
`generators/`) lives one level up, in `src/enerdatasynth/` directly — see
[`../README.md`](../README.md).

## Adding a third resource

A new resource (e.g. water usage) nests here the same way: `resources/<name>/` with
whichever of the seven categories genuinely apply, plus its own `README.md`. See
[`docs/extending.md`](../../../docs/extending.md) for the full checklist — registering
it with `registry.register_resource(...)` is what makes `enerdatasynth.model("<name>",
...)` work, without touching `sdk.py` at all.
