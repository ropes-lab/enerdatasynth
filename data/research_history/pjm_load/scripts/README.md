# `data/research_history/pjm_load/scripts/`

The actual weekly research scripts — frozen, not maintained, not part of the package. See the [provenance table](../../README.md#provenance-table) in the parent README for what each script did, its status, and what in `src/enerdatasynth/` superseded it.

Don't run these as a way to use the package. Several hardcode a machine-specific path from the original research machine, and a `week3_outputs/`-relative fallback that no longer matches this repository's layout (that data now lives under `../outputs/week3_outputs/`). They're kept exactly as they were run, non-portability included, because they're historical evidence rather than tooling.

If you want to use the package: `import enerdatasynth`, or `enerdatasynth --help`. If you want to understand why it works the way it does: start at the parent [`research_history/README.md`](../../README.md), not these files directly.
