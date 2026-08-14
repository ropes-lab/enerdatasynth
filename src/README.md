# `src/`

**One product, `enerdatasynth`, generating two resources — PJM load and wind — through
one shared interface.** Underneath that interface, two packages actually live here:

- **[`enerdatasynth/`](enerdatasynth/README.md)** — the product itself: the fluent
  `model()`/`.set()`/`.sample()` interface, the shared generator contract
  (`generators/`), PJM load's full from-scratch pipeline (data/models/diffusion/
  training/evaluation), and the CLI. Start here.
- **[`wind_diffusion/`](wind_diffusion/README.md)** — wind's generative pipeline,
  vendored from a separately-developed Data-For-Good research project rather than
  rebuilt from scratch, and wrapped by `enerdatasynth.generators.WindDiffusionModel` so
  it's reachable through the same `enerdatasynth.model("wind", ...)` call as load. Has
  its own independent API, config, and checkpoint-resolution logic.

This split exists because the two resources' underlying science and pipelines are
genuinely different (see [`docs/architecture.md`](../docs/architecture.md) for the full
comparison) — forcing wind's working, independently-tuned pipeline into
`enerdatasynth`'s load-specific code would mean rewriting it for no scientific benefit.
`enerdatasynth` depends on `wind_diffusion` (its wind resource wraps it); `wind_diffusion`
does not depend on `enerdatasynth` and means something on its own outside this project.
If you're using the product, you shouldn't need to care about this split at all —
`enerdatasynth.model(...)` looks and works identically for both resources.
