# `src/`

Two installable packages live here:

- **[`enerdatasynth/`](enerdatasynth/README.md)** — the product: the fluent
  `model()`/`.set()`/`.sample()` interface, both resources' generators, training/
  evaluation tooling, and the CLI. Start here.
- **[`wind_diffusion/`](wind_diffusion/README.md)** — a separately-developed diffusion
  pipeline (vendored from a Data-For-Good research project), wrapped by
  `enerdatasynth.generators.WindDiffusionModel` rather than reimplemented. Has its own
  API, config, and checkpoint-resolution logic, independent of `enerdatasynth`.

`enerdatasynth` depends on `wind_diffusion` (its wind resource wraps it); `wind_diffusion`
does not depend on `enerdatasynth` and would build and mean something on its own outside
this project. See [`docs/architecture.md`](../docs/architecture.md) for how the two
connect through the `BaseGenerator`/registry contract.
