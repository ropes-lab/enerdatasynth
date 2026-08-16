# `src/`

One installable package, `enerdatasynth`, generating two resources (PJM load and wind) through a shared interface. There's exactly one top-level package here (`enerdatasynth/`); wind's vendored pipeline lives nested inside it at [`enerdatasynth/resources/wind/`](enerdatasynth/resources/wind/README.md) rather than as a separate sibling package. A third resource would nest the same way, at `enerdatasynth/resources/<name>/`.

```
src/enerdatasynth/
├── sdk.py, cli.py, registry.py, conditioning.py, utils/, generators/   <- shared layer
└── resources/
    ├── pjm_load/   <- built for this project (data/models/diffusion/training/...)
    └── wind/       <- vendored from a separate Data-For-Good research project
```

This split exists because load and wind's underlying science and pipelines are genuinely different (see [`docs/architecture.md`](../docs/architecture.md) for the full comparison). Forcing wind's working, independently-tuned pipeline to share load's DDPM/Transformer/metrics code would mean rewriting something that already works, for no real benefit. Wind keeps its own API, config, and checkpoint-resolution logic inside `resources/wind/`, same as it would if it were still a standalone package. Nesting it here only changed its import path, from a separate top-level `wind_diffusion` package to `enerdatasynth.resources.wind` — not its design.

If you're just using the product, none of this should matter to you. It's an implementation detail behind `enerdatasynth.model(...)`, which looks and works the same for both resources. See [`src/enerdatasynth/README.md`](enerdatasynth/README.md) for the full package map.
