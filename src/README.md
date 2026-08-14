# `src/`

One installable package, `enerdatasynth`, generating two resources — PJM load and wind
— through one shared interface. There is exactly one top-level package here now
(`enerdatasynth/`); wind's vendored pipeline lives nested inside it at
[`enerdatasynth/resources/wind/`](enerdatasynth/resources/wind/README.md), not as a
separate sibling package — a third resource would nest the same way, at
`enerdatasynth/resources/<name>/`.

```
src/enerdatasynth/
├── sdk.py, cli.py, registry.py, conditioning.py, utils/, generators/   <- shared layer
└── resources/
    ├── pjm_load/   <- built for this project (data/models/diffusion/training/...)
    └── wind/       <- vendored from a separate Data-For-Good research project
```

This split exists because the two resources' underlying science and pipelines are
genuinely different (see [`docs/architecture.md`](../docs/architecture.md) for the full
comparison) — forcing wind's working, independently-tuned pipeline to share load's
DDPM/Transformer/metrics code would mean rewriting it for no scientific benefit. Wind
keeps its own independent API, config, and checkpoint-resolution logic inside its
`resources/wind/` folder, same as it would as a standalone package — nesting it here
changed its import path (`enerdatasynth.resources.wind` instead of a separate
top-level `wind_diffusion` package), not its design.

If you're using the product, you shouldn't need to care about any of this — it's an
implementation detail behind `enerdatasynth.model(...)`, which looks and works
identically for both resources. See
[`src/enerdatasynth/README.md`](enerdatasynth/README.md) for the full package map.
