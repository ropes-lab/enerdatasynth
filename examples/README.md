# `examples/`

Runnable examples, one folder per resource, laid out like `data/` and `src/enerdatasynth/resources/`.

| Folder | Examples | README |
|---|---|---|
| `pjm_load/` | Eleven scripts: inspect the data, generate, plot real vs. generated, compare checkpoints, benchmark, run the pipeline, and plot, train, score and compare the 17 zone models | [`pjm_load/README.md`](pjm_load/README.md) |
| `wind/` | Two scripts: generate wind days, plot generated days per regime. No real-vs-generated example yet, because this repository has no real wind data | [`wind/README.md`](wind/README.md) |

Every script ends its docstring with a **Reproduce** block: the exact commands, what you have to provide, and where the output goes. A third resource gets its own `examples/<resource>/` folder in the same way.
