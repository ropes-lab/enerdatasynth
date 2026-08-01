# Extending: Adding a New Generator (e.g. Wind)

This package's public interface, `pjm_diffusion.generators.BaseGenerator`,
is deliberately thin so a different PJM resource type (wind, solar, ...)
could be added as a first-class citizen of the SDK — callable through
the same `.sample(z, seed=...)` contract as `LoadDiffusionModel` — without
requiring downstream tooling (CLIs, dashboards, benchmarking harnesses)
to special-case it.

**Nothing wind-specific exists in this package today.** There is no
wind data, wind-trained checkpoint, or wind-validated architecture
anywhere in this repository or in `research_history/`. This page
describes what implementing one would involve, not something already
built — don't infer from its existence that wind support is ready to
use.

## What `BaseGenerator` requires

```python
from pjm_diffusion.generators.base import BaseGenerator

class WindDiffusionModel(BaseGenerator):
    domain = "wind"

    def sample(self, z, seed=None, **kwargs) -> np.ndarray:
        ...
```

That's the entire hard requirement: a `domain` string and a `sample`
method returning one synthetic profile per row of `z`. `count_parameters()`
is optional (override it if there's an underlying `nn.Module`).

## What it does NOT prescribe

Deliberately, `BaseGenerator` says nothing about:

- **Conditioning schema.** Wind generation would plausibly need wind
  speed/direction at multiple heights, turbine power curves, or other
  variables entirely unlike the load conditioning vector
  (month/dow/holiday/Tmax/Tmin/spread) — a different `cond_dim`,
  different scalers, different preprocessing.
- **Temporal resolution.** Load profiles here are fixed at 24 hourly
  values per day; wind may need sub-hourly resolution or a different
  profile length. Nothing in `BaseGenerator` assumes `x_dim=24`.
- **Constructor signature.** `LoadDiffusionModel.__init__` and
  `.from_run_id()` are load-specific conveniences (checkpoint layout,
  `result.json` schema); a wind implementation isn't obligated to match
  them, only the `.sample()` contract.
- **Underlying architecture.** There's no requirement to reuse
  `TransformerDenoiser` or the DDPM process in `diffusion/ddpm.py` —
  those were validated for load; a genuinely different resource type
  may warrant different modeling choices, decided by whoever actually
  does that research.

## Suggested real first step

Before writing any model code: get real wind data, do the EDA, and
figure out what a sensible conditioning vector and evaluation metric
suite look like for wind — mirroring how `research_history/` shows the
load pipeline's conditioning and metrics were arrived at empirically,
not assumed upfront. `pjm_diffusion.evaluation.metrics.compute_metrics`
may or may not be the right metric suite for wind; check before reusing
it uncritically.
