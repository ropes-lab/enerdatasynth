"""
src/enerdatasynth/generators/base.py
======================================
Common interface for every domain-specific synthetic-generation model
in this package: electric load (`LoadDiffusionModel`) and wind
(`WindDiffusionModel`, wrapping the vendored `wind_diffusion` package).

The package's ultimate goal is a unified SDK across multiple PJM
resource types that downstream tooling — CLIs, dashboards, benchmarking
harnesses — can drive without a domain-specific branch:

    samples = generator.sample(z, seed=0)

`WindDiffusionModel` wraps real, previously-published wind research
(`wind_diffusion`, vendored from a separate Data-For-Good project).
Real trained checkpoints (1ch/2ch/3ch) are bundled under
`data/checkpoints/wind/<preset>/` and resolved automatically -- see
`generators/wind_conditioning.py`'s `wind_construct`. (The vendored
`wind_diffusion/manifest.json`'s own presets still read `"pending"` --
that describes only its own external-download mechanism, which this
package's bundled-checkpoint convention takes precedence over; it does
not mean wind is unavailable here.)
"""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np


class BaseGenerator(ABC):
    """
    Required contract for a domain-specific generator:

    - `domain` (class attribute): a short string identifying the
      resource type ("load", "wind", ...).
    - `sample(z, seed=None, **kwargs) -> np.ndarray`: generate one
      synthetic profile per row of the conditioning array `z`.

    This is intentionally a thin contract. Domain-specific concerns —
    what conditioning variables exist, what post-processing applies,
    how a checkpoint is structured — belong in the concrete subclass,
    not here. A shared base class that encoded load-specific
    assumptions (24-hour daily profiles, this exact conditioning
    vector) would make it useless for a genuinely different resource
    type; a real wind implementation may need an entirely different
    conditioning schema (wind speed/direction at multiple heights,
    turbine curves, sub-hourly resolution) and constructor.
    """

    domain: str = "unknown"

    @abstractmethod
    def sample(self, z: np.ndarray, seed: int | None = None, **kwargs) -> np.ndarray:
        """Generate one synthetic profile per row of `z`."""
        raise NotImplementedError

    def count_parameters(self) -> int:
        """Optional: override if the underlying model exposes a parameter count."""
        raise NotImplementedError(f"{type(self).__name__} does not implement count_parameters()")
