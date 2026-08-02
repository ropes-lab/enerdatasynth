"""
src/enerdatasynth/generators/base.py
======================================
Common interface for every domain-specific synthetic-generation model
in this package (today: electric load, via `LoadDiffusionModel`).

The package's ultimate goal is a unified SDK across multiple PJM
resource types (electric load, and eventually wind or other
generation types) that downstream tooling — CLIs, dashboards,
benchmarking harnesses — can drive without a domain-specific branch:

    samples = generator.sample(z, seed=0)

A wind-generation counterpart is NOT implemented here: there is no
wind data, trained model, or validated research behind one anywhere
in this project (see MIGRATION.md and research_history/ — everything
is PJM electric load). Fabricating a `WindDiffusionModel` with no
underlying research would misrepresent what's actually been validated.
This module exists so that when real wind work is ready, it has a
concrete contract to implement rather than an ad hoc, incompatible API
— see `docs/extending.md` for what implementing a new domain involves.
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
