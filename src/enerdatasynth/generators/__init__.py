"""
src/enerdatasynth/generators
==============================
The SDK layer: `BaseGenerator` is the common interface every
resource-type generator implements. `LoadDiffusionModel` (electric load)
and `WindDiffusionModel` (wind, wrapping the vendored `wind_diffusion`
package, checkpoints bundled under `data/checkpoints/wind/`) are the
two concrete implementations.
"""

from enerdatasynth.generators.base import BaseGenerator
from enerdatasynth.generators.load import LoadDiffusionModel
from enerdatasynth.generators.wind import WindDiffusionModel

# Importing these runs each resource's `register_resource(...)` call
# (see `registry.py`) as a side effect, adding it to the shared registry
# `sdk.py` reads from. This line is the only per-resource edit needed
# anywhere outside a resource's own two files (`generators/<name>.py` +
# `generators/<name>_conditioning.py`) -- sdk.py itself needs no edit to
# add a resource.
from enerdatasynth.generators import load_conditioning as _load_conditioning  # noqa: F401
from enerdatasynth.generators import wind_conditioning as _wind_conditioning  # noqa: F401

__all__ = ["BaseGenerator", "LoadDiffusionModel", "WindDiffusionModel"]
