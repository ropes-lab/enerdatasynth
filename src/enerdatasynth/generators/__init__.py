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

__all__ = ["BaseGenerator", "LoadDiffusionModel", "WindDiffusionModel"]
