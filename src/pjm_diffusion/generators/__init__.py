"""
src/pjm_diffusion/generators
==============================
The SDK layer: `BaseGenerator` is the common interface every
resource-type generator implements; `LoadDiffusionModel` is the one
concrete implementation that exists today (electric load). See
`base.py` for why a wind counterpart isn't stubbed out here.
"""

from pjm_diffusion.generators.base import BaseGenerator
from pjm_diffusion.generators.load import LoadDiffusionModel

__all__ = ["BaseGenerator", "LoadDiffusionModel"]
