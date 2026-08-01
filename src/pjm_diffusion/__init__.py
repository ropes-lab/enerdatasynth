"""
pjm_diffusion
================
A conditional diffusion model SDK for generating synthetic PJM
resource profiles. Today that means electric load; the public
interface (`generators.BaseGenerator`) is designed so a future
resource type (e.g. wind) can plug in through the same contract — see
`pjm_diffusion.generators.base` for what that requires.

For most users, `LoadDiffusionModel` is the entry point:

    from pjm_diffusion import LoadDiffusionModel
    model = LoadDiffusionModel.from_run_id("transformer_T1000_H64_L2")
    samples = model.sample(z, seed=0)

This top-level import surface also re-exports the lower-level pieces
used to build and train models, so callers don't need to know the
internal module layout for everyday use, e.g.:

    from pjm_diffusion import RunConfig, train_one_config

For anything not re-exported here, import directly from the relevant
submodule (`pjm_diffusion.data`, `pjm_diffusion.models`,
`pjm_diffusion.diffusion`, `pjm_diffusion.training`,
`pjm_diffusion.sampling`, `pjm_diffusion.evaluation`,
`pjm_diffusion.visualization`, `pjm_diffusion.generators`,
`pjm_diffusion.utils`).

Production code in this package never imports from `research_history/`
— that directory exists only for reproducibility/provenance and is
not installed as part of the package.
"""

from pjm_diffusion.utils.config import RunConfig, set_seed, DEFAULT_ARCHITECTURE
from pjm_diffusion.models import build_denoiser, TransformerDenoiser, SimpleMLPDenoiser
from pjm_diffusion.diffusion.ddpm import DDPM
from pjm_diffusion.data.preprocess import run_preprocessing
from pjm_diffusion.training.train import train_one_config
from pjm_diffusion.sampling.generate import generate_for_run
from pjm_diffusion.evaluation.metrics import compute_metrics
from pjm_diffusion.evaluation.calibration import calibrate_checkpoint
from pjm_diffusion.generators import BaseGenerator, LoadDiffusionModel

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "BaseGenerator",
    "LoadDiffusionModel",
    "RunConfig",
    "set_seed",
    "DEFAULT_ARCHITECTURE",
    "build_denoiser",
    "TransformerDenoiser",
    "SimpleMLPDenoiser",
    "DDPM",
    "run_preprocessing",
    "train_one_config",
    "generate_for_run",
    "calibrate_checkpoint",
    "compute_metrics",
]
