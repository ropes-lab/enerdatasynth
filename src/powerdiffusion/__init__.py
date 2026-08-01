"""
powerdiffusion
================
A conditional diffusion model pipeline for generating synthetic daily
PJM electricity load profiles.

For most users, `LoadDiffusionModel` is the entry point:

    from powerdiffusion import LoadDiffusionModel
    model = LoadDiffusionModel.from_run_id("transformer_T1000_H64_L2")
    samples = model.sample(z, seed=0)

This top-level import surface also re-exports the lower-level pieces
used to build and train models, so callers don't need to know the
internal module layout for everyday use, e.g.:

    from powerdiffusion import RunConfig, train_one_config

For anything not re-exported here, import directly from the relevant
submodule (`powerdiffusion.data`, `powerdiffusion.models`,
`powerdiffusion.diffusion`, `powerdiffusion.training`,
`powerdiffusion.sampling`, `powerdiffusion.evaluation`,
`powerdiffusion.visualization`, `powerdiffusion.utils`).

Production code in this package never imports from `research_history/`
— that directory exists only for reproducibility/provenance and is
not installed as part of the package.
"""

from powerdiffusion.utils.config import RunConfig, set_seed, DEFAULT_ARCHITECTURE
from powerdiffusion.models import build_denoiser, TransformerDenoiser, SimpleMLPDenoiser
from powerdiffusion.diffusion.ddpm import DDPM
from powerdiffusion.data.preprocess import run_preprocessing
from powerdiffusion.training.train import train_one_config
from powerdiffusion.sampling.generate import generate_for_run
from powerdiffusion.evaluation.metrics import compute_metrics
from powerdiffusion.evaluation.calibration import calibrate_checkpoint
from powerdiffusion.api import LoadDiffusionModel

__version__ = "0.2.0"

__all__ = [
    "__version__",
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
