"""
powerdiffusion
================
A conditional diffusion model pipeline for generating synthetic daily
PJM electricity load profiles.

This top-level import surface re-exports the most commonly used
pieces so callers don't need to know the internal module layout for
everyday use, e.g.:

    from powerdiffusion import RunConfig, train_one_config

For anything not re-exported here, import directly from the relevant
submodule (`powerdiffusion.data`, `powerdiffusion.models`,
`powerdiffusion.diffusion`, `powerdiffusion.training`,
`powerdiffusion.sampling`, `powerdiffusion.evaluation`,
`powerdiffusion.visualization`, `powerdiffusion.utils`).
"""

from powerdiffusion.utils.config import RunConfig, set_seed
from powerdiffusion.models.denoiser import SimpleMLPDenoiser
from powerdiffusion.diffusion.ddpm import DDPM
from powerdiffusion.data.preprocess import run_preprocessing
from powerdiffusion.training.train import train_one_config
from powerdiffusion.sampling.generate import generate_for_run
from powerdiffusion.evaluation.metrics import compute_metrics

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "RunConfig",
    "set_seed",
    "SimpleMLPDenoiser",
    "DDPM",
    "run_preprocessing",
    "train_one_config",
    "generate_for_run",
    "compute_metrics",
]
