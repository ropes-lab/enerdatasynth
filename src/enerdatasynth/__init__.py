"""
enerdatasynth
================
A research SDK for synthetic PJM energy resource profiles. Today that
means electric load; the public interface is designed so a future
resource type (e.g. wind) can be added through the same contract —
see `docs/extending.md`.

Primary entry point — you should not need anything else for everyday
generation:

    import enerdatasynth

    load = enerdatasynth.model("pjm_load", zone="PS")
    load.set(month=7, Tmax=86, Tmin=70, weekday=True, holiday=False)
    samples = load.sample(n=100)   # (100, 24) array, MW

`enerdatasynth.model(...)` never exposes tensors, architecture
classes, checkpoint paths, or the fact that a DDPM/transformer is
involved at all. See `docs/api_reference.md` for the full contract.

For research/development use — training new checkpoints, computing
evaluation metrics, running calibration — this top-level namespace
also re-exports the lower-level building blocks:

    from enerdatasynth import RunConfig, train_one_config, LoadDiffusionModel

For anything not re-exported here, import directly from the relevant
submodule (`enerdatasynth.data`, `enerdatasynth.models`,
`enerdatasynth.diffusion`, `enerdatasynth.training`,
`enerdatasynth.sampling`, `enerdatasynth.evaluation`,
`enerdatasynth.visualization`, `enerdatasynth.generators`,
`enerdatasynth.utils`).

Production code in this package never imports from `research_history/`
— that directory exists only for reproducibility/provenance and is
not installed as part of the package.
"""

from enerdatasynth.sdk import model, ResourceModel
from enerdatasynth.utils.config import RunConfig, set_seed, DEFAULT_ARCHITECTURE
from enerdatasynth.models import build_denoiser, TransformerDenoiser, SimpleMLPDenoiser
from enerdatasynth.diffusion.ddpm import DDPM
from enerdatasynth.data.preprocess import run_preprocessing
from enerdatasynth.training.train import train_one_config
from enerdatasynth.sampling.generate import generate_for_run
from enerdatasynth.evaluation.metrics import compute_metrics
from enerdatasynth.evaluation.calibration import calibrate_checkpoint
from enerdatasynth.generators import BaseGenerator, LoadDiffusionModel

__version__ = "0.4.0"

__all__ = [
    "__version__",
    "model",
    "ResourceModel",
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
