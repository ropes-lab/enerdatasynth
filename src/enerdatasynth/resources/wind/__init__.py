"""wind_diffusion: conditional DDPM wind-speed profile generation.

Extracted package core (M1) — mirrors the three original monolithic
training scripts (`transformer_1ch/2ch/conv1d_ddpm_wind.py`) as an
importable, testable package. See repo_and_api_plan.md for the roadmap.
"""

from enerdatasynth.resources.wind.api import (
    WindProfile,
    describe_model,
    generate_wind_profile,
    list_models,
)
from enerdatasynth.resources.wind.api import preload as preload  # noqa: F401 (see note below)
from enerdatasynth.resources.wind.config import (
    PRESET_MODEL_CHANNELS,
    FEATURE_COLS,
    WindDiffusionConfig,
    get_config,
)
from enerdatasynth.resources.wind.data.data import (
    DailyWindowDataset,
    NormStats,
    PreparedData,
    prepare_datasets,
)
from enerdatasynth.resources.wind.diffusion.diffusion import (
    DiffusionSchedule,
    ddpm_sample,
    make_beta_schedule,
    q_sample,
)
from enerdatasynth.resources.wind.evaluation.evaluate import (
    autocorr,
    generate_interactive,
    generate_multiple_for_condition,
    get_cached_model,
    reconstruct_to_dataframe,
)
from enerdatasynth.resources.wind.exceptions import (
    CheckpointNotFoundError,
    InvalidConditionError,
    WeightsDownloadError,
    WindDiffusionError,
)
from enerdatasynth.resources.wind.models.models import (
    CondMLP,
    ResidualConvBlock,
    SinusoidalTimeEmbedding,
    TransformerDenoiser,
)
from enerdatasynth.resources.wind.diffusion.samplers import sample
from enerdatasynth.resources.wind.training.train import build_model, run_epoch, train
from enerdatasynth.resources.wind.data.transforms import (
    WindTransformPipeline,
    assign_wind_bucket,
    build_condition_matrix,
    derive_channels,
    destandardize_windows,
    speed_dir_to_uv,
    standardize_windows,
    uv_to_speed_dir,
)
from enerdatasynth.resources.wind.utils import resolve_device, set_seed

# NOTE (M4): `preload` at package top level is now `enerdatasynth.resources.wind.api.preload`
# (resolves weights for a preset via the registry, then warms the model
# cache) rather than `enerdatasynth.resources.wind.evaluation.evaluate.preload` (which takes a raw
# checkpoint path directly). The latter is still available as
# `enerdatasynth.resources.wind.evaluation.evaluate.preload` / `wind_diffusion.get_cached_model` for
# callers that already have a resolved checkpoint path.

__all__ = [
    "PRESET_MODEL_CHANNELS",
    "FEATURE_COLS",
    "WindDiffusionConfig",
    "get_config",
    "DailyWindowDataset",
    "NormStats",
    "PreparedData",
    "prepare_datasets",
    "DiffusionSchedule",
    "ddpm_sample",
    "make_beta_schedule",
    "q_sample",
    "autocorr",
    "generate_interactive",
    "generate_multiple_for_condition",
    "get_cached_model",
    "preload",
    "reconstruct_to_dataframe",
    "CondMLP",
    "ResidualConvBlock",
    "SinusoidalTimeEmbedding",
    "TransformerDenoiser",
    "sample",
    "build_model",
    "run_epoch",
    "train",
    "WindTransformPipeline",
    "assign_wind_bucket",
    "build_condition_matrix",
    "derive_channels",
    "destandardize_windows",
    "speed_dir_to_uv",
    "standardize_windows",
    "uv_to_speed_dir",
    "resolve_device",
    "set_seed",
    "WindProfile",
    "generate_wind_profile",
    "list_models",
    "describe_model",
    "WindDiffusionError",
    "CheckpointNotFoundError",
    "WeightsDownloadError",
    "InvalidConditionError",
]

__version__ = "0.1.0"
