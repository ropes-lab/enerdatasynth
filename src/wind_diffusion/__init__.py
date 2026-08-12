"""wind_diffusion: conditional DDPM wind-speed profile generation.

Extracted package core (M1) — mirrors the three original monolithic
training scripts (`transformer_1ch/2ch/conv1d_ddpm_wind.py`) as an
importable, testable package. See repo_and_api_plan.md for the roadmap.
"""

from wind_diffusion.api import (
    WindProfile,
    describe_model,
    generate_wind_profile,
    list_models,
)
from wind_diffusion.api import preload as preload  # noqa: F401 (see note below)
from wind_diffusion.config import (
    PRESET_MODEL_CHANNELS,
    FEATURE_COLS,
    WindDiffusionConfig,
    get_config,
)
from wind_diffusion.data import (
    DailyWindowDataset,
    NormStats,
    PreparedData,
    prepare_datasets,
)
from wind_diffusion.diffusion import (
    DiffusionSchedule,
    ddpm_sample,
    make_beta_schedule,
    q_sample,
)
from wind_diffusion.evaluate import (
    autocorr,
    generate_interactive,
    generate_multiple_for_condition,
    get_cached_model,
    reconstruct_to_dataframe,
)
from wind_diffusion.exceptions import (
    CheckpointNotFoundError,
    InvalidConditionError,
    WeightsDownloadError,
    WindDiffusionError,
)
from wind_diffusion.models import (
    CondMLP,
    ResidualConvBlock,
    SinusoidalTimeEmbedding,
    TransformerDenoiser,
)
from wind_diffusion.samplers import sample
from wind_diffusion.train import build_model, run_epoch, train
from wind_diffusion.transforms import (
    WindTransformPipeline,
    assign_wind_bucket,
    build_condition_matrix,
    derive_channels,
    destandardize_windows,
    speed_dir_to_uv,
    standardize_windows,
    uv_to_speed_dir,
)
from wind_diffusion.utils import resolve_device, set_seed

# NOTE (M4): `preload` at package top level is now `wind_diffusion.api.preload`
# (resolves weights for a preset via the registry, then warms the model
# cache) rather than `wind_diffusion.evaluate.preload` (which takes a raw
# checkpoint path directly). The latter is still available as
# `wind_diffusion.evaluate.preload` / `wind_diffusion.get_cached_model` for
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
