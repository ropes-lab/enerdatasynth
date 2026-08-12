"""Configuration for wind_diffusion models.

Encodes the CONFIG dict structure shared by the three original monolithic
training scripts (`transformer_1ch_conv1d_ddpm_wind.py`,
`transformer_2ch_conv1d_ddpm_wind.py`, `transformer_conv1d_ddpm_wind.py`),
plus the one axis they differ on: the model's target channels
("presets" 1ch / 2ch / 3ch).

Frozen contract (see repo_and_api_plan.md, C1/C2/C6):
    seq_len=288, freq="5min", diffusion_timesteps=200,
    beta_schedule="cosine", beta_start=1e-4, beta_end=0.02.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

PresetName = Literal["1ch", "2ch", "3ch"]

# Raw columns read from the source CSV. Identical across all three presets.
# temperature_100m is used only for conditioning (temp_min_max_spread), never
# as a target channel in any preset.
FEATURE_COLS: list[str] = ["windspeed_100m", "winddirection_100m", "temperature_100m"]

# Per-preset target-channel ("MODEL_CHANNELS") names. The actual derivation
# logic (how these columns get computed from the raw feature_cols) lives in
# transforms.py: derive_channels().
PRESET_MODEL_CHANNELS: dict[PresetName, list[str]] = {
    "1ch": ["u"],
    "2ch": ["U", "V"],
    "3ch": ["u", "v_sin", "v_cos"],
}


@dataclass
class WindDiffusionConfig:
    """Mirrors the CONFIG dict used by the original training scripts."""

    # ---- Preset ----
    preset: PresetName = "2ch"

    # ---- Data ----
    data_path: str = "data/temporal/grid_22532/grid_22532_2010_2020.csv"
    timestamp_col: str = "timestamp"
    feature_cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS))
    output_dir: str = "outputs/ddpm_transformer_run"

    # ---- Reproducibility ----
    seed: int = 42

    # ---- Train / validation split ----
    split_date: str | None = None
    split_frac: float = 0.85

    # ---- Sequence / resampling ----
    seq_len: int = 288
    expected_freq: str = "5min"
    resample: bool = False
    drop_incomplete_days: bool = True

    # ---- Training ----
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    num_epochs: int = 100
    early_stopping_patience: int = 15

    # ---- Model architecture (Transformer + Conv1D denoiser) ----
    # hidden_dim=128 (was 64): measured better across the board in the same
    # ablations that settled norm_first (see reports/BASELINE_1CH_H128*.md) --
    # best val_loss and no fidelity-gate regression at the larger size.
    hidden_dim: int = 128
    num_conv_blocks: int = 3
    num_transformer_layers: int = 3
    num_heads: int = 4
    kernel_size: int = 5
    dropout: float = 0.05

    # ---- Conditioning / time embeddings ----
    time_emb_dim: int = 64
    cond_emb_dim: int = 64

    # ---- Diffusion ----
    diffusion_timesteps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "cosine"  # "linear" or "cosine"
    sample_clip_margin: float = 1.0

    # ---- Conditioning features ----
    condition_on: dict[str, bool] = field(
        default_factory=lambda: {
            "month": True,
            "day_of_week": True,
            "temp_min_max_spread": True,
            "wind_regime_bucket": True,
        }
    )
    # How wind_regime_bucket is encoded in the condition vector. "onehot"
    # (was the only option; kept as the default for backward compatibility)
    # gives the network three independent, unordered directions -- there is
    # no representation of "mid" as being between "low" and "high"; measured
    # across every architecture/guidance/sampling change tried this project,
    # "mid" regime-steering accuracy never once crossed its 50%
    # no-conditioning-effect chance floor, the one problem nothing else
    # fixed. "ordinal" (transforms.ORDINAL_WIND_BUCKET_VALUES: low=1, mid=2,
    # high=3 -- deliberately not centered on 0, see that constant's
    # docstring) encodes regime as a single continuous value where mid is
    # the literal midpoint of low and high, testing the hypothesis that this
    # geometry is what one-hot was missing.
    wind_regime_encoding: str = "onehot"

    # ---- Device ----
    # "auto" resolves to CUDA if available, else CPU (see utils.resolve_device).
    # Also accepts an explicit "cpu"/"cuda" to force one or the other.
    device: str = "auto"

    # ---- Architecture optimization ----
    # Pre-LN transformer blocks (norm_first=True) are generally believed to
    # train more stably than Post-LN. Measured against this denoiser it does
    # NOT hold: a controlled ablation (same warmup+cosine schedule,
    # hidden_dim=128, 60 epochs) showed norm_first=True roughly doubles
    # spectral_high_band and ramp_wasserstein fidelity-gate error relative to
    # norm_first=False, for no offsetting gain elsewhere (val_loss was also
    # worse: 0.056 vs 0.052). See reports/BASELINE_1CH_POSTLN_E60.md vs
    # reports/BASELINE_1CH_H128_E60.md. Defaulting to Post-LN.
    norm_first: bool = False

    # Probability of replacing (a slice of) a training sample's condition
    # vector z with zero (classifier-free guidance training). 0.3 (was 0.0):
    # measured as a real win when paired with cond_dropout_scope="wind_regime"
    # and guidance_scale~40 at generation time -- regime_accuracy roughly
    # doubled for "low" (0.38->0.78) and improved for "high" (0.47->0.67),
    # with every other fidelity gate improving alongside it, not trading off
    # against it. See reports/BASELINE_1CH_REGIMECFG_GS*.md. Set to 0.0 to
    # fully disable (byte-identical to every checkpoint trained before this
    # existed) -- 0.0 makes `cond_dropout_scope` and `guidance_scale`/
    # `guidance_slice` at sampling time no-ops.
    cond_dropout_prob: float = 0.3
    # Which part of the condition vector cond_dropout_prob/guidance_scale
    # zero out. "wind_regime" (was "full"): zeroes only the 3-dim
    # wind_regime one-hot slice (see transforms.wind_regime_slice), leaving
    # month/day_of_week/temp untouched in both branches. Switched from
    # "full" after a measured full-vector CFG run made things worse, not
    # better -- it amplified the whole 25-dim vector (dominated by month=12
    # + day_of_week=7 dims) rather than the 3-dim regime signal we actually
    # want strengthened, and degraded every fidelity gate as guidance
    # increased. Regime-scoped guidance does not have that failure mode; see
    # reports/BASELINE_1CH_CFG_GS*.md ("full", negative) vs
    # reports/BASELINE_1CH_REGIMECFG_GS*.md ("wind_regime", positive).
    cond_dropout_scope: str = "wind_regime"
    # Epochs of linear LR warmup before cosine annealing decay kicks in.
    # Set to 0 to disable warmup and use plain cosine annealing over all
    # num_epochs.
    warmup_epochs: int = 5

    # ---- Wind persistence (opt-in) ----
    # Conditions on yesterday's actual daily-mean windspeed (see
    # data.py's prepare_datasets), added specifically to give "mid"-regime
    # generation a real signal to steer on: a class-balanced classifier
    # predicting wind_regime from month/day_of_week/temperature ALONE (no
    # wind data) beat chance for low (0.69 vs 0.25) and high (0.45 vs 0.25)
    # but did WORSE than chance for mid (0.41 vs 0.50) -- see
    # reports/MID_REGIME_ANALYSIS.md. Persistence is untried there and is
    # the most promising candidate: unlike season/temperature, it isn't
    # structurally biased toward the extremes. Opt-in via
    # `condition_on["wind_persistence"]=True` (absent/False by default --
    # fully backward compatible with every checkpoint trained before this
    # existed). Independent of `cond_dropout_prob`/`cond_dropout_scope`
    # (which govern regime-CFG) since this feature serves a different
    # purpose and needs its own masking so the model learns to work
    # sensibly when no previous-day value is supplied (the API's default
    # UX) as well as when one is.
    wind_persistence_dropout_prob: float = 0.3

    # ---- Joint channel-pair consistency loss ----
    # For presets whose channels encode a 2D vector magnitude (2ch's U/V --
    # NOT 3ch's v_sin/v_cos, which are already bounded by construction), the
    # primary noise-prediction loss can match each channel's marginal
    # distribution well while still getting the pair's *joint* radius
    # (sqrt(c0^2+c1^2)) wrong -- measured directly: 2ch's synthetic
    # sqrt(U^2+V^2) came out ~2.8x inflated vs real, a training-time defect
    # that generation-time clamping (independent or joint-radius) could not
    # fix (see reports/2CH_CLAMP_INVESTIGATION.md). This adds an auxiliary
    # training loss comparing the predicted-x0 radius for
    # `joint_consistency_channels` against the real x0's radius, directly
    # penalizing exactly that failure mode. None/0.0 (default) disables it
    # -- fully backward compatible with every checkpoint trained before this
    # existed, including the current 1ch/3ch recipe (which don't need it).
    joint_consistency_channels: tuple[int, int] | None = None
    joint_consistency_weight: float = 0.0

    # ---- Generation ----
    num_synthetic_samples: int = 50
    generation_mode: str = "random"  # "random" or "all"

    # ---- Saving / plotting ----
    save_checkpoints: bool = True
    make_plots: bool = True

    # ---- Sample clamping (standardized-space clip bounds during DDPM sampling) ----
    sample_clip_std: float = 3.5

    @property
    def model_channels(self) -> list[str]:
        return PRESET_MODEL_CHANNELS[self.preset]

    @property
    def n_model_channels(self) -> int:
        return len(self.model_channels)

    def to_dict(self) -> dict[str, Any]:
        """A CONFIG-dict-shaped view, for parity with the original scripts
        (e.g. for json.dump into config.json / checkpoint["config"])."""
        d = copy.deepcopy(self.__dict__)
        d["model_channels"] = self.model_channels
        return d


def get_config(preset: PresetName = "2ch", **overrides: Any) -> WindDiffusionConfig:
    """Factory for a WindDiffusionConfig with a given preset and any field overrides.

    Example:
        cfg = get_config("2ch", num_epochs=30, data_path="my.csv")
    """
    if preset not in PRESET_MODEL_CHANNELS:
        raise ValueError(
            f"Unknown preset {preset!r}; expected one of {sorted(PRESET_MODEL_CHANNELS)}"
        )
    cfg = WindDiffusionConfig(preset=preset)
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise TypeError(f"WindDiffusionConfig has no field {key!r}")
        setattr(cfg, key, value)
    return cfg
