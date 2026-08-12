"""
src/enerdatasynth/utils/config.py
====================================
Central configuration for the entire pipeline. Single source of truth
for paths, conditioning definitions, architecture defaults, and
reproducibility settings.

Pipeline summary:
  - conditioning: month (one-hot), dow (one-hot), holiday (binary),
                  Tmax (z-scored), Tmin (z-scored), spread (z-scored)
  - architecture: transformer denoiser (default) — the Week-6
                  architecture-comparison winner, promoted to primary
                  baseline in Week 7 research; a simple MLP remains
                  available (`architecture="mlp"`) as a legacy baseline
  - schedule:     linear beta schedule — validated against cosine and
                  quadratic alternatives in research_history and found
                  best (see research_history/reports/week6/noise/)
  - loss:         standard diffusion MSE only

Anything not listed above (MW_anchor, Fourier modes, custom losses,
a unified multi-zone model) is out of scope for this baseline and is
tracked only as future work in reports/future_experiments.md.
"""

from __future__ import annotations
import os
import random
import dataclasses
import numpy as np
import torch
import yaml
from dataclasses import dataclass, field
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
# Deliberately NOT derived from `__file__`: that only works for an
# editable install, where the source tree stays put next to data/ and
# checkpoints/. A real `pip install .` / wheel install copies the code
# into site-packages, far from any project directory — `__file__`-based
# resolution silently produces nonsense paths there (this broke a real
# `pip install dist/*.whl` verification; see MIGRATION.md). Instead:
# an explicit override via $ENERDATASYNTH_PROJECT_ROOT, or the current
# working directory — the project this SDK operates on is wherever you
# run `enerdatasynth ...` / import it from, not wherever pip put the code.
PROJECT_ROOT     = Path(os.environ.get("ENERDATASYNTH_PROJECT_ROOT", Path.cwd())).resolve()
# Everything below lives under data/ — the repo root is kept to 5
# top-level folders (src/, tests/, docs/, examples/, data/) so a first
# look at the project isn't cluttered with a dozen artifact folders.
DATA_ROOT        = PROJECT_ROOT / "data"
CONFIGS_DIR      = DATA_ROOT / "configs"
RAW_DIR          = DATA_ROOT / "raw"
PROCESSED_DIR    = DATA_ROOT / "processed"
PROCESSED_BACKUP = DATA_ROOT / "processed_backup"
CHECKPOINT_DIR   = DATA_ROOT / "checkpoints"
OUTPUTS_DIR      = DATA_ROOT / "outputs"
GENERATED_DIR    = OUTPUTS_DIR / "generated"
LOGS_DIR         = OUTPUTS_DIR / "logs"
REPORTS_DIR      = DATA_ROOT / "reports"
FIGURES_DIR      = REPORTS_DIR / "figures"
RESEARCH_HISTORY_DIR = DATA_ROOT / "research_history"

_ALL_DATA_DIRS = [RAW_DIR, PROCESSED_DIR, PROCESSED_BACKUP, CHECKPOINT_DIR,
                   GENERATED_DIR, LOGS_DIR, REPORTS_DIR, FIGURES_DIR]


def ensure_dirs() -> None:
    """Create the data/ directory scaffold. Previously ran unconditionally
    at *module import time*, which meant a plain `import enerdatasynth`
    from inside any unrelated project silently created data/raw/,
    data/processed/, etc. in that project's cwd (PROJECT_ROOT defaults to
    Path.cwd() when $ENERDATASYNTH_PROJECT_ROOT isn't set) -- a library
    shouldn't mutate the filesystem just from being imported. Now called
    explicitly, at call time, by every function that actually writes into
    this tree (run_preprocessing, train_one_config, run_sweep,
    generate_for_run, calibrate_checkpoint, run_all_validation_plots,
    write_failure_analysis_md, generate_results_summary -- the same set
    re-exported as this package's public API), so the directories still
    appear exactly when needed, not merely from importing the module.
    Idempotent (`exist_ok=True`), safe to call redundantly.
    """
    for d in _ALL_DATA_DIRS:
        d.mkdir(parents=True, exist_ok=True)


ZONE = "PS"   # PSE&G / New Jersey

# ─────────────────────────────────────────────────────────────────────────────
# Conditioning vector definition
# ─────────────────────────────────────────────────────────────────────────────
# z = [month_onehot(12), dow_onehot(7), is_holiday(1), Tmax_z(1), Tmin_z(1), spread_z(1)]
#
# Caveat: the transformer architecture below was validated in
# research_history against a DIFFERENT 6-dimensional raw conditioning
# vector ("Z5": month and day-of-week as integers, not one-hot) — not
# this one. The one-hot encoding here is this package's deliberate,
# documented choice, but it has not itself been re-validated against
# the transformer/calibration results quoted in this file. See the
# migration report for details; this is flagged, not silently resolved.
N_MONTHS    = 12
N_DOW       = 7
COND_DIM    = N_MONTHS + N_DOW + 1 + 1 + 1 + 1   # = 23
X_DIM       = 24   # hours per day

CONDITION_SPEC = {
    "month":   {"encoding": "one-hot", "dim": N_MONTHS},
    "dow":     {"encoding": "one-hot", "dim": N_DOW},
    "holiday": {"encoding": "binary",  "dim": 1},
    "Tmax":    {"encoding": "z-score", "dim": 1},
    "Tmin":    {"encoding": "z-score", "dim": 1},
    "spread":  {"encoding": "z-score", "dim": 1, "definition": "max(load_day) - min(load_day)"},
}

EXCLUDED_BY_DESIGN = [
    "MW_anchor", "Fourier modes", "custom losses", "higher moments",
    "user-facing abstractions", "unified multi-zone model",
]

# ─────────────────────────────────────────────────────────────────────────────
# Architecture defaults
# ─────────────────────────────────────────────────────────────────────────────
# "transformer": the Week-6 architecture-comparison winner, promoted
#   to primary baseline in Week 7 (research_history/reports/week6/arch/,
#   research_history/reports/week7/summary_report.md). model_dim=64,
#   num_layers=2, num_heads=4, T=1000 are the exact hyperparameters
#   that produced those validated results.
# "mlp": the original simple baseline, kept as a legacy/research option.
DEFAULT_ARCHITECTURE = "transformer"
DEFAULT_N_HEADS       = 4   # transformer-only

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter sweep grid
# ─────────────────────────────────────────────────────────────────────────────
SWEEP_DIFFUSION_STEPS = [50, 100, 200]
SWEEP_HIDDEN_DIM      = [64, 128, 256]
SWEEP_N_LAYERS        = [2, 4, 6]

DEFAULT_T          = 1000   # matches the validated transformer/calibration research
DEFAULT_HIDDEN_DIM = 64     # transformer model_dim (also a valid MLP hidden_dim)
DEFAULT_N_LAYERS   = 2      # transformer encoder layers (also a valid MLP block count)

# ─────────────────────────────────────────────────────────────────────────────
# Training defaults
# ─────────────────────────────────────────────────────────────────────────────
SEED            = 42
BATCH_SIZE      = 64
LEARNING_RATE   = 3e-4
MAX_EPOCHS      = 400        # matches the transformer convergence run (week7.py --epochs 400)
EARLY_STOP_PATIENCE = 40     # matches week7.py --patience 40; set to None to disable
# NOTE: kept for informational/backward-compat use (e.g. displaying "what
# would resolve_device() pick right now"), but NOT used as a function
# parameter default anywhere -- `device: str = DEVICE` bakes this value in
# at *function-definition* (import) time, since Python evaluates defaults
# once. If CUDA becomes available later in the same process (common in
# notebooks, or workers that attach a GPU after start), every call would
# silently keep using the stale default. Use `resolve_device(device)`
# instead, called at call time, with `device: str | None = None`.
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_device(device: str | None = None) -> str:
    """`device` if given, else "cuda"/"cpu" resolved fresh, at call time.

    Use as `device = resolve_device(device)` inside a function whose own
    parameter defaults to `None` -- NOT as `device: str = resolve_device()`
    in a signature, which would just reintroduce the same frozen-at-import
    problem this exists to fix.
    """
    return device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RunConfig:
    """One configuration in the hyperparameter sweep."""
    architecture: str    = DEFAULT_ARCHITECTURE
    diffusion_steps: int = DEFAULT_T
    hidden_dim: int      = DEFAULT_HIDDEN_DIM
    n_layers: int        = DEFAULT_N_LAYERS
    n_heads: int         = DEFAULT_N_HEADS
    batch_size: int      = BATCH_SIZE
    lr: float            = LEARNING_RATE
    max_epochs: int      = MAX_EPOCHS
    patience: int | None = EARLY_STOP_PATIENCE
    seed: int            = SEED

    @property
    def run_id(self) -> str:
        return f"{self.architecture}_T{self.diffusion_steps}_H{self.hidden_dim}_L{self.n_layers}"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RunConfig":
        """
        Load a RunConfig from a YAML file (see configs/transformer.yaml
        and configs/mlp.yaml for the validated defaults). Unknown keys
        raise rather than being silently ignored — a typo in a config
        file should fail loudly, not train the wrong thing quietly.
        """
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - valid_fields
        if unknown:
            raise ValueError(f"Unknown RunConfig field(s) in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save this configuration to YAML, e.g. to snapshot the exact settings used for a run."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(dataclasses.asdict(self), f, sort_keys=False)


def set_seed(seed: int = SEED) -> None:
    """Make every run reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
