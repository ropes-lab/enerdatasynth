"""
src/powerdiffusion/utils/config.py
====================================
Central configuration for the entire pipeline. Single source of truth
for paths, conditioning definitions, and reproducibility settings.

Per the PI's request, this baseline is intentionally simple:
  - conditioning: month (one-hot), dow (one-hot), holiday (binary),
                  Tmax (z-scored), Tmin (z-scored), spread (z-scored)
  - architecture: simple MLP
  - loss: standard diffusion MSE only

Anything not listed above (MW_anchor, Fourier modes, attention,
custom losses, unified model) is explicitly out of scope for this
baseline and is tracked only as future work.
"""

from __future__ import annotations
import random
import numpy as np
import torch
from dataclasses import dataclass, field
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT     = Path(__file__).resolve().parents[3]
RAW_DIR          = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR    = PROJECT_ROOT / "data" / "processed"
PROCESSED_BACKUP = PROJECT_ROOT / "data" / "processed_backup"
CHECKPOINT_DIR   = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR      = PROJECT_ROOT / "outputs"
GENERATED_DIR    = OUTPUTS_DIR / "generated"
LOGS_DIR         = OUTPUTS_DIR / "logs"
REPORTS_DIR      = PROJECT_ROOT / "reports"
FIGURES_DIR      = REPORTS_DIR / "figures"

for d in [RAW_DIR, PROCESSED_DIR, PROCESSED_BACKUP, CHECKPOINT_DIR,
          GENERATED_DIR, LOGS_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ZONE = "PS"   # PSE&G / New Jersey

# ─────────────────────────────────────────────────────────────────────────────
# Conditioning vector definition  (PI-approved baseline — DO NOT extend)
# ─────────────────────────────────────────────────────────────────────────────
# z = [month_onehot(12), dow_onehot(7), is_holiday(1), Tmax_z(1), Tmin_z(1), spread_z(1)]
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
    "user-facing abstractions", "unified PJM model", "attention",
    "transformers",
]

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter sweep grid
# ─────────────────────────────────────────────────────────────────────────────
SWEEP_DIFFUSION_STEPS = [50, 100, 200]
SWEEP_HIDDEN_DIM      = [64, 128, 256]
SWEEP_N_LAYERS        = [2, 4, 6]

DEFAULT_T          = 100
DEFAULT_HIDDEN_DIM = 128
DEFAULT_N_LAYERS   = 4

# ─────────────────────────────────────────────────────────────────────────────
# Training defaults
# ─────────────────────────────────────────────────────────────────────────────
SEED            = 42
BATCH_SIZE      = 64
LEARNING_RATE   = 3e-4
MAX_EPOCHS      = 200
EARLY_STOP_PATIENCE = 10   # set to None to disable early stopping
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RunConfig:
    """One configuration in the hyperparameter sweep."""
    diffusion_steps: int = DEFAULT_T
    hidden_dim: int      = DEFAULT_HIDDEN_DIM
    n_layers: int        = DEFAULT_N_LAYERS
    batch_size: int      = BATCH_SIZE
    lr: float            = LEARNING_RATE
    max_epochs: int      = MAX_EPOCHS
    patience: int | None = EARLY_STOP_PATIENCE
    seed: int            = SEED

    @property
    def run_id(self) -> str:
        return f"T{self.diffusion_steps}_H{self.hidden_dim}_L{self.n_layers}"


def set_seed(seed: int = SEED) -> None:
    """Make every run reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
