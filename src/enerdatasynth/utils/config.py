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
#
# Every category folder is resource-nested (data/<category>/<resource>/)
# so a third resource (e.g. water) mirrors this exactly by adding one
# same-named subfolder per category, not a parallel top-level tree. The
# constants below are PJM load's paths specifically -- wind's equivalent
# paths are computed independently where needed (see
# generators/wind_conditioning.py), since wind has no from-scratch
# pipeline in this package to mirror most of these (no RAW_DIR,
# PROCESSED_DIR, etc. equivalent -- it ships pre-trained; see
# docs/models/wind.md).
#
# Zones. PJM load has one trained model per zone. The default zone, PS, keeps
# the original flat layout (data/<category>/pjm_load/...), so everything that
# existed before multi-zone support is exactly where it was. Every other zone
# lives in a `zones/<ZONE>/` subfolder of the same category folder. Which zone
# the training/evaluation commands work on is chosen with `--zone` on the CLI
# (which sets $ENERDATASYNTH_ZONE before this module loads) or by setting that
# variable yourself; generation (`enerdatasynth.model("pjm_load", zone=...)`)
# takes the zone as an argument and does not depend on it.
DEFAULT_ZONE = "PS"   # PSE&G / New Jersey
PJM_ZONES = [
    "PS", "AE", "AP", "BC", "CE", "DAY", "DEOK", "DOM", "DPL", "DUQ",
    "EKPC", "JC", "ME", "PE", "PL", "PN", "RECO",
]
ZONE = os.environ.get("ENERDATASYNTH_ZONE", DEFAULT_ZONE).strip().upper()
if ZONE not in PJM_ZONES:
    raise ValueError(f"ENERDATASYNTH_ZONE={ZONE!r} is not a PJM zone this package supports: {PJM_ZONES}")


def zone_dir(base: Path, zone: str | None = None) -> Path:
    """`base` for the default zone, `base/zones/<ZONE>` for any other."""
    zone = (zone or ZONE).upper()
    return base if zone == DEFAULT_ZONE else base / "zones" / zone


DATA_ROOT        = PROJECT_ROOT / "data"
CONFIGS_DIR      = DATA_ROOT / "configs" / "pjm_load"
RAW_DIR          = DATA_ROOT / "raw" / "pjm_load"
PROCESSED_DIR    = zone_dir(DATA_ROOT / "processed" / "pjm_load")
PROCESSED_BACKUP = zone_dir(DATA_ROOT / "processed" / "pjm_load_backup")
CHECKPOINT_DIR   = zone_dir(DATA_ROOT / "checkpoints" / "pjm_load")
OUTPUTS_DIR      = zone_dir(DATA_ROOT / "outputs" / "pjm_load")
GENERATED_DIR    = OUTPUTS_DIR / "generated"
LOGS_DIR         = OUTPUTS_DIR / "logs"
REPORTS_DIR      = zone_dir(DATA_ROOT / "reports" / "pjm_load")
FIGURES_DIR      = REPORTS_DIR / "figures"
RESEARCH_HISTORY_DIR = DATA_ROOT / "research_history" / "pjm_load"

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


# Files shipped inside the installed package (see setup.py): the default
# trained checkpoints and the scaler file they need. This one path *is*
# derived from `__file__`, because it locates package data, not the user's
# project. The project's own data/ tree always wins; these are the fallback
# that makes a plain `pip install` able to generate without a repo clone.
BUNDLED_ROOT           = Path(__file__).resolve().parent.parent / "_bundled"
BUNDLED_CHECKPOINT_DIR = BUNDLED_ROOT / "checkpoints" / "pjm_load"
BUNDLED_WIND_DIR       = BUNDLED_ROOT / "checkpoints" / "wind"
BUNDLED_PROCESSED_DIR  = BUNDLED_ROOT / "processed" / "pjm_load"


def _project_zone_dir(current: Path, zone: str | None) -> Path:
    """`current` (the working zone's folder, e.g. `CHECKPOINT_DIR`) as it is for
    `zone`: unchanged for the working zone, the sibling zone folder otherwise."""
    zone = (zone or ZONE).upper()
    if zone == ZONE:
        return current
    flat = current if ZONE == DEFAULT_ZONE else current.parent.parent
    return zone_dir(flat, zone)


def checkpoint_search_dirs(zone: str | None = None) -> list[Path]:
    """Where to look for trained load checkpoints of `zone` (default: the
    working zone), in priority order: the project's own, then the bundled."""
    return [_project_zone_dir(CHECKPOINT_DIR, zone), zone_dir(BUNDLED_CHECKPOINT_DIR, zone)]


def find_checkpoint_dir(run_id: str, zone: str | None = None) -> Path:
    """Directory holding `run_id`'s files for `zone`: the project's own copy if
    it has one, otherwise the bundled one. Returns the project path when neither
    exists, so the caller's "not found" message names a sensible location."""
    roots = checkpoint_search_dirs(zone)
    for root in roots:
        if (root / run_id / "result.json").exists():
            return root / run_id
    return roots[0] / run_id


def processed_metadata_path(zone: str | None = None) -> Path:
    """`metadata.json` for `zone` (default: the working zone) from the project's
    `data/processed/pjm_load/` if present, else the copy bundled in the package."""
    project = _project_zone_dir(PROCESSED_DIR, zone) / "metadata.json"
    return project if project.exists() else zone_dir(BUNDLED_PROCESSED_DIR, zone) / "metadata.json"


def available_zones() -> list[str]:
    """Zones that have both a scaler file and a trained checkpoint, in
    `PJM_ZONES` order (project files and bundled files both count)."""
    found = []
    for z in PJM_ZONES:
        if not processed_metadata_path(z).exists():
            continue
        if any(root.exists() and any(root.glob("*/result.json")) for root in checkpoint_search_dirs(z)):
            found.append(z)
    return found


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
#   data/research_history/pjm_load/reports/week7/summary_report.md). model_dim=64,
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



def project_relative(path: str | Path) -> str:
    """`path` as a POSIX string relative to PROJECT_ROOT when it lives under it, else unchanged.

    Used when recording file locations inside checkpoint metadata, so committed
    `result.json`/`calibration.json` files don't embed one machine's home directory.
    """
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)



def processed_path(filename: str) -> Path:
    """`PROCESSED_DIR / filename`, or an error explaining how to create it.

    This repository ships no real PJM data (only `metadata.json`, the scalers the
    shipped checkpoint needs). Anything that trains, evaluates, or calibrates needs
    the processed files, which you build from your own copy of the source data.
    """
    path = PROCESSED_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. This repository ships no real data: get the PJM load and "
            f"NOAA temperature files yourself (see data/raw/pjm_load/README.md), put them in "
            f"data/raw/pjm_load/, then run `enerdatasynth preprocess`."
        )
    return path


def load_processed(filename: str) -> np.ndarray:
    """Load one `.npy` array from `PROCESSED_DIR` (see `processed_path`)."""
    return np.load(processed_path(filename))


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
