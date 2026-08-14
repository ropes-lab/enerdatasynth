"""Resource-scoped access point for PJM load's configuration, added for structural
parity with `resources/wind/config.py` (wind's config lives inside its resource
folder; load's canonical definitions lived only in the *shared* `utils/config.py`).

**This is a thin re-export layer, not a migration.** `PROJECT_ROOT`/`DATA_ROOT`/
`ensure_dirs()` genuinely need to stay in `utils/config.py` — both resources' path
resolution depends on them, and duplicating them per-resource would risk two
disagreeing copies. What's re-exported here is specifically the constants that are
load-specific in practice (conditioning shape, training defaults, load's own data
paths) even though they're physically defined in the shared module today. Import from
here if you want the load-scoped name; `utils.config` still works identically and nothing
that already imports from there needs to change.
"""

from __future__ import annotations

from enerdatasynth.utils.config import (
    RunConfig,
    COND_DIM,
    N_MONTHS,
    N_DOW,
    X_DIM,
    ZONE,
    DEFAULT_ARCHITECTURE,
    DEFAULT_T,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_N_LAYERS,
    DEFAULT_N_HEADS,
    RAW_DIR,
    PROCESSED_DIR,
    PROCESSED_BACKUP,
    CHECKPOINT_DIR,
    CONFIGS_DIR,
    OUTPUTS_DIR,
    GENERATED_DIR,
    LOGS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
)

__all__ = [
    "RunConfig", "COND_DIM", "N_MONTHS", "N_DOW", "X_DIM", "ZONE",
    "DEFAULT_ARCHITECTURE", "DEFAULT_T", "DEFAULT_HIDDEN_DIM", "DEFAULT_N_LAYERS", "DEFAULT_N_HEADS",
    "RAW_DIR", "PROCESSED_DIR", "PROCESSED_BACKUP", "CHECKPOINT_DIR",
    "CONFIGS_DIR", "OUTPUTS_DIR", "GENERATED_DIR", "LOGS_DIR", "REPORTS_DIR", "FIGURES_DIR",
]
