"""Exception hierarchy for wind_diffusion's public API (M4).

All exceptions raised by `wind_diffusion.api` / `wind_diffusion.registry`
inherit from `WindDiffusionError`, so callers can catch broadly with
`except WindDiffusionError` or narrowly with a specific subclass.
"""

from __future__ import annotations


class WindDiffusionError(Exception):
    """Base class for all wind_diffusion library errors.

    Catch this to handle any failure raised by the public API
    (`wind_diffusion.api`, `wind_diffusion.registry`) without needing to
    enumerate every specific subclass.
    """


class CheckpointNotFoundError(WindDiffusionError):
    """No usable checkpoint/norm_stats could be resolved for a preset.

    Raised by `wind_diffusion.registry.resolve_weights` (and, transitively,
    `wind_diffusion.api.generate_wind_profile` / `preload`) when none of the
    following produced a usable pair of files: explicit `checkpoint=`/
    `norm_stats=` arguments, the `WIND_DIFFUSION_CHECKPOINTS` directory env
    var, or a downloadable entry in the bundled manifest. Messages raised at
    call sites name the preset, the paths/env vars that were checked, the
    `WIND_DIFFUSION_OFFLINE` flag's state, and the local training command
    (`wind-diffusion-train --preset <preset> ...`) as the fallback.
    """


class WeightsDownloadError(WindDiffusionError):
    """A weights download failed or the downloaded artifact was corrupt.

    Raised by `wind_diffusion.registry` when a download's sha256 does not
    match the manifest-declared checksum, or the download itself fails.
    Any partially-written file is removed before this is raised, so a
    cache directory never contains a corrupt/half-written artifact.
    """


class InvalidConditionError(WindDiffusionError):
    """A requested generation condition is not recognized.

    Raised by `wind_diffusion.api.generate_wind_profile` when, e.g.,
    `wind_regime` is not one of the checkpoint's `NormStats.wind_buckets`.
    Raised before any model loading or sampling work is performed.
    """
