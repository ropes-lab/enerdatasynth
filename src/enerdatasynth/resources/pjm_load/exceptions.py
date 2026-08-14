"""Exception hierarchy for pjm_load's standalone API (`resources/pjm_load/api.py`),
mirroring `resources/wind/exceptions.py`'s pattern so both resources fail the same way
for the same kinds of problems.

Each subclasses the builtin exception type the rest of this codebase already raised for
that situation (`FileNotFoundError`, `ValueError`) — existing `except FileNotFoundError`/
`except ValueError` call sites keep working unchanged; new code can additionally catch
`PjmLoadError` or a specific subclass.
"""

from __future__ import annotations


class PjmLoadError(Exception):
    """Base class for all pjm_load library errors. Catch this to handle any
    failure raised by `resources.pjm_load.api` without enumerating subclasses."""


class CheckpointNotFoundError(PjmLoadError, FileNotFoundError):
    """No usable checkpoint could be resolved.

    Raised when `data/processed/pjm_load/metadata.json` is missing (run
    `enerdatasynth preprocess` first) or no trained checkpoint exists under
    `data/checkpoints/pjm_load/` (run `enerdatasynth train`). Subclasses
    `FileNotFoundError` for backward compatibility with existing callers.
    """


class InvalidConditionError(PjmLoadError, ValueError):
    """A requested generation condition is not recognized — e.g. `month` outside
    1-12, or an unknown `.set()`/`generate_load_profile()` keyword. Subclasses
    `ValueError` for backward compatibility with existing callers.
    """
