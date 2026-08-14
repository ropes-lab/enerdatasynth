"""Small shared utilities."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make runs reproducible across numpy, python random, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a device string into a `torch.device`.

    "auto" -> cuda if available, else cpu. "cpu"/"cuda"/"cuda:0"/etc. pass
    through unchanged to `torch.device(...)`.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
