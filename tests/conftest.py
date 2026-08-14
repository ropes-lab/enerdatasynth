"""
tests/conftest.py
====================
Shared fixtures. Tests use small synthetic arrays rather than the
real (multi-hundred-MB) dataset under data/, so the suite runs
without requiring `enerdatasynth preprocess` to have been run first.

One real constraint, noted rather than hidden: `enerdatasynth.utils.
config` binds its path constants (CHECKPOINT_DIR, PROCESSED_DIR, ...)
at import time, and other modules do `from ...config import
CHECKPOINT_DIR` — a name binding, not a live attribute lookup — so
monkeypatching `config.CHECKPOINT_DIR` in a test does not redirect
code in already-imported modules. Tests that need a real trained
checkpoint (test_generators.py) therefore write to the project's
actual `checkpoints/`/`outputs/logs/` directories under a clearly
disposable run_id, and clean up in a fixture teardown. This is a
pragmatic tradeoff for a config module designed for a single-process
research pipeline, not a fully hermetic test setup.
"""

from __future__ import annotations
import shutil

import numpy as np
import pytest
import torch

from enerdatasynth.utils.config import CHECKPOINT_DIR, LOGS_DIR

N_DAYS = 40
X_DIM = 24
COND_DIM = 23


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def synthetic_profiles(rng):
    """A small, deterministic (X, Z) dataset shaped like the real pipeline's output."""
    X = rng.uniform(0, 1, size=(N_DAYS, X_DIM)).astype(np.float32)
    Z = rng.normal(size=(N_DAYS, COND_DIM)).astype(np.float32)
    return X, Z


@pytest.fixture
def tiny_trained_checkpoint(synthetic_profiles):
    """
    Trains a genuinely tiny model (a couple of epochs, small dims) via
    the real `train_one_config` so integration tests exercise the
    actual training/checkpoint/result.json code path, not a mock.
    Cleans up the checkpoint dir and log file it creates afterward.
    """
    from enerdatasynth.utils.config import RunConfig
    from enerdatasynth.resources.pjm_load.training.train import train_one_config

    X, Z = synthetic_profiles
    cfg = RunConfig(
        architecture="transformer", diffusion_steps=10, hidden_dim=8,
        n_layers=1, n_heads=2, max_epochs=2, patience=None, batch_size=8,
    )
    result = train_one_config(X[:32], Z[:32], X[32:], Z[32:], cfg, verbose=False)

    yield result

    shutil.rmtree(CHECKPOINT_DIR / result["run_id"], ignore_errors=True)
    log_path = LOGS_DIR / f"{result['run_id']}_training_log.csv"
    log_path.unlink(missing_ok=True)
