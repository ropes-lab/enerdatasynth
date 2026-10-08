"""Tests for enerdatasynth.utils.config: RunConfig, run_id, seeding, YAML I/O."""

from __future__ import annotations
import numpy as np
import torch

from enerdatasynth.utils.config import RunConfig, set_seed, DEFAULT_ARCHITECTURE


def test_run_config_defaults_match_validated_transformer():
    cfg = RunConfig()
    assert cfg.architecture == DEFAULT_ARCHITECTURE == "transformer"
    assert cfg.diffusion_steps == 1000
    assert cfg.hidden_dim == 64
    assert cfg.n_layers == 2
    assert cfg.n_heads == 4


def test_run_id_includes_architecture_to_avoid_collisions():
    transformer_cfg = RunConfig(architecture="transformer", diffusion_steps=100, hidden_dim=64, n_layers=2)
    mlp_cfg = RunConfig(architecture="mlp", diffusion_steps=100, hidden_dim=64, n_layers=2)
    assert transformer_cfg.run_id != mlp_cfg.run_id
    assert transformer_cfg.run_id == "transformer_T100_H64_L2"
    assert mlp_cfg.run_id == "mlp_T100_H64_L2"


def test_set_seed_makes_torch_and_numpy_reproducible():
    set_seed(123)
    a = (torch.randn(5), np.random.rand(5))
    set_seed(123)
    b = (torch.randn(5), np.random.rand(5))
    assert torch.equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_yaml_round_trip(tmp_path):
    cfg = RunConfig(architecture="mlp", diffusion_steps=50, hidden_dim=32, n_layers=3, seed=7)
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    reloaded = RunConfig.from_yaml(path)
    assert reloaded == cfg


def test_from_yaml_rejects_unknown_fields(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("architecture: transformer\nnonexistent_field: 1\n")
    try:
        RunConfig.from_yaml(path)
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "nonexistent_field" in str(e)


def test_shipped_configs_load(tmp_path):
    """data/configs/transformer.yaml and data/configs/mlp.yaml (shipped with the repo) must actually parse."""
    from enerdatasynth.utils.config import CONFIGS_DIR
    transformer_cfg = RunConfig.from_yaml(CONFIGS_DIR / "transformer.yaml")
    mlp_cfg = RunConfig.from_yaml(CONFIGS_DIR / "mlp.yaml")
    assert transformer_cfg == RunConfig()
    assert mlp_cfg.architecture == "mlp"


def test_project_root_is_not_derived_from_installed_package_location():
    """
    Regression test for a real bug caught by `pip install dist/*.whl`
    verification: PROJECT_ROOT used to be `Path(__file__).resolve()
    .parents[3]`, which only works for an editable install (source
    stays put next to data/). A real (non-editable) install copies the
    code into site-packages, far from any project directory, so that
    resolution silently produced a nonsense path there. PROJECT_ROOT
    must now come from $ENERDATASYNTH_PROJECT_ROOT or cwd — verified
    here via a real subprocess launched from a different cwd, since
    PROJECT_ROOT is fixed at import time within a single process.
    """
    import os
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as elsewhere:
        result = subprocess.run(
            [sys.executable, "-c", "from enerdatasynth.utils.config import PROJECT_ROOT; print(PROJECT_ROOT)"],
            cwd=elsewhere, capture_output=True, text=True, check=True,
        )
        # Resolve both sides through realpath: tempfile dirs are often
        # themselves symlinks (e.g. /tmp -> /private/tmp on macOS), and
        # PROJECT_ROOT is built from a Path.resolve() call, so a naive
        # string comparison would spuriously fail on such platforms.
        assert os.path.realpath(result.stdout.strip()) == os.path.realpath(elsewhere)

        # Explicit override takes precedence over cwd. Uses a second
        # real, writable tempdir (not a fake root-level path) since
        # importing config.py creates data/checkpoints/etc. under
        # PROJECT_ROOT as a side effect, which would fail with
        # PermissionError against an unwritable made-up path.
        with tempfile.TemporaryDirectory() as override_target:
            env = {**os.environ, "ENERDATASYNTH_PROJECT_ROOT": override_target}
            result2 = subprocess.run(
                [sys.executable, "-c", "from enerdatasynth.utils.config import PROJECT_ROOT; print(PROJECT_ROOT)"],
                cwd=elsewhere, capture_output=True, text=True, check=True, env=env,
            )
            assert os.path.realpath(result2.stdout.strip()) == os.path.realpath(override_target)


def test_project_relative_strips_the_project_root_only():
    # checkpoint metadata must not embed one machine's home directory
    from enerdatasynth.utils.config import PROJECT_ROOT, project_relative

    inside = PROJECT_ROOT / "data" / "checkpoints" / "pjm_load" / "run" / "best.pt"
    assert project_relative(inside) == "data/checkpoints/pjm_load/run/best.pt"
    # a path outside the project is returned untouched rather than mangled
    assert project_relative("/somewhere/else/best.pt") == "/somewhere/else/best.pt"
