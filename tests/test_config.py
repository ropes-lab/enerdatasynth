"""Tests for pjm_diffusion.utils.config: RunConfig, run_id, seeding, YAML I/O."""

from __future__ import annotations
import numpy as np
import torch

from pjm_diffusion.utils.config import RunConfig, set_seed, DEFAULT_ARCHITECTURE


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
    """configs/transformer.yaml and configs/mlp.yaml (shipped with the repo) must actually parse."""
    from pjm_diffusion.utils.config import PROJECT_ROOT
    transformer_cfg = RunConfig.from_yaml(PROJECT_ROOT / "configs" / "transformer.yaml")
    mlp_cfg = RunConfig.from_yaml(PROJECT_ROOT / "configs" / "mlp.yaml")
    assert transformer_cfg == RunConfig()
    assert mlp_cfg.architecture == "mlp"
