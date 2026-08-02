"""Tests for enerdatasynth.models: forward-pass shapes and the build_denoiser factory."""

from __future__ import annotations
import pytest
import torch

from enerdatasynth.models import build_denoiser
from enerdatasynth.models.denoiser import SimpleMLPDenoiser
from enerdatasynth.models.transformer import TransformerDenoiser

X_DIM, COND_DIM, BATCH = 24, 23, 5


def _forward_shape_ok(model):
    x_t = torch.randn(BATCH, X_DIM)
    t = torch.randint(0, 100, (BATCH,))
    z = torch.randn(BATCH, COND_DIM)
    eps_hat = model(x_t, t, z)
    assert eps_hat.shape == (BATCH, X_DIM)
    assert torch.isfinite(eps_hat).all()


def test_simple_mlp_denoiser_forward_shape():
    _forward_shape_ok(SimpleMLPDenoiser(x_dim=X_DIM, cond_dim=COND_DIM, hidden_dim=16, n_layers=2))


def test_transformer_denoiser_forward_shape():
    _forward_shape_ok(TransformerDenoiser(x_dim=X_DIM, cond_dim=COND_DIM, model_dim=16, num_heads=2, num_layers=1))


def test_transformer_denoiser_uses_x_dim_not_hardcoded_length():
    """Regression test for the module-global-H bug fixed during extraction from research_history."""
    model = TransformerDenoiser(x_dim=12, cond_dim=COND_DIM, model_dim=16, num_heads=2, num_layers=1)
    x_t = torch.randn(BATCH, 12)
    t = torch.randint(0, 100, (BATCH,))
    z = torch.randn(BATCH, COND_DIM)
    assert model(x_t, t, z).shape == (BATCH, 12)


@pytest.mark.parametrize("architecture,cls", [("transformer", TransformerDenoiser), ("mlp", SimpleMLPDenoiser)])
def test_build_denoiser_selects_architecture(architecture, cls):
    model = build_denoiser(architecture, x_dim=X_DIM, cond_dim=COND_DIM, hidden_dim=16, n_layers=1, n_heads=2)
    assert isinstance(model, cls)
    _forward_shape_ok(model)


def test_build_denoiser_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_denoiser("cnn", x_dim=X_DIM, cond_dim=COND_DIM)


def test_transformer_param_count_matches_validated_research():
    """The exact hyperparameters from research_history/reports/week7/ should reproduce 110,401 params."""
    model = TransformerDenoiser(x_dim=24, cond_dim=6, model_dim=64, num_heads=4, num_layers=2, t_embed_dim=64)
    assert model.count_parameters() == 110_401
