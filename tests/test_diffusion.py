"""Tests for pjm_diffusion.diffusion: DDPM forward/reverse process and beta schedules."""

from __future__ import annotations
import torch

from pjm_diffusion.diffusion.ddpm import DDPM
from pjm_diffusion.diffusion.schedules import make_betas
from pjm_diffusion.models.denoiser import SimpleMLPDenoiser

X_DIM, COND_DIM, BATCH, T = 24, 23, 6, 20


def _tiny_model():
    return SimpleMLPDenoiser(x_dim=X_DIM, cond_dim=COND_DIM, hidden_dim=8, n_layers=1)


def test_q_sample_shape_and_range():
    ddpm = DDPM(T=T)
    x0 = torch.rand(BATCH, X_DIM)
    t = torch.randint(0, T, (BATCH,))
    x_t, noise = ddpm.q_sample(x0, t)
    assert x_t.shape == x0.shape == noise.shape
    assert torch.isfinite(x_t).all()


def test_training_loss_is_finite_scalar():
    ddpm = DDPM(T=T)
    model = _tiny_model()
    x0 = torch.rand(BATCH, X_DIM)
    z = torch.randn(BATCH, COND_DIM)
    loss = ddpm.training_loss(model, x0, z)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_sample_and_generate_batch_shapes():
    ddpm = DDPM(T=T)
    model = _tiny_model()
    z = torch.randn(BATCH, COND_DIM)
    samples = ddpm.sample(model, z, X_DIM)
    assert samples.shape == (BATCH, X_DIM)
    assert torch.isfinite(samples).all()

    batch_samples = ddpm.generate_batch(model, z, X_DIM, batch_size=4)
    assert batch_samples.shape == (BATCH, X_DIM)


def test_linear_schedule_matches_ddpm_default():
    """DDPM's own linear schedule and schedules.make_betas('linear', ...) must agree exactly."""
    ddpm = DDPM(T=T, beta_start=1e-4, beta_end=0.02)
    betas = make_betas("linear", T=T, beta_start=1e-4, beta_end=0.02)
    assert torch.allclose(ddpm.betas, betas)


def test_schedules_are_valid_probabilities():
    for schedule in ["linear", "quadratic", "cosine"]:
        betas = make_betas(schedule, T=50)
        assert betas.shape == (50,)
        assert (betas > 0).all() and (betas < 1).all()


def test_unknown_schedule_raises():
    import pytest
    with pytest.raises(ValueError):
        make_betas("nonexistent", T=10)
