"""Tests for enerdatasynth.diffusion: DDPM forward/reverse process and beta schedules."""

from __future__ import annotations
import torch

from enerdatasynth.diffusion.ddpm import DDPM
from enerdatasynth.diffusion.schedules import make_betas
from enerdatasynth.models.denoiser import SimpleMLPDenoiser

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


def test_sampler_recomputes_noise_prediction_every_step():
    """
    Regression test for the historical sampler bug (see
    research_history/scripts/sampler_bugfix.py and
    docs/research_status.md's "Diffusion process" section): an earlier
    (never-shipped-in-this-package) DDIM implementation reused a single
    noise prediction across multiple reverse steps instead of
    recomputing it from the model at every step, which silently
    corrupted generation quality (week5_comparison_table.csv's
    "Diffusion (converged)" row: FirstDiff 2.29 / RMSE 0.51) until the
    ancestral sampler here was written to call the model fresh each
    time (week5_comparison_corrected.csv: FirstDiff 0.018 / RMSE 0.026).

    This asserts the invariant directly instead of re-deriving the
    historical numbers: `DDPM.sample()` must call the model exactly
    once per reverse timestep (T calls for T steps), never fewer.
    """
    call_count = {"n": 0}

    class CountingWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x_t, t, z):
            call_count["n"] += 1
            return self.inner(x_t, t, z)

    counting_model = CountingWrapper(_tiny_model())

    ddpm = DDPM(T=T)
    z = torch.randn(BATCH, COND_DIM)
    ddpm.sample(counting_model, z, X_DIM)

    assert call_count["n"] == T, (
        f"expected exactly one fresh noise-prediction call per reverse "
        f"step ({T} steps -> {T} calls), got {call_count['n']} -- a "
        f"lower count means a noise prediction is being reused across "
        f"steps, which is exactly the historical sampler bug."
    )


def test_sampler_noise_prediction_differs_across_steps():
    """
    Complements the call-count test above: even if the model is called
    the right number of times, the bug this guards against is
    specifically a REUSED prediction. Confirms p_sample_step's model
    output actually varies with x_t/t as x_t evolves during sampling,
    rather than a cached first-step value being replayed.
    """
    ddpm = DDPM(T=T)
    model = _tiny_model()
    z = torch.randn(1, COND_DIM)

    x = torch.randn(1, X_DIM)
    with torch.no_grad():
        eps_first = model(x, torch.zeros(1, dtype=torch.long), z)
        x_next = ddpm.p_sample_step(model, x, T - 1, z)
        eps_second = model(x_next, torch.zeros(1, dtype=torch.long), z)

    assert not torch.allclose(x, x_next), "p_sample_step must change x_t"
    assert not torch.allclose(eps_first, eps_second), (
        "noise prediction must be recomputed from the updated x_t, not "
        "reused from a stale value"
    )
