"""Tests for the sampler abstraction (enerdatasynth.resources.wind.diffusion.samplers), the device
resolution helper, and the process-level model cache in evaluate.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from enerdatasynth.resources.wind.diffusion.diffusion import DiffusionSchedule, _cfg_predict, clamp_x0
from enerdatasynth.resources.wind.models.models import TransformerDenoiser
from enerdatasynth.resources.wind.diffusion.samplers import sample
from enerdatasynth.resources.wind.utils import resolve_device

SEQ_LEN = 16
HIDDEN_DIM = 8
COND_DIM = 4


def make_tiny_model(n_channels: int = 2) -> TransformerDenoiser:
    return TransformerDenoiser(
        n_channels=n_channels,
        cond_dim=COND_DIM,
        hidden_dim=HIDDEN_DIM,
        num_conv_blocks=1,
        num_transformer_layers=1,
        num_heads=2,
        kernel_size=3,
        dropout=0.0,
        time_emb_dim=8,
        cond_emb_dim=8,
        seq_len=SEQ_LEN,
    )


def make_tiny_schedule(T: int = 20, device: str = "cpu") -> DiffusionSchedule:
    return DiffusionSchedule.build(T=T, beta_start=1e-4, beta_end=0.02, beta_schedule="cosine", device=device)


def make_clip_bounds(n_channels: int, device: str = "cpu"):
    clip_min = torch.full((1, n_channels, 1), -3.5, device=device)
    clip_max = torch.full((1, n_channels, 1), 3.5, device=device)
    return clip_min, clip_max


# ---------------------------------------------------------------------------
# Shape correctness
# ---------------------------------------------------------------------------


def test_ddim_sample_shape():
    n_channels = 2
    model = make_tiny_model(n_channels)
    schedule = make_tiny_schedule(T=20)
    clip_min, clip_max = make_clip_bounds(n_channels)

    n_conditions = 5
    cond_matrix = np.random.randn(n_conditions, COND_DIM).astype(np.float32)

    out = sample(
        model,
        schedule,
        cond_matrix,
        n_channels,
        SEQ_LEN,
        clip_min,
        clip_max,
        device="cpu",
        n_repeats=3,
        sampler="ddim",
        steps=10,
    )

    assert out.shape == (n_conditions * 3, n_channels, SEQ_LEN)
    assert isinstance(out, np.ndarray)


def test_ancestral_sample_shape():
    n_channels = 1
    model = make_tiny_model(n_channels)
    schedule = make_tiny_schedule(T=15)
    clip_min, clip_max = make_clip_bounds(n_channels)

    n_conditions = 2
    cond_matrix = np.random.randn(n_conditions, COND_DIM).astype(np.float32)

    out = sample(
        model,
        schedule,
        cond_matrix,
        n_channels,
        SEQ_LEN,
        clip_min,
        clip_max,
        device="cpu",
        n_repeats=2,
        sampler="ancestral",
    )

    assert out.shape == (n_conditions * 2, n_channels, SEQ_LEN)


# ---------------------------------------------------------------------------
# DDIM makes strictly fewer model calls than ancestral
# ---------------------------------------------------------------------------


class _CallCountingModel(torch.nn.Module):
    """Wraps a TransformerDenoiser and counts forward() invocations."""

    def __init__(self, inner: TransformerDenoiser):
        super().__init__()
        self.inner = inner
        self.call_count = 0

    def forward(self, x, t, z):
        self.call_count += 1
        return self.inner(x, t, z)

    def eval(self):
        self.inner.eval()
        return super().eval()

    def train(self, mode: bool = True):
        self.inner.train(mode)
        return super().train(mode)


def test_ddim_calls_model_fewer_times_than_ancestral():
    n_channels = 2
    T = 40
    schedule = make_tiny_schedule(T=T)
    clip_min, clip_max = make_clip_bounds(n_channels)
    cond_matrix = np.random.randn(3, COND_DIM).astype(np.float32)

    ancestral_model = _CallCountingModel(make_tiny_model(n_channels))
    sample(
        ancestral_model,
        schedule,
        cond_matrix,
        n_channels,
        SEQ_LEN,
        clip_min,
        clip_max,
        device="cpu",
        sampler="ancestral",
    )
    assert ancestral_model.call_count == T

    ddim_model = _CallCountingModel(make_tiny_model(n_channels))
    sample(
        ddim_model,
        schedule,
        cond_matrix,
        n_channels,
        SEQ_LEN,
        clip_min,
        clip_max,
        device="cpu",
        sampler="ddim",
        steps=10,
    )
    assert ddim_model.call_count <= 10
    assert ddim_model.call_count < ancestral_model.call_count


# ---------------------------------------------------------------------------
# Classifier-free guidance (_cfg_predict / guidance_scale)
# ---------------------------------------------------------------------------


def test_cfg_predict_guidance_scale_one_is_single_call_pass_through():
    """guidance_scale=1.0 must be a pure pass-through: exactly one model
    call, identical output to calling the model directly. This is the
    backward-compatibility guarantee -- every checkpoint trained before
    guidance_scale existed must behave exactly as before."""
    model = _CallCountingModel(make_tiny_model(n_channels=2))
    model.eval()
    x = torch.randn(3, 2, SEQ_LEN)
    t = torch.zeros(3, dtype=torch.long)
    z = torch.randn(3, COND_DIM)

    with torch.no_grad():
        direct = model.inner(x, t, z)
    model.call_count = 0

    with torch.no_grad():
        guided = _cfg_predict(model, x, t, z, guidance_scale=1.0)

    assert model.call_count == 1
    assert torch.allclose(guided, direct)


def test_cfg_predict_guidance_scale_not_one_makes_two_calls_and_differs():
    model = _CallCountingModel(make_tiny_model(n_channels=2))
    model.eval()
    x = torch.randn(3, 2, SEQ_LEN)
    t = torch.zeros(3, dtype=torch.long)
    z = torch.randn(3, COND_DIM)

    with torch.no_grad():
        pred_cond = model.inner(x, t, z)
    model.call_count = 0

    with torch.no_grad():
        guided = _cfg_predict(model, x, t, z, guidance_scale=2.0)

    assert model.call_count == 2
    # guidance_scale=2.0 extrapolates away from the unconditional prediction;
    # for a freshly-initialized (non-degenerate) model that should move the
    # output away from the plain conditional prediction.
    assert not torch.allclose(guided, pred_cond)


def test_cfg_predict_guidance_slice_only_zeroes_that_slice_in_uncond_branch():
    """guidance_slice=slice(1,3) must zero only columns 1:3 of z for the
    'uncond' branch's model call, capturing the model as it's actually
    invoked -- not just re-deriving the expected z by slicing outside."""
    model = make_tiny_model(n_channels=2)
    model.eval()
    x = torch.randn(3, 2, SEQ_LEN)
    t = torch.zeros(3, dtype=torch.long)
    z = torch.randn(3, COND_DIM)

    seen_z = []
    original_forward = model.forward

    def spy_forward(x_in, t_in, z_in):
        seen_z.append(z_in.clone())
        return original_forward(x_in, t_in, z_in)

    model.forward = spy_forward

    with torch.no_grad():
        _cfg_predict(model, x, t, z, guidance_scale=2.0, guidance_slice=slice(1, 3))

    assert len(seen_z) == 2
    cond_call, uncond_call = seen_z
    assert torch.allclose(cond_call, z)  # conditional branch is untouched
    assert torch.allclose(uncond_call[:, 0], z[:, 0])  # outside the slice: unchanged
    assert torch.all(uncond_call[:, 1:3] == 0.0)  # inside the slice: zeroed
    assert torch.allclose(uncond_call[:, 3], z[:, 3])  # outside the slice: unchanged


def test_sample_ddim_accepts_guidance_scale_and_preserves_shape():
    n_channels = 2
    model = make_tiny_model(n_channels)
    schedule = make_tiny_schedule(T=20)
    clip_min, clip_max = make_clip_bounds(n_channels)
    cond_matrix = np.random.randn(3, COND_DIM).astype(np.float32)

    out = sample(
        model,
        schedule,
        cond_matrix,
        n_channels,
        SEQ_LEN,
        clip_min,
        clip_max,
        device="cpu",
        sampler="ddim",
        steps=10,
        guidance_scale=2.5,
    )
    assert out.shape == (3, n_channels, SEQ_LEN)


def test_sample_ddim_guidance_scale_doubles_model_calls():
    n_channels = 2
    schedule = make_tiny_schedule(T=20)
    clip_min, clip_max = make_clip_bounds(n_channels)
    cond_matrix = np.random.randn(3, COND_DIM).astype(np.float32)

    unguided_model = _CallCountingModel(make_tiny_model(n_channels))
    sample(
        unguided_model, schedule, cond_matrix, n_channels, SEQ_LEN, clip_min, clip_max,
        device="cpu", sampler="ddim", steps=10, guidance_scale=1.0,
    )

    guided_model = _CallCountingModel(make_tiny_model(n_channels))
    sample(
        guided_model, schedule, cond_matrix, n_channels, SEQ_LEN, clip_min, clip_max,
        device="cpu", sampler="ddim", steps=10, guidance_scale=2.0,
    )

    assert guided_model.call_count == 2 * unguided_model.call_count


# ---------------------------------------------------------------------------
# clamp_x0 (dynamic thresholding fix for the clip-boundary lock-in)
# ---------------------------------------------------------------------------


def test_clamp_x0_static_is_a_pure_clamp():
    """dynamic_threshold=False must behave exactly like plain torch.clamp --
    every checkpoint's behavior before this parameter existed."""
    x0 = torch.tensor([[[0.5, 5.0, -5.0, 2.0]]])
    clip_min = torch.tensor([[[-3.5]]])
    clip_max = torch.tensor([[[3.5]]])
    out = clamp_x0(x0, clip_min, clip_max, dynamic_threshold=False)
    assert torch.allclose(out, torch.clamp(x0, clip_min, clip_max))


def test_clamp_x0_dynamic_rescales_proportionally_instead_of_truncating():
    """The failure mode this fixes: static clamp truncates an outlier to
    exactly the boundary (where it can get stuck across many reverse steps).
    Dynamic thresholding should instead rescale the whole prediction so nothing
    gets flattened to a hard wall -- the relative shape between elements is
    preserved, just uniformly compressed."""
    x0 = torch.tensor([[[1.0, 2.0, 7.0]]])
    clip_min = torch.tensor([[[-3.5]]])
    clip_max = torch.tensor([[[3.5]]])
    out = clamp_x0(x0, clip_min, clip_max, dynamic_threshold=True, dynamic_threshold_percentile=100.0)
    # s = max(|x0|) = 7.0 (exceeds the static bound, so it's NOT floored) ->
    # rescale factor = static_bound / s = 3.5 / 7.0 = 0.5
    expected = x0 * 0.5
    assert torch.allclose(out, expected)
    assert out.abs().max().item() == pytest.approx(3.5)
    # unlike a static clamp, the largest and smallest elements keep their
    # relative ratio (2:1 before, 2:1 after) instead of both being truncated
    # to the same boundary value
    assert (out[0, 0, 1] / out[0, 0, 0]).item() == pytest.approx(2.0)


def test_clamp_x0_dynamic_floors_at_static_bound_when_prediction_is_well_behaved():
    """When nothing needs clamping, dynamic thresholding must not invent a
    tighter bound than the static one -- it only ever tightens an
    already-violated bound, never loosens or over-corrects a fine prediction."""
    x0 = torch.tensor([[[0.1, 0.2, -0.15]]])
    clip_min = torch.tensor([[[-3.5]]])
    clip_max = torch.tensor([[[3.5]]])
    out = clamp_x0(x0, clip_min, clip_max, dynamic_threshold=True, dynamic_threshold_percentile=100.0)
    assert torch.allclose(out, x0)


def test_sample_ddim_accepts_dynamic_threshold_and_preserves_shape():
    n_channels = 2
    model = make_tiny_model(n_channels)
    schedule = make_tiny_schedule(T=20)
    clip_min, clip_max = make_clip_bounds(n_channels)
    cond_matrix = np.random.randn(3, COND_DIM).astype(np.float32)

    out = sample(
        model, schedule, cond_matrix, n_channels, SEQ_LEN, clip_min, clip_max,
        device="cpu", sampler="ddim", steps=10, dynamic_threshold=True, dynamic_threshold_percentile=99.0,
    )
    assert out.shape == (3, n_channels, SEQ_LEN)


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


def test_resolve_device_auto_returns_torch_device():
    result = resolve_device("auto")
    assert isinstance(result, torch.device)
    if not torch.cuda.is_available():
        assert result.type == "cpu"
    else:
        assert result.type == "cuda"


def test_resolve_device_explicit_cpu():
    result = resolve_device("cpu")
    assert isinstance(result, torch.device)
    assert result.type == "cpu"


# ---------------------------------------------------------------------------
# Unknown sampler
# ---------------------------------------------------------------------------


def test_unknown_sampler_raises_value_error():
    n_channels = 1
    model = make_tiny_model(n_channels)
    schedule = make_tiny_schedule(T=10)
    clip_min, clip_max = make_clip_bounds(n_channels)
    cond_matrix = np.random.randn(1, COND_DIM).astype(np.float32)

    with pytest.raises(ValueError):
        sample(
            model,
            schedule,
            cond_matrix,
            n_channels,
            SEQ_LEN,
            clip_min,
            clip_max,
            device="cpu",
            sampler="not_a_real_sampler",
        )


# ---------------------------------------------------------------------------
# get_cached_model
# ---------------------------------------------------------------------------


def _make_fake_checkpoint(tmp_path, n_channels: int = 2, cond_dim: int = COND_DIM):
    """Build and save a tiny real checkpoint compatible with
    load_checkpoint_and_model's expectations (config dict + state dict with
    an inferable cond_mlp.net.0.weight shape).
    """
    from enerdatasynth.resources.wind.config import get_config
    from enerdatasynth.resources.wind.training.train import build_model

    config = get_config(
        "2ch",
        hidden_dim=HIDDEN_DIM,
        num_conv_blocks=1,
        num_transformer_layers=1,
        num_heads=2,
        kernel_size=3,
        time_emb_dim=8,
        cond_emb_dim=8,
        seq_len=SEQ_LEN,
        diffusion_timesteps=10,
    )
    model = build_model(config, config.n_model_channels, cond_dim=cond_dim, seq_len=config.seq_len)

    ckpt_path = tmp_path / "tiny_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "epoch": 1,
            "val_loss": 0.0,
        },
        ckpt_path,
    )
    return str(ckpt_path)


def test_get_cached_model_returns_same_object_and_loads_once(tmp_path, monkeypatch):
    import enerdatasynth.resources.wind.evaluation.evaluate as evaluate_module

    ckpt_path = _make_fake_checkpoint(tmp_path)

    # Reset the module-level cache so this test is independent of others.
    evaluate_module._MODEL_CACHE.clear()

    load_calls = {"count": 0}
    real_torch_load = torch.load

    def counting_torch_load(*args, **kwargs):
        load_calls["count"] += 1
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", counting_torch_load)

    result1 = evaluate_module.get_cached_model(ckpt_path, device="cpu")
    result2 = evaluate_module.get_cached_model(ckpt_path, device="cpu")

    assert result1 is result2
    assert load_calls["count"] == 1
