from __future__ import annotations

import importlib

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from wind_diffusion.config import get_config
from wind_diffusion.diffusion import DiffusionSchedule, q_sample
from wind_diffusion.models import TransformerDenoiser
from wind_diffusion.train import _joint_radius_consistency_loss, build_lr_scheduler, build_model, run_epoch
from wind_diffusion.utils import resolve_device

# wind_diffusion/__init__.py does `from wind_diffusion.train import ..., train`,
# which overwrites the `train` attribute on the `wind_diffusion` package with
# the `train()` function -- shadowing the `wind_diffusion.train` *submodule*
# on that same attribute. `import wind_diffusion.train as train_module` still
# resolves through that shadowed attribute and gets the function, not the
# module, so importlib is used here to fetch the actual module object.
train_module = importlib.import_module("wind_diffusion.train")


def _tiny_model_and_schedule(n_channels: int = 1) -> tuple[TransformerDenoiser, DiffusionSchedule]:
    model = TransformerDenoiser(
        n_channels=n_channels,
        cond_dim=4,
        hidden_dim=8,
        num_conv_blocks=1,
        num_transformer_layers=1,
        num_heads=2,
        kernel_size=3,
        dropout=0.0,
        time_emb_dim=8,
        cond_emb_dim=8,
        seq_len=16,
        norm_first=False,
    )
    schedule = DiffusionSchedule.build(T=10, beta_start=1e-4, beta_end=0.02, beta_schedule="cosine", device="cpu")
    return model, schedule


def test_config_device_defaults_to_auto():
    config = get_config("1ch")
    assert config.device == "auto"


def test_config_norm_first_defaults_to_false():
    config = get_config("1ch")
    assert config.norm_first is False


def test_config_warmup_epochs_defaults_to_five():
    config = get_config("1ch")
    assert config.warmup_epochs == 5


def test_config_cond_dropout_prob_defaults_to_point_three():
    config = get_config("1ch")
    assert config.cond_dropout_prob == 0.3


def test_config_cond_dropout_scope_defaults_to_wind_regime():
    config = get_config("1ch")
    assert config.cond_dropout_scope == "wind_regime"


def test_config_joint_consistency_defaults_disabled():
    config = get_config("1ch")
    assert config.joint_consistency_channels is None


def test_config_wind_persistence_dropout_prob_defaults_to_point_three():
    config = get_config("1ch")
    assert config.wind_persistence_dropout_prob == 0.3


def test_config_wind_persistence_condition_on_absent_by_default():
    """wind_persistence must be opt-in -- absent from the default condition_on
    dict, not just False, so .get() based callers keep working unchanged."""
    config = get_config("1ch")
    assert "wind_persistence" not in config.condition_on
    assert config.joint_consistency_weight == 0.0


def test_build_model_respects_norm_first_true():
    config = get_config("1ch", hidden_dim=16, num_conv_blocks=1, num_transformer_layers=1, num_heads=2, norm_first=True)
    model = build_model(config, n_channels=1, cond_dim=5, seq_len=16)
    layer = model.transformer.layers[0]
    assert layer.norm_first is True


def test_build_model_respects_norm_first_false():
    config = get_config("1ch", hidden_dim=16, num_conv_blocks=1, num_transformer_layers=1, num_heads=2, norm_first=False)
    model = build_model(config, n_channels=1, cond_dim=5, seq_len=16)
    layer = model.transformer.layers[0]
    assert layer.norm_first is False


def test_transformer_denoiser_forward_shape_unaffected_by_norm_first():
    """norm_first is purely an internal LN placement change -- output shape
    (and the fact that it runs at all) must be identical either way."""
    for norm_first in (True, False):
        model = TransformerDenoiser(
            n_channels=2,
            cond_dim=5,
            hidden_dim=16,
            num_conv_blocks=1,
            num_transformer_layers=1,
            num_heads=2,
            kernel_size=3,
            dropout=0.0,
            time_emb_dim=8,
            cond_emb_dim=8,
            seq_len=16,
            norm_first=norm_first,
        )
        x = torch.randn(3, 2, 16)
        t = torch.randint(0, 200, (3,))
        z = torch.randn(3, 5)
        out = model(x, t, z)
        assert out.shape == (3, 2, 16)


def test_lr_scheduler_warms_up_then_decays():
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_lr_scheduler(optimizer, num_epochs=10, warmup_epochs=3)

    lrs = []
    for _ in range(10):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # Warmup: LR strictly increases for the first `warmup_epochs` steps.
    assert lrs[0] < lrs[1] < lrs[2]
    # After warmup, cosine decay: LR at the end is lower than the post-warmup peak.
    assert lrs[-1] < max(lrs)
    # Never exceeds the configured base LR (up to float slop).
    assert max(lrs) <= 1e-3 * 1.01


def test_lr_scheduler_zero_warmup_is_plain_cosine():
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_lr_scheduler(optimizer, num_epochs=10, warmup_epochs=0)

    lrs = []
    for _ in range(10):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    # No warmup ramp: LR should already be at (near) the base value on step 0,
    # then monotonically decrease (cosine decay, no warmup bump).
    assert lrs[0] == 1e-3
    assert lrs[-1] < lrs[0]


def test_run_epoch_zeroes_condition_when_cond_dropout_prob_is_one():
    """cond_dropout_prob=1.0 during training should zero every sample's z
    before it reaches the model -- the extreme case that makes the effect
    unambiguous to assert on (a partial probability would be flaky)."""
    torch.manual_seed(0)
    model, schedule = _tiny_model_and_schedule()
    x = torch.randn(6, 1, 16)
    z = torch.ones(6, 4)  # nonzero condition so dropout is observable
    loader = DataLoader(TensorDataset(x, z), batch_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    seen_z = []
    original_forward = model.forward

    def spy_forward(x_in, t_in, z_in):
        seen_z.append(z_in.clone())
        return original_forward(x_in, t_in, z_in)

    model.forward = spy_forward

    run_epoch(model, loader, schedule, optimizer, loss_fn, torch.device("cpu"), train=True, cond_dropout_prob=1.0)

    assert len(seen_z) == 2  # two batches of size 3
    for z_batch in seen_z:
        assert torch.all(z_batch == 0.0)


def test_run_epoch_never_drops_condition_during_validation():
    """cond_dropout_prob must not affect val_loss -- it should keep measuring
    the checkpoint's real conditional performance regardless of the training
    dropout setting."""
    torch.manual_seed(0)
    model, schedule = _tiny_model_and_schedule()
    x = torch.randn(6, 1, 16)
    z = torch.ones(6, 4)
    loader = DataLoader(TensorDataset(x, z), batch_size=3)
    loss_fn = torch.nn.MSELoss()

    seen_z = []
    original_forward = model.forward

    def spy_forward(x_in, t_in, z_in):
        seen_z.append(z_in.clone())
        return original_forward(x_in, t_in, z_in)

    model.forward = spy_forward

    run_epoch(model, loader, schedule, None, loss_fn, torch.device("cpu"), train=False, cond_dropout_prob=1.0)

    assert len(seen_z) == 2
    for z_batch in seen_z:
        assert torch.all(z_batch == 1.0)


def test_run_epoch_cond_dropout_slice_zeroes_only_that_slice():
    """cond_dropout_slice=slice(1,3) with cond_dropout_prob=1.0 should zero
    only columns 1:3 of every dropped sample's z, leaving the rest of the
    4-dim condition vector (columns 0 and 3) untouched -- this is the
    regime-only guidance mechanism, as opposed to zeroing the whole vector."""
    torch.manual_seed(0)
    model, schedule = _tiny_model_and_schedule()
    x = torch.randn(6, 1, 16)
    z = torch.ones(6, 4)
    loader = DataLoader(TensorDataset(x, z), batch_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    seen_z = []
    original_forward = model.forward

    def spy_forward(x_in, t_in, z_in):
        seen_z.append(z_in.clone())
        return original_forward(x_in, t_in, z_in)

    model.forward = spy_forward

    run_epoch(
        model, loader, schedule, optimizer, loss_fn, torch.device("cpu"), train=True,
        cond_dropout_prob=1.0, cond_dropout_slice=slice(1, 3),
    )

    assert len(seen_z) == 2
    for z_batch in seen_z:
        assert torch.all(z_batch[:, 0] == 1.0)
        assert torch.all(z_batch[:, 1:3] == 0.0)
        assert torch.all(z_batch[:, 3] == 1.0)


def test_run_epoch_persistence_dropout_zeroes_only_that_slice_independently_of_cond_dropout():
    """persistence_dropout_prob=1.0 with cond_dropout_prob=0 should zero only
    the persistence slice (column 3), leaving the rest of z untouched -- and
    must not require cond_dropout_prob to be set at all (independent
    mechanism, see transforms.wind_persistence_slice)."""
    torch.manual_seed(0)
    model, schedule = _tiny_model_and_schedule()
    x = torch.randn(6, 1, 16)
    z = torch.ones(6, 4)
    loader = DataLoader(TensorDataset(x, z), batch_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    seen_z = []
    original_forward = model.forward

    def spy_forward(x_in, t_in, z_in):
        seen_z.append(z_in.clone())
        return original_forward(x_in, t_in, z_in)

    model.forward = spy_forward

    run_epoch(
        model, loader, schedule, optimizer, loss_fn, torch.device("cpu"), train=True,
        persistence_dropout_prob=1.0, persistence_slice=slice(3, 4),
    )

    assert len(seen_z) == 2
    for z_batch in seen_z:
        assert torch.all(z_batch[:, 0:3] == 1.0)
        assert torch.all(z_batch[:, 3] == 0.0)


def test_run_epoch_persistence_dropout_never_applied_during_validation():
    torch.manual_seed(0)
    model, schedule = _tiny_model_and_schedule()
    x = torch.randn(6, 1, 16)
    z = torch.ones(6, 4)
    loader = DataLoader(TensorDataset(x, z), batch_size=3)
    loss_fn = torch.nn.MSELoss()

    seen_z = []
    original_forward = model.forward

    def spy_forward(x_in, t_in, z_in):
        seen_z.append(z_in.clone())
        return original_forward(x_in, t_in, z_in)

    model.forward = spy_forward

    run_epoch(
        model, loader, schedule, None, loss_fn, torch.device("cpu"), train=False,
        persistence_dropout_prob=1.0, persistence_slice=slice(3, 4),
    )

    assert len(seen_z) == 2
    for z_batch in seen_z:
        assert torch.all(z_batch == 1.0)


def test_run_epoch_default_cond_dropout_prob_leaves_condition_unchanged():
    """Default cond_dropout_prob=0.0 must be a no-op -- backward compatible
    with every checkpoint trained before this feature existed."""
    torch.manual_seed(0)
    model, schedule = _tiny_model_and_schedule()
    x = torch.randn(6, 1, 16)
    z = torch.ones(6, 4)
    loader = DataLoader(TensorDataset(x, z), batch_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    seen_z = []
    original_forward = model.forward

    def spy_forward(x_in, t_in, z_in):
        seen_z.append(z_in.clone())
        return original_forward(x_in, t_in, z_in)

    model.forward = spy_forward

    run_epoch(model, loader, schedule, optimizer, loss_fn, torch.device("cpu"), train=True)

    assert len(seen_z) == 2
    for z_batch in seen_z:
        assert torch.all(z_batch == 1.0)


# ---------------------------------------------------------------------------
# Joint radius consistency loss (2ch U/V fix attempt)
# ---------------------------------------------------------------------------


def test_joint_radius_consistency_loss_near_zero_when_pred_noise_is_exact():
    """If pred_noise exactly matches the noise q_sample actually added,
    x0_pred reconstructs the real x0 exactly (up to clamping, which is a
    no-op here since x0 is kept well within clip_std) -- so the radius
    penalty should be ~0."""
    torch.manual_seed(0)
    schedule = DiffusionSchedule.build(T=50, beta_start=1e-4, beta_end=0.02, beta_schedule="cosine", device="cpu")
    x0 = torch.randn(4, 3, 8) * 0.3  # small magnitude, stays well inside +-3.5 clip
    t = torch.randint(0, 50, (4,))
    x_t, noise = q_sample(schedule, x0, t)

    loss = _joint_radius_consistency_loss(x0, x_t, noise, t, schedule, channels=(0, 1), clip_std=3.5)
    assert loss.item() < 1e-4


def test_joint_radius_consistency_loss_positive_when_pred_noise_is_wrong():
    torch.manual_seed(0)
    schedule = DiffusionSchedule.build(T=50, beta_start=1e-4, beta_end=0.02, beta_schedule="cosine", device="cpu")
    x0 = torch.randn(4, 3, 8) * 0.3
    t = torch.randint(0, 50, (4,))
    x_t, noise = q_sample(schedule, x0, t)
    wrong_noise = noise + 2.0

    loss_correct = _joint_radius_consistency_loss(x0, x_t, noise, t, schedule, (0, 1), 3.5)
    loss_wrong = _joint_radius_consistency_loss(x0, x_t, wrong_noise, t, schedule, (0, 1), 3.5)
    assert loss_wrong.item() > loss_correct.item()


def test_joint_radius_consistency_loss_ignores_channels_outside_the_pair():
    """Perturbing a channel NOT in `channels` must not change the loss --
    it should only look at the two named channels' joint radius."""
    torch.manual_seed(0)
    schedule = DiffusionSchedule.build(T=50, beta_start=1e-4, beta_end=0.02, beta_schedule="cosine", device="cpu")
    x0 = torch.randn(4, 3, 8) * 0.3
    t = torch.randint(0, 50, (4,))
    x_t, noise = q_sample(schedule, x0, t)

    noise_perturbed_other_channel = noise.clone()
    noise_perturbed_other_channel[:, 2, :] += 5.0  # channel 2 is not in (0, 1)

    loss_base = _joint_radius_consistency_loss(x0, x_t, noise, t, schedule, (0, 1), 3.5)
    loss_perturbed = _joint_radius_consistency_loss(x0, x_t, noise_perturbed_other_channel, t, schedule, (0, 1), 3.5)
    assert loss_base.item() == pytest.approx(loss_perturbed.item(), abs=1e-6)


def test_run_epoch_applies_joint_consistency_loss_when_weight_positive(monkeypatch):
    model, schedule = _tiny_model_and_schedule(n_channels=2)
    x = torch.randn(6, 2, 16)
    z = torch.ones(6, 4)
    loader = DataLoader(TensorDataset(x, z), batch_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    call_count = {"n": 0}
    original = train_module._joint_radius_consistency_loss

    def spy(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(train_module, "_joint_radius_consistency_loss", spy)

    run_epoch(
        model, loader, schedule, optimizer, loss_fn, torch.device("cpu"), train=True,
        joint_consistency_channels=(0, 1), joint_consistency_weight=0.5,
    )

    assert call_count["n"] == 2  # two batches of size 3


def test_run_epoch_skips_joint_consistency_loss_when_weight_zero(monkeypatch):
    model, schedule = _tiny_model_and_schedule(n_channels=2)
    x = torch.randn(6, 2, 16)
    z = torch.ones(6, 4)
    loader = DataLoader(TensorDataset(x, z), batch_size=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    call_count = {"n": 0}
    monkeypatch.setattr(
        train_module, "_joint_radius_consistency_loss",
        lambda *a, **kw: call_count.__setitem__("n", call_count["n"] + 1),
    )

    run_epoch(
        model, loader, schedule, optimizer, loss_fn, torch.device("cpu"), train=True,
        joint_consistency_channels=(0, 1), joint_consistency_weight=0.0,
    )

    assert call_count["n"] == 0


def test_run_epoch_skips_joint_consistency_loss_during_validation(monkeypatch):
    """val_loss must stay a pure noise-prediction MSE regardless of
    joint_consistency_weight, so it remains comparable across every run."""
    model, schedule = _tiny_model_and_schedule(n_channels=2)
    x = torch.randn(6, 2, 16)
    z = torch.ones(6, 4)
    loader = DataLoader(TensorDataset(x, z), batch_size=3)
    loss_fn = torch.nn.MSELoss()

    call_count = {"n": 0}
    monkeypatch.setattr(
        train_module, "_joint_radius_consistency_loss",
        lambda *a, **kw: call_count.__setitem__("n", call_count["n"] + 1),
    )

    run_epoch(
        model, loader, schedule, None, loss_fn, torch.device("cpu"), train=False,
        joint_consistency_channels=(0, 1), joint_consistency_weight=0.5,
    )

    assert call_count["n"] == 0


def test_resolve_device_forced_cpu_regardless_of_cuda_availability():
    """Cross-architecture requirement: explicitly requesting CPU must always
    yield CPU, even on a machine that also has a GPU -- this is the case
    `resolve_device("auto")` alone can't cover, since on a CUDA-having dev
    machine "auto" would resolve to cuda and this exact code path would
    never get exercised by the other (auto-only) tests in this suite.
    """
    device = resolve_device("cpu")
    assert device.type == "cpu"


def test_resolve_device_auto_never_raises_regardless_of_cuda_availability():
    """Whatever hardware this happens to run on, "auto" must resolve to a
    valid torch.device without raising -- the actual cross-arch contract.
    """
    device = resolve_device("auto")
    assert device.type in ("cpu", "cuda")
