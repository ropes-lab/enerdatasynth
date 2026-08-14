"""Tests for enerdatasynth.resources.wind.evaluation.evaluate: load_checkpoint_and_model's preset
resolution, in particular the `preset=` override added to support legacy
checkpoints saved by the original monolithic scripts.

Those legacy checkpoints have no `"preset"` key at all in their saved
`checkpoint["config"]` dict (only package-produced checkpoints save one),
so `load_checkpoint_and_model` used to silently default to `"2ch"` via
`config_dict.get("preset", "2ch")` -- wrong for e.g. a real 1-channel
checkpoint, and it blows up as a state_dict shape mismatch at
`load_state_dict` rather than failing cleanly.
"""

from __future__ import annotations

import pytest
import torch

from enerdatasynth.resources.wind.config import PresetName, get_config
from enerdatasynth.resources.wind.evaluation.evaluate import load_checkpoint_and_model
from enerdatasynth.resources.wind.training.train import build_model

SEQ_LEN = 16
HIDDEN_DIM = 8
COND_DIM = 5


def _build_legacy_style_checkpoint(tmp_path, preset: PresetName, name: str = "legacy_checkpoint.pt") -> str:
    """Build+save a real checkpoint for `preset`, then strip the `"preset"`
    key out of its saved config dict -- mimicking a checkpoint produced by
    the original monolithic scripts, whose CONFIG dict never had a
    `"preset"` field.
    """
    config = get_config(
        preset,
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
    model = build_model(config, config.n_model_channels, cond_dim=COND_DIM, seq_len=config.seq_len)

    config_dict = config.to_dict()
    config_dict.pop("preset", None)  # simulate the legacy (pre-package) CONFIG shape

    ckpt_path = tmp_path / name
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config_dict,
            "epoch": 31,
            "val_loss": 0.0560,
        },
        ckpt_path,
    )
    return str(ckpt_path)


def test_preset_override_loads_correct_channel_count_for_legacy_checkpoint(tmp_path):
    """A legacy-style 1ch checkpoint (no 'preset' key) loads correctly when
    preset='1ch' is passed explicitly."""
    ckpt_path = _build_legacy_style_checkpoint(tmp_path, "1ch")

    model, config, checkpoint = load_checkpoint_and_model(ckpt_path, device="cpu", preset="1ch")

    assert config.preset == "1ch"
    assert config.n_model_channels == 1
    assert "preset" not in checkpoint["config"]  # confirms this really is legacy-shaped

    # Model is actually usable: a forward pass with the expected channel count works.
    x = torch.randn(2, 1, SEQ_LEN)
    t = torch.zeros(2, dtype=torch.long)
    z = torch.randn(2, COND_DIM)
    out = model(x, t, z)
    assert out.shape == (2, 1, SEQ_LEN)


def test_preset_override_loads_correct_channel_count_for_3ch_legacy_checkpoint(tmp_path):
    """Same as above but for a 3ch legacy checkpoint, to also exercise a
    preset different from the old '2ch' default."""
    ckpt_path = _build_legacy_style_checkpoint(tmp_path, "3ch")

    model, config, checkpoint = load_checkpoint_and_model(ckpt_path, device="cpu", preset="3ch")

    assert config.preset == "3ch"
    assert config.n_model_channels == 3


def test_no_override_still_falls_back_to_old_2ch_default(tmp_path):
    """Without an explicit preset= override, behavior is unchanged from
    before this fix: a legacy checkpoint with no 'preset' key still resolves
    to the old '2ch' default. For a real non-2ch legacy checkpoint (e.g. the
    1ch one this bug was found on) that default is wrong and manifests as a
    state_dict shape mismatch -- this test locks in that this is still what
    happens absent an explicit override (i.e. default behavior truly didn't
    change), not that it's desirable.
    """
    ckpt_path = _build_legacy_style_checkpoint(tmp_path, "1ch")

    with pytest.raises(RuntimeError, match=r"(?i)size mismatch|error\(s\) in loading state_dict"):
        load_checkpoint_and_model(ckpt_path, device="cpu")


def test_no_override_matches_default_get_when_checkpoint_really_is_2ch(tmp_path):
    """Sanity check the other direction: when the legacy checkpoint's
    architecture actually *is* 2ch, the old default (no override) loads it
    just fine -- the bug only bites when the true preset differs from 2ch.
    """
    ckpt_path = _build_legacy_style_checkpoint(tmp_path, "2ch")

    model, config, checkpoint = load_checkpoint_and_model(ckpt_path, device="cpu")

    assert config.preset == "2ch"
    assert config.n_model_channels == 2
