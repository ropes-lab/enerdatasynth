"""
Tests for enerdatasynth.generators: the BaseGenerator interface and
LoadDiffusionModel, including a real (if tiny) end-to-end round trip
through train -> checkpoint -> LoadDiffusionModel.from_run_id -> sample.
"""

from __future__ import annotations
import numpy as np
import pytest

from enerdatasynth.generators.base import BaseGenerator
from enerdatasynth.generators import LoadDiffusionModel


def test_base_generator_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseGenerator()


def test_incomplete_subclass_rejected():
    class Incomplete(BaseGenerator):
        domain = "incomplete"
        # doesn't implement sample()

    with pytest.raises(TypeError):
        Incomplete()


def test_complete_subclass_can_be_instantiated():
    class Trivial(BaseGenerator):
        domain = "trivial"

        def sample(self, z, seed=None, **kwargs):
            return np.zeros((len(z), 1))

    instance = Trivial()
    assert instance.domain == "trivial"
    assert instance.sample(np.zeros((3, 2))).shape == (3, 1)


def test_load_diffusion_model_is_a_base_generator():
    assert issubclass(LoadDiffusionModel, BaseGenerator)
    assert LoadDiffusionModel.domain == "load"


def test_load_diffusion_model_from_run_id_and_sample(tiny_trained_checkpoint, synthetic_profiles):
    result = tiny_trained_checkpoint
    _, Z = synthetic_profiles

    model = LoadDiffusionModel.from_run_id(result["run_id"])
    assert model.x_dim == result["x_dim"]
    assert model.cond_dim == result["cond_dim"]

    samples = model.sample(Z[:4], seed=0)
    assert samples.shape == (4, result["x_dim"])
    assert np.isfinite(samples).all()


def test_load_diffusion_model_sample_reproducible_with_seed(tiny_trained_checkpoint, synthetic_profiles):
    result = tiny_trained_checkpoint
    _, Z = synthetic_profiles
    model = LoadDiffusionModel.from_run_id(result["run_id"])

    a = model.sample(Z[:3], seed=42)
    b = model.sample(Z[:3], seed=42)
    assert np.array_equal(a, b)


def test_load_diffusion_model_from_run_id_missing_raises():
    with pytest.raises(FileNotFoundError):
        LoadDiffusionModel.from_run_id("definitely_not_a_real_run_id")


def test_from_run_id_defaults_missing_architecture_to_mlp_not_current_default(tmp_path, monkeypatch):
    """
    Regression test: checkpoints trained before the `architecture`
    field existed in result.json (the original MLP sweep) must load
    as "mlp", not silently default to whatever DEFAULT_ARCHITECTURE
    ("transformer") happens to be today -- that would try to load MLP
    weights into a transformer and fail.
    """
    import json
    import torch
    from enerdatasynth.resources.pjm_load.models.denoiser import SimpleMLPDenoiser

    run_id = "legacy_no_architecture_field"
    ckpt_dir = tmp_path / run_id
    ckpt_dir.mkdir()

    model = SimpleMLPDenoiser(x_dim=24, cond_dim=23, hidden_dim=8, n_layers=1)
    torch.save(model.state_dict(), ckpt_dir / "best.pt")
    # Deliberately omit "architecture", "x_dim", "cond_dim", "n_heads" --
    # exactly what the pre-transformer result.json files look like.
    (ckpt_dir / "result.json").write_text(json.dumps({
        "run_id": run_id, "diffusion_steps": 100, "hidden_dim": 8, "n_layers": 1,
    }))

    loaded = LoadDiffusionModel.from_run_id(run_id, checkpoint_root=tmp_path)
    assert loaded.architecture == "mlp"
