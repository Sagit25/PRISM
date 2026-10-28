from __future__ import annotations

import pytest
import torch

from refractive_mam2.checkpoint_composition import compose_checkpoint


def _checkpoint(*, mam2_value: float, adapter_value: float | None) -> dict:
    state = {"matter.weight": torch.tensor([3.0])}
    if adapter_value is not None:
        state["background_model.diffusion_completion.pixel_refiner.weight"] = (
            torch.tensor([adapter_value])
        )
    return {
        "format_version": 8,
        "mam2_architecture": "same-mam2",
        "prism_architecture": "same-prism",
        "mam2_integration_config": {"matte": {"backend": "same-backend"}},
        "predictor_mam2": {"mask.weight": torch.tensor([mam2_value])},
        "physics_pipeline": state,
        "pipeline_config": {"background": {"completion_variant": "diffusion" if adapter_value is not None else "base"}},
    }


def test_compose_preserves_optics_while_transferring_trained_mam2(tmp_path) -> None:
    base = tmp_path / "base.pt"
    donor = tmp_path / "donor.pt"
    output = tmp_path / "out.pt"
    torch.save(_checkpoint(mam2_value=1, adapter_value=None), base)
    torch.save(_checkpoint(mam2_value=2, adapter_value=5), donor)
    compose_checkpoint(base, output, mam2_donor_path=donor)
    merged = torch.load(output, weights_only=True)
    assert merged["predictor_mam2"]["mask.weight"].item() == 2
    assert merged["physics_pipeline"]["matter.weight"].item() == 3
    assert merged["pipeline_config"]["background"]["completion_variant"] == "base"


def test_compose_transfers_diffusion_adapter_without_replacing_optics(tmp_path) -> None:
    base = tmp_path / "base.pt"
    donor = tmp_path / "donor.pt"
    output = tmp_path / "out.pt"
    torch.save(_checkpoint(mam2_value=1, adapter_value=None), base)
    torch.save(_checkpoint(mam2_value=2, adapter_value=5), donor)
    compose_checkpoint(base, output, background_donor_path=donor)
    merged = torch.load(output, weights_only=True)
    assert merged["predictor_mam2"]["mask.weight"].item() == 1
    assert merged["physics_pipeline"]["matter.weight"].item() == 3
    assert merged["physics_pipeline"]["background_model.diffusion_completion.pixel_refiner.weight"].item() == 5
    assert merged["pipeline_config"]["background"]["completion_variant"] == "diffusion"


def test_background_donor_can_precede_new_foreground_refiner(tmp_path) -> None:
    base = _checkpoint(mam2_value=1, adapter_value=None)
    base["prism_architecture"] += "-foreground-residual-v1"
    base["pipeline_config"]["matter"] = {
        "refine_straight_foreground": True, "flow_refinement_width": 32,
    }
    base["physics_pipeline"]["matter.foreground_refiner.weight"] = torch.tensor([7.0])
    donor = _checkpoint(mam2_value=2, adapter_value=5)
    donor["pipeline_config"]["matter"] = {
        "refine_straight_foreground": False, "flow_refinement_width": 32,
    }
    base_path, donor_path, output = (
        tmp_path / "base.pt", tmp_path / "donor.pt", tmp_path / "out.pt"
    )
    torch.save(base, base_path)
    torch.save(donor, donor_path)
    compose_checkpoint(base_path, output, background_donor_path=donor_path)
    merged = torch.load(output, weights_only=True)
    assert merged["physics_pipeline"]["matter.foreground_refiner.weight"].item() == 7
    assert merged["physics_pipeline"][
        "background_model.diffusion_completion.pixel_refiner.weight"
    ].item() == 5

    donor["pipeline_config"]["matter"]["flow_refinement_width"] = 64
    torch.save(donor, donor_path)
    with pytest.raises(RuntimeError, match="prism_architecture"):
        compose_checkpoint(base_path, output, background_donor_path=donor_path)


def test_background_donor_can_precede_new_mam2_flow_bridge(tmp_path) -> None:
    base = _checkpoint(mam2_value=1, adapter_value=None)
    base["prism_architecture"] += (
        "-flow-guided-latent-bridge-v1-pretrained-mam2-flow-bridge-v1"
    )
    base["pipeline_config"]["matter"] = {
        "flow_guided_latent_bridge": True,
        "use_mam2_foreground_prior": True,
        "latent_bridge_width": 32,
    }
    base["physics_pipeline"]["matter.latent_bridge.weight"] = torch.tensor([7.0])
    donor = _checkpoint(mam2_value=2, adapter_value=5)
    donor["pipeline_config"]["matter"] = {
        "flow_guided_latent_bridge": False,
        "use_mam2_foreground_prior": False,
        "latent_bridge_width": 32,
    }
    base_path, donor_path, output = (
        tmp_path / "base.pt", tmp_path / "donor.pt", tmp_path / "out.pt"
    )
    torch.save(base, base_path)
    torch.save(donor, donor_path)
    compose_checkpoint(base_path, output, background_donor_path=donor_path)
    merged = torch.load(output, weights_only=True)
    assert merged["physics_pipeline"]["matter.latent_bridge.weight"].item() == 7
    assert merged["physics_pipeline"][
        "background_model.diffusion_completion.pixel_refiner.weight"
    ].item() == 5

    donor["pipeline_config"]["matter"]["latent_bridge_width"] = 64
    torch.save(donor, donor_path)
    with pytest.raises(RuntimeError, match="prism_architecture"):
        compose_checkpoint(base_path, output, background_donor_path=donor_path)


def test_compose_rejects_incompatible_architecture(tmp_path) -> None:
    base = tmp_path / "base.pt"
    donor = tmp_path / "donor.pt"
    output = tmp_path / "out.pt"
    torch.save(_checkpoint(mam2_value=1, adapter_value=None), base)
    incompatible = _checkpoint(mam2_value=2, adapter_value=5)
    incompatible["prism_architecture"] = "different"
    torch.save(incompatible, donor)
    with pytest.raises(RuntimeError, match="prism_architecture"):
        compose_checkpoint(base, output, mam2_donor_path=donor)
