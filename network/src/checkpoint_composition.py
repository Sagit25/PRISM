"""Compose compatible PRISM warm starts without discarding trained components."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch


def _load(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 8:
        raise RuntimeError(f"incompatible PRISM checkpoint format: {path}")
    return payload


def _compatible(
    base: dict[str, Any], donor: dict[str, Any], *, background_only: bool,
) -> None:
    if base.get("mam2_architecture") != donor.get("mam2_architecture"):
        raise RuntimeError("checkpoint mam2_architecture differs between warm-start sources")
    if base.get("prism_architecture") != donor.get("prism_architecture"):
        base_matter = base.get("pipeline_config", {}).get("matter", {})
        donor_matter = donor.get("pipeline_config", {}).get("matter", {})
        adding_foreground_refiner = (
            background_only
            and base.get("prism_architecture")
            == f"{donor.get('prism_architecture')}-foreground-residual-v1"
            and base_matter.get("refine_straight_foreground") is True
            and donor_matter.get("refine_straight_foreground") is False
            and all(
                base_matter.get(key) == donor_matter.get(key)
                for key in set(base_matter) | set(donor_matter)
                if key != "refine_straight_foreground"
            )
        )
        adding_mam2_flow_bridge = (
            background_only
            and base.get("prism_architecture")
            == (
                f"{donor.get('prism_architecture')}"
                "-flow-guided-latent-bridge-v1"
                "-pretrained-mam2-flow-bridge-v1"
            )
            and base_matter.get("flow_guided_latent_bridge") is True
            and base_matter.get("use_mam2_foreground_prior") is True
            and donor_matter.get("flow_guided_latent_bridge") is False
            and donor_matter.get("use_mam2_foreground_prior") is False
            and all(
                base_matter.get(key) == donor_matter.get(key)
                for key in set(base_matter) | set(donor_matter)
                if key not in {"flow_guided_latent_bridge", "use_mam2_foreground_prior"}
            )
        )
        if not adding_foreground_refiner and not adding_mam2_flow_bridge:
            raise RuntimeError("checkpoint prism_architecture differs between warm-start sources")
    base_backend = base.get("mam2_integration_config", {}).get("matte", {}).get("backend")
    donor_backend = donor.get("mam2_integration_config", {}).get("matte", {}).get("backend")
    if base_backend != donor_backend:
        raise RuntimeError("checkpoint matting backends differ")


def compose_checkpoint(
    base_path: str | Path,
    output_path: str | Path,
    *,
    mam2_donor_path: str | Path | None = None,
    background_donor_path: str | Path | None = None,
) -> None:
    if (mam2_donor_path is None) == (background_donor_path is None):
        raise ValueError("select exactly one compatible warm-start donor")
    base = _load(base_path)
    donor_path = mam2_donor_path or background_donor_path
    assert donor_path is not None
    donor = _load(donor_path)
    _compatible(base, donor, background_only=background_donor_path is not None)

    if mam2_donor_path is not None:
        base_state = base["predictor_mam2"]
        donor_state = donor["predictor_mam2"]
        if base_state.keys() != donor_state.keys():
            raise RuntimeError("MAM2 checkpoint tensor keys differ")
        for key in base_state:
            if base_state[key].shape != donor_state[key].shape:
                raise RuntimeError(f"MAM2 checkpoint tensor shape differs: {key}")
        base["predictor_mam2"] = donor_state
        print(f"PRISM_MAM2_WARMSTART_COMPOSED donor={donor_path}", flush=True)
    else:
        prefix = "background_model.diffusion_completion."
        base_state = base["physics_pipeline"]
        donor_state = donor["physics_pipeline"]
        selected = {key: value for key, value in donor_state.items() if key.startswith(prefix)}
        if not selected:
            raise RuntimeError("background donor has no diffusion adapter tensors")
        for key, value in selected.items():
            if key in base_state and base_state[key].shape != value.shape:
                raise RuntimeError(f"background adapter tensor shape differs: {key}")
        base_state.update(selected)
        base["pipeline_config"]["background"] = donor["pipeline_config"]["background"]
        print(
            f"PRISM_BACKGROUND_WARMSTART_COMPOSED donor={donor_path} "
            f"tensors={len(selected)}",
            flush=True,
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    torch.save(base, temporary)
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    donor = parser.add_mutually_exclusive_group(required=True)
    donor.add_argument("--mam2-donor")
    donor.add_argument("--background-donor")
    args = parser.parse_args()
    compose_checkpoint(
        args.base,
        args.output,
        mam2_donor_path=args.mam2_donor,
        background_donor_path=args.background_donor,
    )


if __name__ == "__main__":
    main()
