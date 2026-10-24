"""Submit the isolated flow-guided FBA foreground experiment to VESSL.

The completed v7r2 Stage 1 checkpoints are copied into a fresh run. Stages
2--4 train the new PAM/bridge while the official FBA encoder and decoder stay
frozen. Official noncommercial FBA assets are fetched and SHA-checked inside
the run; they are never committed to this repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from submit_prism_full_64_optical_priority_v7r7 import (
    RUN_NAME as PREVIOUS_RUN_NAME,
    build_spec as previous_build_spec,
    verify_donor_with_available_sdk,
)
from submit_prism_full_64_exclusion1_v2 import PACK_RUN_ID


RUN_NAME = "prism-full-64-flowguided-fba-v9-v1"


def build_spec(commit: str) -> dict:
    spec = previous_build_spec(commit)
    command = spec["run"][0]["command"]
    if command.count(PREVIOUS_RUN_NAME) < 4:
        raise RuntimeError("Expected isolated output, checkpoint and W&B paths")
    command = command.replace(PREVIOUS_RUN_NAME, RUN_NAME)

    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "third_party" / "FBA_ASSETS.json").read_text()
    )
    source_commit = manifest["source_commit"]
    checkpoint_sha256 = manifest["checkpoint_sha256"]
    checkpoint_url = manifest["checkpoint_source"]
    source_repository = manifest["source_repository"]
    if len(source_commit) != 40 or len(checkpoint_sha256) != 64:
        raise RuntimeError("FBA asset manifest requires complete hashes")
    asset_setup = f"""
python -m pip install --quiet gdown
git clone --no-checkout {source_repository} /root/workspace/FBA_Matting
git -C /root/workspace/FBA_Matting fetch --depth 1 origin {source_commit}
git -C /root/workspace/FBA_Matting checkout --detach {source_commit}
test "$(git -C /root/workspace/FBA_Matting rev-parse HEAD)" = "{source_commit}"
mkdir -p /root/workspace/fba-assets
gdown --fuzzy '{checkpoint_url}' -O /root/workspace/fba-assets/FBA.pth
printf '%s  %s\\n' {checkpoint_sha256} /root/workspace/fba-assets/FBA.pth | sha256sum -c -
echo PRISM_FBA_OFFICIAL_ASSETS_VERIFIED {source_commit} {checkpoint_sha256}
""".strip()
    anchor = "PYTHON_BIN=python network/scripts/install_official_sam2.sh"
    if command.count(anchor) != 1:
        raise RuntimeError("Expected exactly one official SAM2 setup command")
    command = command.replace(anchor, asset_setup + "\n" + anchor, 1)

    launcher = "network/scripts/train_prism_all_stages.sh"
    if command.count(launcher) != 1:
        raise RuntimeError("Expected exactly one all-stage launcher")
    settings = "\n".join(
        (
            "export PRISM_FOREGROUND_REFINEMENT=true",
            "export PRISM_FLOW_GUIDED_LATENT_BRIDGE=true",
            "export PRISM_USE_FBA_FOREGROUND_PRIOR=true",
            "export PRISM_FBA_SOURCE_ROOT=/root/workspace/FBA_Matting",
            "export PRISM_FBA_CHECKPOINT=/root/workspace/fba-assets/FBA.pth",
            "export PRISM_FBA_INPUT_SIDE=128",
            "export PRISM_FBA_TRAIN_DECODER=false",
            "export PRISM_DIFFUSION_MIN_WORKING_SIDE=128",
        )
    )
    command = command.replace(launcher, settings + "\n" + launcher, 1)
    spec["name"] = RUN_NAME
    spec["run"][0]["command"] = command
    spec["description"] += (
        " Separate v9 flow-guided latent bridge with frozen official FBA"
        " pretrained features, foreground refinement, direct F/T/flow losses,"
        " FLUX boundary and inverse-evidence contamination fixes, and per-head"
        " gradient diagnostics. The v7r7 and SGS code/checkpoints are preserved."
    )
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    spec = build_spec(args.git_commit)
    print(f"PRISM_FLOWGUIDED_FBA_READY name={RUN_NAME} commit={args.git_commit}", flush=True)
    if not args.submit:
        return

    from vessl import vessl_api
    from vessl.run import list_run_logs, list_runs, read_run

    pack = read_run(PACK_RUN_ID)
    if pack.status != "completed":
        raise RuntimeError(f"Pack run is not completed: {pack.status}")
    messages = [
        str(getattr(item, "message", item))
        for item in list_run_logs(PACK_RUN_ID, tail=200)
    ]
    if not any("PRISM_FULL_64_PACK_VERIFIED" in message for message in messages):
        raise RuntimeError("Pack verification marker is missing")
    verify_donor_with_available_sdk()
    existing = [
        run
        for run in list_runs(organization_name="snu-eng-dgx-heavy", project_name="PRISM")
        if run.run_spec.title == RUN_NAME
    ]
    if existing:
        print(f"PRISM_FLOWGUIDED_FBA_ALREADY_EXISTS ids={[run.id for run in existing]}", flush=True)
        return

    response = vessl_api.run_spec_create_from_yamlapi(
        organization_name="snu-eng-dgx-heavy",
        project_name="PRISM",
        run_spec_create_from_yamlapi_input={
            "yaml_spec": yaml.safe_dump(spec, sort_keys=False)
        },
    )
    print(f"PRISM_FLOWGUIDED_FBA_SUBMITTED id={response.run_execution.id}", flush=True)


if __name__ == "__main__":
    main()
