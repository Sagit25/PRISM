"""Build or submit the isolated SAM2/MEMatte–flow–FLUX experiment.

Unlike the FBA ablation, this variant uses the already-pretrained SAM2/MEMatte
visual representation as its foreground-side prior. A small trainable bridge
samples the completed FLUX background at the predicted refractive coordinate.
The original foreground and background foundation models stay frozen except
for the existing Stage-4 MAM2 adapters/decoder schedule.
"""

from __future__ import annotations

import argparse

import yaml

from submit_prism_full_64_optical_priority_v7r7 import (
    RUN_NAME as PREVIOUS_RUN_NAME,
    build_spec as previous_build_spec,
    verify_donor_with_available_sdk,
)
from submit_prism_full_64_exclusion1_v2 import PACK_RUN_ID


RUN_NAME = "prism-full-64-sota-flow-coupled-v10-v1"


def build_spec(commit: str) -> dict:
    spec = previous_build_spec(commit)
    command = spec["run"][0]["command"]
    if command.count(PREVIOUS_RUN_NAME) < 4:
        raise RuntimeError("Expected isolated output, checkpoint and W&B paths")
    command = command.replace(PREVIOUS_RUN_NAME, RUN_NAME)
    launcher = "network/scripts/train_prism_all_stages.sh"
    if command.count(launcher) != 1:
        raise RuntimeError("Expected exactly one all-stage launcher")
    settings = "\n".join((
        "export PRISM_FOREGROUND_REFINEMENT=true",
        "export PRISM_FLOW_GUIDED_LATENT_BRIDGE=true",
        "export PRISM_USE_MAM2_FOREGROUND_PRIOR=true",
        "export PRISM_USE_FBA_FOREGROUND_PRIOR=false",
        "export PRISM_DIFFUSION_MIN_WORKING_SIDE=128",
    ))
    command = command.replace(launcher, settings + "\n" + launcher, 1)
    spec["name"] = RUN_NAME
    spec["run"][0]["command"] = command
    spec["description"] += (
        " Separate v10 SAM2/MEMatte foreground-feature and FLUX background"
        " fusion through predicted refractive flow. No FBA weights or code"
        " are downloaded. Stage 1 is restored; Stage 2--4 use the optical"
        " priority schedule and isolated checkpoints/W&B group."
    )
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    spec = build_spec(args.git_commit)
    print(f"PRISM_SOTA_FLOW_COUPLED_READY name={RUN_NAME} commit={args.git_commit}", flush=True)
    if not args.submit:
        return

    from vessl import vessl_api
    from vessl.run import list_run_logs, list_runs, read_run

    pack = read_run(PACK_RUN_ID)
    if pack.status != "completed":
        raise RuntimeError(f"Pack run is not completed: {pack.status}")
    messages = [str(getattr(item, "message", item)) for item in list_run_logs(PACK_RUN_ID, tail=200)]
    if not any("PRISM_FULL_64_PACK_VERIFIED" in message for message in messages):
        raise RuntimeError("Pack verification marker is missing")
    verify_donor_with_available_sdk()
    existing = [
        run for run in list_runs(
            organization_name="snu-eng-dgx-heavy", project_name="PRISM"
        ) if run.run_spec.title == RUN_NAME
    ]
    if existing:
        print(f"PRISM_SOTA_FLOW_COUPLED_ALREADY_EXISTS ids={[run.id for run in existing]}", flush=True)
        return
    response = vessl_api.run_spec_create_from_yamlapi(
        organization_name="snu-eng-dgx-heavy",
        project_name="PRISM",
        run_spec_create_from_yamlapi_input={"yaml_spec": yaml.safe_dump(spec, sort_keys=False)},
    )
    print(f"PRISM_SOTA_FLOW_COUPLED_SUBMITTED id={response.run_execution.id}", flush=True)


if __name__ == "__main__":
    main()
