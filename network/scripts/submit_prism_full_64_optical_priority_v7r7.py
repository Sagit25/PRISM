"""Submit an isolated optical-priority PRISM run, reusing completed Stage 1.

The shorter Stage 2--4 schedule provides an early full-pipeline result while
preserving best checkpoints and leaving all prior VESSL artifacts untouched.
"""

from __future__ import annotations

import argparse

import yaml

from submit_prism_full_64_exclusion1_v2 import PACK_RUN_ID
from submit_prism_full_64_riskfix_v7r4 import (
    RUN_NAME as PREVIOUS_RUN_NAME,
    build_spec as previous_build_spec,
    verify_stage1_donor,
)


RUN_NAME = "prism-full-64-optical-priority-v7r7-v1"


def build_spec(commit: str) -> dict:
    spec = previous_build_spec(commit)
    command = spec["run"][0]["command"]
    if command.count(PREVIOUS_RUN_NAME) < 4:
        raise RuntimeError("Expected isolated v7r4 output, checkpoint and W&B paths")
    command = command.replace(PREVIOUS_RUN_NAME, RUN_NAME)
    launcher = "network/scripts/train_prism_all_stages.sh"
    if command.count(launcher) != 1:
        raise RuntimeError("Expected exactly one all-stage launcher")
    command = command.replace(
        launcher,
        "export PRISM_LOSS_PROFILE=optical_priority\n"
        "export PRISM_STAGE2_EPOCHS=8\n"
        "export PRISM_STAGE3_EPOCHS=10\n"
        "export PRISM_STAGE4_EPOCHS=12\n"
        + launcher,
        1,
    )
    spec["name"] = RUN_NAME
    spec["run"][0]["command"] = command
    spec["description"] += (
        " Optical-priority objective: supervise RGB transmission/chroma,"
        " transmittance, foreground, residual and object recomposition more"
        " strongly while retaining a lower-weight direct pixel-flow loss."
        " Stage 2/3/4 run 8/10/12 epochs for faster feedback; best checkpoints"
        " and isolated W&B history remain available for a longer continuation."
    )
    return spec


def verify_donor_with_available_sdk() -> None:
    try:
        verify_stage1_donor()
        return
    except RuntimeError as error:
        if "VESSL SDK 0.1.199 or newer" not in str(error):
            raise

    # The local SDK can read runs but cannot list the newer storage volumes.
    # Stage 4 checkpoint uploads establish that the donor finished Stages 1A
    # and 1B; the cloud launcher still validates and restores their files.
    from vessl.run import list_run_logs, list_runs

    donor_name = "prism-full-64-temporal-transmission-flux-v7r2-maskgate"
    donors = [
        run for run in list_runs(
            organization_name="snu-eng-dgx-heavy", project_name="PRISM"
        ) if run.run_spec.title == donor_name
    ]
    if len(donors) != 1:
        raise RuntimeError(f"Expected exactly one Stage 1 donor, found {len(donors)}")
    messages = [
        str(getattr(item, "message", item))
        for item in list_run_logs(donors[0].id, tail=2000)
    ]
    if not any(
        "PRISM_CHECKPOINT_UPLOADED" in message
        and "/checkpoints/stage4/" in message
        for message in messages
    ):
        raise RuntimeError("Stage 1 donor has no evidence of subsequent Stage 4 training")
    print("PRISM_STAGE1_DONOR_PROGRESSION_VERIFIED; restore validates files", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    spec = build_spec(args.git_commit)
    print(
        f"PRISM_OPTICAL_PRIORITY_READY name={RUN_NAME} commit={args.git_commit} "
        "stage1=donor stages_to_train=2,3,4 epochs=8,10,12",
        flush=True,
    )
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
        for run in list_runs(
            organization_name="snu-eng-dgx-heavy", project_name="PRISM"
        )
        if run.run_spec.title == RUN_NAME
    ]
    if existing:
        print(
            f"PRISM_OPTICAL_PRIORITY_ALREADY_EXISTS ids={[run.id for run in existing]}",
            flush=True,
        )
        return

    response = vessl_api.run_spec_create_from_yamlapi(
        organization_name="snu-eng-dgx-heavy",
        project_name="PRISM",
        run_spec_create_from_yamlapi_input={
            "yaml_spec": yaml.safe_dump(spec, sort_keys=False)
        },
    )
    print(f"PRISM_OPTICAL_PRIORITY_SUBMITTED id={response.run_execution.id}", flush=True)


if __name__ == "__main__":
    main()
