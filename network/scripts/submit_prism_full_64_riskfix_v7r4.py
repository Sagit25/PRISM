"""Submit the v7r4 risk fixes, reusing only completed v7r2 Stage 1 MAM2.

The Stage 1B donor contains an untrained PAM with the old output convention.
Stage 2 explicitly transfers MAM2 only and initializes the new PAM afresh.
"""

from __future__ import annotations

import argparse
import re

import yaml

from submit_prism_full_64_exclusion1_v2 import ARCHIVE_VOLUME, PACK_RUN_ID
from submit_prism_full_64_straightf_v7r3 import (
    RUN_NAME as PREVIOUS_RUN_NAME,
    build_spec as previous_build_spec,
)


RUN_NAME = "prism-full-64-temporal-transmission-flux-v7r4-riskfix"
STAGE1_DONOR = "prism-full-64-temporal-transmission-flux-v7r2-maskgate"


def build_spec(commit: str) -> dict:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("A published full 40-character Git SHA is required")
    spec = previous_build_spec(commit)
    command = spec["run"][0]["command"]
    if command.count(PREVIOUS_RUN_NAME) < 5:
        raise RuntimeError("Expected isolated v7r3 output, checkpoint and W&B paths")
    command = command.replace(PREVIOUS_RUN_NAME, RUN_NAME)
    restore_uri = (
        f"volume://vessl-storage/{ARCHIVE_VOLUME}/{RUN_NAME}"
    )
    donor_uri = (
        f"volume://vessl-storage/{ARCHIVE_VOLUME}/{STAGE1_DONOR}"
    )
    replacements = {
        f"export PRISM_RESTORE_CHECKPOINT_URI={restore_uri}": (
            f"export PRISM_RESTORE_CHECKPOINT_URI={donor_uri}"
        ),
        "export PRISM_RESTORE_CHECKPOINT_STAGES=1a,1b,2,3,4": (
            "export PRISM_RESTORE_CHECKPOINT_STAGES=1a,1b"
        ),
        "network/scripts/train_prism_all_stages.sh": (
            "export PRISM_STAGE1_TRANSFER=true\n"
            "network/scripts/train_prism_all_stages.sh"
        ),
    }
    for old, new in replacements.items():
        if command.count(old) != 1:
            raise RuntimeError(f"Expected exactly one training setting: {old}")
        command = command.replace(old, new, 1)
    spec["name"] = RUN_NAME
    spec["run"][0]["command"] = command
    spec["description"] += (
        " Restore only completed Stage 1A/1B checkpoints from v7r2; skip both "
        "stages and transfer final Stage 1B MAM2 into fresh v7r4 PAM at Stage 2. "
        "Stage 2-4 use corrected inverse evidence, completion, oracle render "
        "supervision, and fixed-GT checkpoint selection."
    )
    return spec


def verify_stage1_donor() -> None:
    from sync_prism_checkpoints import build_store, join_key, select_remote_object

    store = build_store("vessl-storage", ARCHIVE_VOLUME)
    for stage in ("1a", "1b"):
        prefix = join_key(STAGE1_DONOR, "checkpoints", f"stage{stage}")
        objects = list(store.iter_prefix(prefix))
        best = select_remote_object(
            objects, join_key(prefix, f"prism_stage{stage}_best.pt")
        )
        marker = select_remote_object(objects, join_key(prefix, ".training_complete"))
        if best is None or best.size <= 0 or marker is None:
            raise RuntimeError(f"Stage {stage} final donor checkpoint is incomplete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    spec = build_spec(args.git_commit)
    print(
        f"PRISM_V7R4_READY name={RUN_NAME} commit={args.git_commit} "
        f"stage1_donor={STAGE1_DONOR} stages_to_train=2,3,4",
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
    verify_stage1_donor()
    existing = [
        run
        for run in list_runs(
            organization_name="snu-eng-dgx-heavy", project_name="PRISM"
        )
        if run.run_spec.title == RUN_NAME
    ]
    if existing:
        print(f"PRISM_V7R4_ALREADY_EXISTS ids={[run.id for run in existing]}", flush=True)
        return

    from vessl.openapi_client import APIV1Api

    response = APIV1Api.run_spec_create_from_yamlapi(
        vessl_api,
        organization_name="snu-eng-dgx-heavy",
        project_name="PRISM",
        run_spec_create_from_yamlapi_input={
            "yaml_spec": yaml.safe_dump(spec, sort_keys=False)
        },
    )
    print(f"PRISM_V7R4_SUBMITTED id={response.run_execution.id}", flush=True)


if __name__ == "__main__":
    main()
