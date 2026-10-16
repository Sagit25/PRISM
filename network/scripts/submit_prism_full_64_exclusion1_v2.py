"""Submit the verified 64px PRISM run with local data and a 1x1 hole mask.

This script is intentionally idempotent: a run with the same VESSL title is
reported, not submitted again. Pass a published 40-character Git commit.
"""

from __future__ import annotations

import argparse
import re

import yaml

from wait_and_launch_vessl_training import build_training_spec, parse_args


PACK_RUN_ID = 369367271785
ARCHIVE_VOLUME = "prism-full-64-singleflow-pack-v2-369367271785-output"
RUN_NAME = "prism-full-64-exclusion1-flowresidual-v2"


def build_spec(
    commit: str,
    archive_volume: str = ARCHIVE_VOLUME,
    run_name: str = RUN_NAME,
    *,
    flow_refinement: bool = False,
) -> dict:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("--git-commit must be a full lowercase 40-character SHA")
    if re.fullmatch(r"[a-z0-9][a-z0-9-]*", run_name) is None:
        raise ValueError("--run-name must be a safe lowercase run name")
    args = parse_args(
        [
            "--archive-volume",
            archive_volume,
            "--result-volume",
            archive_volume,
            "--git-commit",
            commit,
            "--run-name",
            run_name,
            "--output-subdir",
            run_name,
            "--restore-output-subdir",
            run_name,
            "--wandb-entity",
            "humangpt",
            "--materialize-all",
            "--background-exclusion-dilation",
            "1",
        ]
    )
    spec = build_training_spec(args)
    spec["description"] = (
        "Full 64x64 PRISM Stage 1A-4 on the verified seven-split archive: "
        "1x1 background exclusion, editable inverse-refractive background "
        "evidence with direct pixels preserved, direct masked flow EPE with "
        "GT-residual smoothness, and SHA-verified local materialization of all "
        "train/validation/test images. "
        + ("Photometric-residual flow refinement enabled. " if flow_refinement else "")
        + "Isolated output and checkpoints."
    )
    # The deployed VESSL YAML API expects this generic-secret form.
    spec["env"] = {
        "WANDB_API_KEY": {"secret": "WANDB_API_KEY"},
        "HF_TOKEN": {"secret": "HF_TOKEN"},
    }
    command = spec["run"][0]["command"]
    checkout = "git checkout --detach FETCH_HEAD\n"
    if command.count(checkout) != 1:
        raise RuntimeError("cannot locate the pinned Git checkout")
    command = command.replace(
        checkout,
        checkout
        + f'test "$(git rev-parse HEAD)" = "{commit}"\n'
        + f"echo PRISM_FULL_64_COMMIT_OK {commit}\n",
        1,
    )
    replacements = {
        "export PRISM_WANDB_GROUP=full-prism-v8": (
            f"export PRISM_WANDB_GROUP={run_name}"
        ),
        "export PRISM_RUNTIME_CONTRACT_CHECKS=false": (
            "export PRISM_RUNTIME_CONTRACT_CHECKS=true"
        ),
    }
    for old, new in replacements.items():
        if command.count(old) != 1:
            raise RuntimeError(f"cannot locate training setting: {old}")
        command = command.replace(old, new, 1)
    if flow_refinement:
        launcher = "network/scripts/train_prism_all_stages.sh"
        if command.count(launcher) != 1:
            raise RuntimeError("cannot locate the all-stage training launcher")
        command = command.replace(
            launcher,
            "export PRISM_FLOW_REFINEMENT=true\n" + launcher,
            1,
        )
    spec["run"][0]["command"] = command
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--archive-volume", default=ARCHIVE_VOLUME)
    parser.add_argument("--run-name", default=RUN_NAME)
    parser.add_argument("--flow-refinement", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    spec = build_spec(
        args.git_commit,
        args.archive_volume,
        args.run_name,
        flow_refinement=args.flow_refinement,
    )
    print(
        f"PRISM_FULL_64_READY name={args.run_name} commit={args.git_commit} "
        f"archive_volume={args.archive_volume} materialize_all=true "
        f"background_exclusion=1 flow_refinement={str(args.flow_refinement).lower()}",
        flush=True,
    )
    if not args.submit:
        return

    from vessl import vessl_api
    from vessl.run import list_run_logs, list_runs, read_run

    pack = read_run(PACK_RUN_ID)
    if pack.status != "completed":
        raise RuntimeError(f"pack run is not completed: {pack.status}")
    messages = [
        str(getattr(item, "message", item))
        for item in list_run_logs(PACK_RUN_ID, tail=200)
    ]
    if not any("PRISM_FULL_64_PACK_VERIFIED" in message for message in messages):
        raise RuntimeError("pack completion verification marker is missing")
    existing = [
        run
        for run in list_runs(
            organization_name="snu-eng-dgx-heavy", project_name="PRISM"
        )
        if run.run_spec.title == args.run_name
    ]
    if existing:
        print(
            f"PRISM_FULL_64_ALREADY_EXISTS ids={[run.id for run in existing]}",
            flush=True,
        )
        return

    from openapi_client import APIV1Api
    from openapi_client.exceptions import ApiException

    try:
        response = APIV1Api.run_spec_create_from_yamlapi(
            vessl_api,
            organization_name="snu-eng-dgx-heavy",
            project_name="PRISM",
            run_spec_create_from_yamlapi_input={
                "yaml_spec": yaml.safe_dump(spec, sort_keys=False)
            },
        )
    except ApiException as exc:
        raise RuntimeError(
            f"VESSL run submission rejected: HTTP {exc.status}; body={exc.body}"
        ) from exc
    print(f"PRISM_FULL_64_SUBMITTED id={response.run_execution.id}", flush=True)


if __name__ == "__main__":
    main()
