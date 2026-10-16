"""Submit the v7 rerun with the SHA-verified official MEMatte AIM-500 weights."""

from __future__ import annotations

import argparse
import re

import yaml

from submit_prism_full_64_exclusion1_v2 import (
    ARCHIVE_VOLUME,
    PACK_RUN_ID,
    build_spec,
)


RUN_NAME = "prism-full-64-temporal-transmission-flux-v7r1-aim500"
CHECKPOINT_KEY = "models/MEMatte_AIM500.pth"
CHECKPOINT_SHA256 = "538cf871933dd0663a6bc28a4bec9b870cc2de909c7431b857b72ce427e82a09"


def rerun_spec(commit: str) -> dict:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("A full 40-character Git commit is required")
    spec = build_spec(
        commit,
        ARCHIVE_VOLUME,
        RUN_NAME,
        flow_refinement=True,
        evidence_flux_only=True,
    )
    spec["description"] += (
        " MEMatte ViT-S initialized from the official AIM-500 checkpoint "
        "(not DIM); mirrored to the archive volume and SHA-256 verified."
    )
    command = spec["run"][0]["command"]
    installer = "PYTHON_BIN=python network/scripts/install_official_mematte.sh"
    if command.count(installer) != 1:
        raise RuntimeError("Could not locate MEMatte installer")
    settings = "\n".join(
        (
            "export PRISM_MEMATTE_CHECKPOINT=/root/workspace/PRISM/network/checkpoints/MEMatte_ViTS_AIM500.pth",
            f"export PRISM_MEMATTE_CHECKPOINT_VOLUME={ARCHIVE_VOLUME}",
            f"export PRISM_MEMATTE_CHECKPOINT_KEY={CHECKPOINT_KEY}",
            f"export PRISM_MEMATTE_CHECKPOINT_SHA256={CHECKPOINT_SHA256}",
        )
    )
    spec["run"][0]["command"] = command.replace(installer, settings + "\n" + installer, 1)
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    spec = rerun_spec(args.git_commit)
    print(
        f"PRISM_V7R1_READY name={RUN_NAME} commit={args.git_commit} "
        f"mematte_sha256={CHECKPOINT_SHA256}",
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
    existing = [
        run
        for run in list_runs(
            organization_name="snu-eng-dgx-heavy", project_name="PRISM"
        )
        if run.run_spec.title == RUN_NAME
    ]
    if existing:
        print(f"PRISM_V7R1_ALREADY_EXISTS ids={[run.id for run in existing]}", flush=True)
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
    print(f"PRISM_V7R1_SUBMITTED id={response.run_execution.id}", flush=True)


if __name__ == "__main__":
    main()
