"""Submit the isolated v7r2 PRISM rerun with hard predicted-mask support."""

from __future__ import annotations

import argparse
import re

import yaml

from submit_prism_full_64_exclusion1_v2 import PACK_RUN_ID
from submit_prism_full_64_v7r1 import (
    CHECKPOINT_SHA256,
    RUN_NAME as PREVIOUS_RUN_NAME,
    rerun_spec,
)


RUN_NAME = "prism-full-64-temporal-transmission-flux-v7r2-maskgate"


def build_spec(commit: str) -> dict:
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("A full 40-character Git commit is required")
    spec = rerun_spec(commit)
    command = spec["run"][0]["command"]
    if command.count(PREVIOUS_RUN_NAME) < 5:
        raise RuntimeError("Expected isolated output, checkpoint, and W&B paths")
    spec["name"] = RUN_NAME
    spec["run"][0]["command"] = command.replace(PREVIOUS_RUN_NAME, RUN_NAME)
    spec["description"] += (
        " Hard predicted object-mask support gates PAM alpha, G, flow, R, "
        "confidence, and geometry, with far-background leakage supervision."
    )
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    spec = build_spec(args.git_commit)
    print(
        f"PRISM_V7R2_READY name={RUN_NAME} commit={args.git_commit} "
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
        print(f"PRISM_V7R2_ALREADY_EXISTS ids={[run.id for run in existing]}", flush=True)
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
    print(f"PRISM_V7R2_SUBMITTED id={response.run_execution.id}", flush=True)


if __name__ == "__main__":
    main()
