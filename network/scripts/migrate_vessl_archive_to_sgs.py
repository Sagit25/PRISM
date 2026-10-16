"""Copy a verified VESSL PRISM archive to an SGS PVC through a transfer pod.

The signed object URLs travel only over kubectl stdin, never through argv or
stdout. Each file is checked by size and SHA-256 before its final rename.
Re-running skips already verified files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import PurePosixPath
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

from repack_vessl_dataset import VesslObjectStore


DEFAULT_VOLUME = "prism-full-64-singleflow-pack-v2-369367271785-output"


def shell_quote(value: object) -> str:
    return shlex.quote(str(value))


def sha256_of_object(store: VesslObjectStore, key: str) -> str:
    response = store.source_client.get_object(Bucket=store.source_bucket, Key=key)
    digest = hashlib.sha256()
    body = response["Body"]
    try:
        for chunk in iter(lambda: body.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    finally:
        body.close()
    return digest.hexdigest()


def copy_one(
    store: VesslObjectStore,
    obj: object,
    expected_sha: str,
    *,
    kubeconfig: str,
    namespace: str,
    pod: str,
    destination_root: str,
) -> None:
    relative = obj.relative_path
    if PurePosixPath(relative).name != relative or not re.fullmatch(r"[A-Za-z0-9_.-]+", relative):
        raise ValueError(f"Unsafe archive filename: {relative}")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise ValueError(f"Invalid SHA-256 for {relative}")
    destination = f"{destination_root.rstrip('/')}/{relative}"
    existing_check = "\n".join(
        (
            f"target={shell_quote(destination)}",
            f"expected_size={shell_quote(obj.size)}",
            f"expected_sha={shell_quote(expected_sha)}",
            'test -f "$target" && '
            '[ "$(wc -c < "$target" | tr -d " ")" = "$expected_size" ] && '
            '[ "$(sha256sum "$target" | cut -d " " -f 1)" = "$expected_sha" ]',
        )
    ) + "\n"
    present = subprocess.run(
        ["kubectl", f"--kubeconfig={kubeconfig}", "-n", namespace,
         "exec", "-i", pod, "--", "sh", "-s"],
        input=existing_check, text=True, capture_output=True, timeout=3600,
        check=False,
    )
    if present.returncode == 0:
        print(f"SGS_ARCHIVE_ALREADY_VERIFIED {relative} bytes={obj.size}", flush=True)
        return
    url = store.source_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": store.source_bucket, "Key": obj.key},
        ExpiresIn=3600,
    )
    script = "\n".join(
        (
            "set -eu",
            f"target={shell_quote(destination)}",
            f"expected_size={shell_quote(obj.size)}",
            f"expected_sha={shell_quote(expected_sha)}",
            f"url={shell_quote(url)}",
            'mkdir -p "$(dirname "$target")"',
            'if [ -f "$target" ] && [ "$(wc -c < "$target" | tr -d " ")" = "$expected_size" ] && [ "$(sha256sum "$target" | cut -d " " -f 1)" = "$expected_sha" ]; then exit 0; fi',
            'rm -f "$target.part"',
            'if ! wget -S -T 120 -O "$target.part" "$url" >/tmp/prism-wget.log 2>&1; then '
            'if grep -qi "bad address\\|resolve" /tmp/prism-wget.log; then echo DNS_ERROR; '
            'elif grep -qi "TLS\\|SSL\\|certificate" /tmp/prism-wget.log; then echo TLS_ERROR; '
            'elif grep -qi "connection refused\\|network unreachable\\|timed out" /tmp/prism-wget.log; then echo NETWORK_ERROR; '
            'elif grep -q "HTTP/" /tmp/prism-wget.log; then echo HTTP_ERROR; '
            'else echo WGET_ERROR; fi; exit 21; fi',
            '[ "$(wc -c < "$target.part" | tr -d " ")" = "$expected_size" ] || exit 22',
            '[ "$(sha256sum "$target.part" | cut -d " " -f 1)" = "$expected_sha" ] || exit 23',
            'mv "$target.part" "$target"',
        )
    ) + "\n"
    result = subprocess.run(
        [
            "kubectl", f"--kubeconfig={kubeconfig}", "-n", namespace,
            "exec", "-i", pod, "--", "sh", "-s",
        ],
        input=script,
        text=True,
        capture_output=True,
        timeout=3600,
        check=False,
    )
    if result.returncode:
        # Do not include stderr: wget or kubectl could echo a signed URL.
        status = result.stdout.strip()
        if status not in {"DNS_ERROR", "TLS_ERROR", "NETWORK_ERROR", "HTTP_ERROR", "WGET_ERROR"}:
            status = "unknown error"
        raise RuntimeError(f"Copy/verification failed for {relative} (exit {result.returncode}, {status})")
    print(f"SGS_ARCHIVE_VERIFIED {relative} bytes={obj.size}", flush=True)


def copy_one_relay(
    store: VesslObjectStore,
    obj: object,
    expected_sha: str,
    *,
    kubeconfig: str,
    namespace: str,
    pod: str,
    destination_root: str,
) -> None:
    relative = obj.relative_path
    if PurePosixPath(relative).name != relative or not re.fullmatch(r"[A-Za-z0-9_.-]+", relative):
        raise ValueError(f"Unsafe archive filename: {relative}")
    destination = f"{destination_root.rstrip('/')}/{relative}"
    kubectl = ["kubectl", f"--kubeconfig={kubeconfig}", "-n", namespace]

    def remote(script: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*kubectl, "exec", "-i", pod, "--", "sh", "-s"],
            input=script, text=True, capture_output=True, timeout=3600, check=False,
        )

    verify = (
        f"target={shell_quote(destination)}\n"
        f"expected_size={shell_quote(obj.size)}\n"
        f"expected_sha={shell_quote(expected_sha)}\n"
        'test -f "$target" && '
        '[ "$(wc -c < "$target" | tr -d " ")" = "$expected_size" ] && '
        '[ "$(sha256sum "$target" | cut -d " " -f 1)" = "$expected_sha" ]\n'
    )
    if remote(verify).returncode == 0:
        print(f"SGS_ARCHIVE_ALREADY_VERIFIED {relative} bytes={obj.size}", flush=True)
        return

    with tempfile.TemporaryDirectory(prefix="prism-sgs-relay-") as temporary:
        local = Path(temporary) / relative
        store.download(obj, local)
        with local.open("rb") as stream:
            local_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        if local.stat().st_size != obj.size or local_sha != expected_sha:
            raise RuntimeError(f"Source download verification failed: {relative}")
        prepare = remote(f"mkdir -p {shell_quote(destination_root)}\n")
        if prepare.returncode:
            raise RuntimeError(f"Cannot prepare SGS destination for {relative}")
        uploaded = subprocess.run(
            [*kubectl, "cp", str(local), f"{pod}:{destination}.part"],
            text=True, capture_output=True, timeout=3600, check=False,
        )
        if uploaded.returncode:
            raise RuntimeError(f"SGS relay upload failed: {relative}")
        finalize = remote(
            f"target={shell_quote(destination)}\n"
            f"expected_size={shell_quote(obj.size)}\n"
            f"expected_sha={shell_quote(expected_sha)}\n"
            '[ "$(wc -c < "$target.part" | tr -d " ")" = "$expected_size" ] || exit 22\n'
            '[ "$(sha256sum "$target.part" | cut -d " " -f 1)" = "$expected_sha" ] || exit 23\n'
            'mv "$target.part" "$target"\n'
        )
        if finalize.returncode:
            raise RuntimeError(f"SGS relay verification failed: {relative} (exit {finalize.returncode})")
    print(f"SGS_ARCHIVE_VERIFIED {relative} bytes={obj.size}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubeconfig", required=True)
    parser.add_argument("--namespace", default="ws-wkuoxg0il92t0")
    parser.add_argument("--pod", default="prism-transfer")
    parser.add_argument("--volume", default=DEFAULT_VOLUME)
    parser.add_argument("--destination-root", default=f"/data/datasets/{DEFAULT_VOLUME}")
    parser.add_argument("--only", help="Copy one exact root filename for a smoke test")
    parser.add_argument("--relay", action="store_true", help="Relay through local temporary storage when pod DNS blocks S3")
    args = parser.parse_args()
    store = VesslObjectStore(
        storage_name="vessl-storage",
        source_volume_name=args.volume,
        destination_volume_name=args.volume,
    )
    objects = {item.relative_path: item for item in store.iter_objects("metadata")}
    required = {"archive_manifest.json", "dataset_manifest.json"}
    manifest_obj = objects["archive_manifest.json"]
    manifest_response = store.source_client.get_object(
        Bucket=store.source_bucket, Key=manifest_obj.key
    )
    manifest = json.load(manifest_response["Body"])
    if manifest.get("complete") is not True or manifest.get("totals", {}).get("shard_count") != 8:
        raise RuntimeError("Source archive is not verified complete")
    expected_tars = {}
    for component in manifest["components"].values():
        for shard in component["shards"]:
            expected_tars[shard["name"]] = (shard["sha256"], shard["tar_bytes"])
            required.add(shard["name"])
            required.add(shard["name"] + ".json")
    if args.only:
        if args.only not in required:
            raise ValueError("--only must name a required root archive file")
        required = {args.only}
    missing = required - objects.keys()
    if missing:
        raise RuntimeError(f"Missing source files: {sorted(missing)}")
    for name in sorted(required):
        obj = objects[name]
        if name in expected_tars:
            expected_sha, expected_size = expected_tars[name]
            if obj.size != expected_size:
                raise RuntimeError(f"Source archive size mismatch: {name}")
        else:
            expected_sha = sha256_of_object(store, obj.key)
        transfer = copy_one_relay if args.relay else copy_one
        transfer(
            store, obj, expected_sha,
            kubeconfig=args.kubeconfig,
            namespace=args.namespace,
            pod=args.pod,
            destination_root=args.destination_root,
        )
    print(f"SGS_ARCHIVE_COPY_COMPLETE files={len(required)}", flush=True)


if __name__ == "__main__":
    main()
