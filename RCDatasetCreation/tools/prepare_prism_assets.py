#!/usr/bin/env python3
"""Acquire and curate a reproducible PRISM research asset pack.

The generated binary assets intentionally live outside Git. A compact manifest
records source identity, license, hashes, selection rules, and generator index
files so the exact pack can be rebuilt and audited.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
import trimesh
from PIL import Image


DIV2K_TRAIN_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
DIV2K_VALID_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
OBJAVERSE_DATASET = "https://huggingface.co/datasets/allenai/objaverse"
FORMAT = "prism-research-assets-v1"
DEFAULT_CATEGORIES = (
    "bottle",
    "wine_glass",
    "vase",
    "bowl",
    "pitcher_(container)",
    "drinking_cup",
    "teapot",
    "sculpture",
)
LICENSES = {"by", "cc0"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, *values: str) -> str:
    return hashlib.sha256(
        ":".join((str(seed), *values)).encode("utf-8")
    ).hexdigest()


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "format": FORMAT,
            "sources": {},
            "selection": {},
            "meshes": [],
            "backgrounds": [],
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT:
        raise ValueError(f"unsupported asset manifest: {path}")
    return payload


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        current = load_manifest(path)
        current["sources"].update(payload.get("sources", {}))
        current["selection"].update(payload.get("selection", {}))
        for key in ("meshes", "backgrounds"):
            if payload.get(key):
                current[key] = payload[key]
        payload = current
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def download(url: str, path: Path, *, retries: int = 30) -> None:
    """Resume a source archive and atomically publish it after completion."""

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(1, retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    partial.unlink()
                    offset = 0
                mode = "ab" if offset else "wb"
                total = int(response.headers.get("Content-Length", 0)) + offset
                written = offset
                with partial.open(mode) as file:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        file.write(chunk)
                        written += len(chunk)
                        if written % (128 * 1024 * 1024) < len(chunk):
                            print(
                                f"download {path.name}: "
                                f"{written / 2**30:.2f}/{total / 2**30:.2f} GiB",
                                flush=True,
                            )
                if total and written != total:
                    raise IOError(f"short download: {written}/{total}")
            partial.replace(path)
            return
        except (requests.RequestException, OSError) as exc:
            if attempt == retries:
                raise
            current = partial.stat().st_size if partial.exists() else 0
            print(
                f"download retry {attempt}/{retries} at {current / 2**30:.2f} GiB: {exc}",
                flush=True,
            )


def prepare_backgrounds(
    root: Path,
    manifest: dict[str, Any],
    *,
    keep_archives: bool,
) -> None:
    background_dir = root / "background"
    archive_dir = root / ".downloads"
    background_dir.mkdir(parents=True, exist_ok=True)
    archives = (
        ("train", DIV2K_TRAIN_URL, archive_dir / "DIV2K_train_HR.zip", range(1, 801)),
        ("test", DIV2K_VALID_URL, archive_dir / "DIV2K_valid_HR.zip", range(801, 901)),
    )
    records: list[dict[str, Any]] = []
    indices: dict[str, list[str]] = {"train": [], "test": []}
    for split, url, archive, expected_ids in archives:
        if not archive.is_file():
            download(url, archive)
        with zipfile.ZipFile(archive) as bundle:
            members = {
                Path(name).name: name
                for name in bundle.namelist()
                if name.lower().endswith(".png")
            }
            for image_id in expected_ids:
                source_name = f"{image_id:04d}.png"
                member = members.get(source_name)
                if member is None:
                    raise RuntimeError(f"{archive}: missing {source_name}")
                output_name = f"div2k_{source_name}"
                output = background_dir / output_name
                if not output.is_file():
                    with bundle.open(member) as source, output.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                indices[split].append(output_name)
                records.append(
                    {
                        "file": output_name,
                        "split_pool": split,
                        "source": "DIV2K",
                        "source_id": f"{image_id:04d}",
                        "source_url": url,
                        "license": "academic-research-only; copyright original owner",
                        "bytes": output.stat().st_size,
                        "sha256": sha256(output),
                    }
                )
        if not keep_archives:
            archive.unlink()
    write_lines(background_dir / "train_background.txt", indices["train"])
    write_lines(background_dir / "test_background.txt", indices["test"])
    manifest["sources"]["DIV2K"] = {
        "homepage": "https://data.vision.ee.ethz.ch/cvl/DIV2K/",
        "train_archive": DIV2K_TRAIN_URL,
        "validation_archive": DIV2K_VALID_URL,
        "license": "academic research only; copyright remains with original owners",
    }
    manifest["backgrounds"] = records
    print(f"backgrounds prepared: train={len(indices['train'])} test={len(indices['test'])}")


def _load_objaverse(root: Path):
    try:
        import objaverse
    except ImportError as exc:
        raise ImportError(
            "mesh acquisition requires `python -m pip install objaverse==0.1.7`"
        ) from exc
    cache = root / ".downloads" / "objaverse"
    cache.mkdir(parents=True, exist_ok=True)
    # Objaverse 1.0 exposes these paths as module constants.
    if hasattr(objaverse, "BASE_PATH"):
        objaverse.BASE_PATH = str(cache)
    if hasattr(objaverse, "_VERSIONED_PATH"):
        objaverse._VERSIONED_PATH = str(cache / "hf-objaverse-v1")
    for path in sorted(cache.rglob("*.json.gz")):
        try:
            with gzip.open(path, "rb") as file:
                for _ in iter(lambda: file.read(1024 * 1024), b""):
                    pass
        except (EOFError, OSError):
            print(f"removing incomplete Objaverse cache shard: {path}", flush=True)
            path.unlink()
    return objaverse


def _as_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("source did not resolve to one mesh")
    mesh = loaded.copy()
    finite_vertices = np.isfinite(mesh.vertices).all(axis=1)
    if not bool(finite_vertices.all()):
        keep = finite_vertices[mesh.faces].all(axis=1)
        mesh.update_faces(keep)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()
    trimesh.repair.fix_normals(mesh, multibody=True)
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(mesh, multibody=True)
    return mesh


def _normalize_mesh(
    mesh: trimesh.Trimesh,
    *,
    gltf_y_up: bool = True,
) -> trimesh.Trimesh:
    if gltf_y_up:
        # glTF is Y-up while the generator's camera convention is Z-up.
        rotation = trimesh.transformations.rotation_matrix(np.pi / 2.0, (1, 0, 0))
        mesh.apply_transform(rotation)
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    maximum = float(mesh.extents.max())
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("degenerate mesh extent")
    mesh.apply_scale(0.9 / maximum)
    return mesh


def _voxel_solid(mesh: trimesh.Trimesh, pitch: float = 0.015) -> trimesh.Trimesh:
    """Convert a visually valid open surface into an explicit closed solid."""

    solid = mesh.voxelized(pitch).fill().marching_cubes
    # marching_cubes is expressed in voxel-index units.
    solid.apply_scale(pitch)
    solid = _normalize_mesh(solid, gltf_y_up=False)
    trimesh.repair.fix_normals(solid, multibody=True)
    return solid


def _mesh_metrics(mesh: trimesh.Trimesh) -> dict[str, Any]:
    components = mesh.split(only_watertight=False)
    component_faces = sorted((len(item.faces) for item in components), reverse=True)
    main_fraction = component_faces[0] / max(len(mesh.faces), 1)
    extents = mesh.extents
    positive = extents[extents > 1e-8]
    aspect = float(positive.max() / positive.min()) if len(positive) else float("inf")
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(len(components)),
        "main_component_face_fraction": float(main_fraction),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "aspect_ratio": aspect,
        "extents": [float(value) for value in extents],
    }


def _acceptable(metrics: dict[str, Any], min_faces: int, max_faces: int) -> bool:
    return bool(
        metrics["watertight"]
        and metrics["winding_consistent"]
        and min_faces <= metrics["faces"] <= max_faces
        and metrics["components"] <= 4
        and metrics["main_component_face_fraction"] >= 0.85
        and metrics["aspect_ratio"] <= 8.0
    )


def _procedural_mesh(seed: int, index: int) -> tuple[trimesh.Trimesh, str, dict[str, Any]]:
    rng = np.random.default_rng(seed + index * 1_000_003)
    family = index % 6
    parameters: dict[str, Any] = {}
    if family in (0, 1, 2):
        count = int(rng.integers(9, 15))
        z = np.linspace(-0.45, 0.45, count)
        phase = float(rng.uniform(-np.pi, np.pi))
        if family == 0:  # bottle/jar-like shoulder and neck
            body = float(rng.uniform(0.22, 0.36))
            neck = float(rng.uniform(0.08, 0.15))
            blend = 1.0 / (1.0 + np.exp(18.0 * (z - rng.uniform(0.15, 0.28))))
            radius = neck + (body - neck) * blend
            name = "hollow_bottle"
        elif family == 1:  # vase/decanter-like waist
            base = float(rng.uniform(0.20, 0.32))
            amplitude = float(rng.uniform(0.04, 0.12))
            radius = base + amplitude * np.sin(2.0 * np.pi * (z + 0.45) + phase)
            radius[-2:] *= np.linspace(0.9, rng.uniform(0.55, 0.85), 2)
            name = "hollow_vase"
        else:  # bowl/cup-like flared wall
            bottom = float(rng.uniform(0.12, 0.23))
            top = float(rng.uniform(0.30, 0.43))
            normalized = (z - z.min()) / np.ptp(z)
            radius = bottom + (top - bottom) * normalized ** rng.uniform(0.7, 1.5)
            name = "hollow_bowl"
        radius *= 1.0 + 0.025 * np.sin(
            np.linspace(0, np.pi * rng.uniform(1.0, 3.0), count) + phase
        )
        wall = float(rng.uniform(0.025, 0.055))
        inner_z = z[2:]
        inner_radius = np.maximum(radius[2:] - wall, 0.025)
        profile = np.column_stack((radius, z))
        inner = np.column_stack((inner_radius, inner_z))[::-1]
        profile = np.vstack((profile, inner, profile[:1]))
        mesh = trimesh.creation.revolve(profile, sections=int(rng.integers(48, 81)))
        parameters = {
            "profile_points": count,
            "wall_thickness": wall,
            "outer_radius": [float(value) for value in radius],
        }
    elif family == 3:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.45)
        scale = rng.uniform((0.55, 0.55, 0.35), (1.0, 1.0, 1.0))
        mesh.apply_scale(scale)
        name = "solid_lens_or_ellipsoid"
        parameters = {"axis_scale": [float(value) for value in scale]}
    elif family == 4:
        sections = int(rng.integers(3, 9))
        mesh = trimesh.creation.cylinder(
            radius=float(rng.uniform(0.22, 0.40)),
            height=float(rng.uniform(0.45, 0.90)),
            sections=sections,
        )
        name = "solid_prism"
        parameters = {"sections": sections}
    else:
        mesh = trimesh.creation.capsule(
            height=float(rng.uniform(0.35, 0.75)),
            radius=float(rng.uniform(0.14, 0.30)),
            count=(24, 24),
        )
        shear = np.eye(4)
        shear[0, 2] = float(rng.uniform(-0.22, 0.22))
        mesh.apply_transform(shear)
        name = "solid_asymmetric_ornament"
        parameters = {"xz_shear": float(shear[0, 2])}
    angle = float(rng.uniform(-np.pi, np.pi))
    mesh.apply_transform(trimesh.transformations.rotation_matrix(angle, (0, 0, 1)))
    mesh = _normalize_mesh(mesh, gltf_y_up=False)
    trimesh.repair.fix_normals(mesh, multibody=True)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise RuntimeError(f"procedural generator produced invalid {name}")
    parameters["z_rotation_radians"] = angle
    return mesh, name, parameters


def _category_uids(lvis: dict[str, list[str]], requested: str) -> list[str]:
    aliases = {
        "bottle": (
            "bottle",
            "water_bottle",
            "wine_bottle",
            "beer_bottle",
            "thermos_bottle",
            "jar",
            "perfume",
        ),
        "wine_glass": ("wineglass", "flute_glass", "glass_(drink_container)"),
        "bowl": ("bowl", "soup_bowl", "sugar_bowl", "fishbowl"),
        "pitcher_(container)": (
            "pitcher_(vessel_for_liquid)",
            "cream_pitcher",
        ),
        "drinking_cup": ("cup", "teacup", "shot_glass"),
        "teapot": ("teapot", "teakettle"),
        "sculpture": ("sculpture", "statue_(sculpture)"),
    }
    keys = aliases.get(requested, (requested,))
    result: list[str] = []
    for key in keys:
        result.extend(lvis.get(key, ()))
    return list(dict.fromkeys(result))


def _quotas(total: int, categories: tuple[str, ...]) -> dict[str, int]:
    base, remainder = divmod(total, len(categories))
    return {
        category: base + int(index < remainder)
        for index, category in enumerate(categories)
    }


def prepare_meshes(
    root: Path,
    manifest: dict[str, Any],
    *,
    train_count: int,
    test_count: int,
    seed: int,
    categories: tuple[str, ...],
    min_faces: int,
    max_faces: int,
    overfetch: int,
    processes: int,
    public_meshes: int,
) -> None:
    objaverse = _load_objaverse(root)
    shape_dir = root / "shape"
    shape_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("objaverse_*.ply", "procedural_*.ply"):
        for stale in shape_dir.glob(pattern):
            stale.unlink()
    lvis = objaverse.load_lvis_annotations()
    total_count = train_count + test_count
    public_count = min(public_meshes, total_count)
    public_test = min(test_count, round(public_count * test_count / total_count))
    public_train = public_count - public_test
    train_targets = _quotas(public_train, categories)
    test_targets = _quotas(public_test, categories)
    candidates_by_category: dict[str, list[str]] = {}
    for category in categories:
        candidates = sorted(
            _category_uids(lvis, category),
            key=lambda uid: stable_key(seed, category, uid),
        )
        if not candidates:
            print(f"warning: Objaverse LVIS category absent: {category}", file=sys.stderr)
        limit = (train_targets[category] + test_targets[category]) * overfetch
        candidates_by_category[category] = candidates[:limit]
    # Interleave categories so downloads and rejection decisions cannot let a
    # large LVIS category crowd out smaller transparent-object categories.
    interleaved: list[tuple[str, str]] = []
    longest = max((len(values) for values in candidates_by_category.values()), default=0)
    for position in range(longest):
        for category in categories:
            values = candidates_by_category[category]
            if position < len(values):
                interleaved.append((category, values[position]))
    unique_candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for category, uid in interleaved:
        if uid not in seen:
            seen.add(uid)
            unique_candidates.append((category, uid))

    desired = public_count
    candidates = unique_candidates
    annotations = objaverse.load_annotations([uid for _, uid in candidates])
    candidates = [
        (category, uid)
        for category, uid in candidates
        if annotations.get(uid, {}).get("license") in LICENSES
        and not annotations.get(uid, {}).get("isAgeRestricted", False)
        and min_faces
        <= int(annotations.get(uid, {}).get("faceCount") or min_faces)
        <= max_faces * 2
    ]
    if len(candidates) < desired:
        print(
            f"warning: only {len(candidates)} licensed metadata candidates for "
            f"{desired} requested public assets",
            file=sys.stderr,
        )

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    accepted_counts: dict[tuple[str, str], int] = defaultdict(int)
    batch_size = max(processes * 4, 8)
    for offset in range(0, len(candidates), batch_size):
        if len(accepted) >= desired:
            break
        batch = candidates[offset : offset + batch_size]
        paths = objaverse.load_objects(
            uids=[uid for _, uid in batch],
            download_processes=processes,
        )
        for category, uid in batch:
            if len(accepted) >= desired:
                break
            if accepted_counts[("train", category)] < train_targets[category]:
                split = "train"
            elif accepted_counts[("test", category)] < test_targets[category]:
                split = "test"
            else:
                continue
            source_value = paths.get(uid)
            if source_value is None:
                rejected.append({"uid": uid, "reason": "download_failed"})
                continue
            source = Path(source_value)
            try:
                source_hash = sha256(source)
                mesh = _normalize_mesh(_as_mesh(source))
                metrics = _mesh_metrics(mesh)
                geometry_repair = "native_or_hole_fill"
                if not _acceptable(metrics, min_faces, max_faces):
                    mesh = _voxel_solid(mesh)
                    metrics = _mesh_metrics(mesh)
                    geometry_repair = "voxel_solid_pitch_0.015"
                if not _acceptable(metrics, min_faces, max_faces):
                    rejected.append(
                        {"uid": uid, "reason": json.dumps(metrics, sort_keys=True)}
                    )
                    continue
                filename = f"objaverse_{uid}.ply"
                output = shape_dir / filename
                mesh.export(output, file_type="ply", encoding="binary")
                annotation = annotations[uid]
                thumbnail = None
                thumbnails = annotation.get("thumbnails", {}).get("images", [])
                if thumbnails:
                    thumbnail = max(
                        thumbnails,
                        key=lambda item: int(item.get("width", 0))
                        * int(item.get("height", 0)),
                    ).get("url")
                accepted.append(
                    {
                        "file": filename,
                        "split_pool": split,
                        "source": "Objaverse-1.0",
                        "source_uid": uid,
                        "source_name": annotation.get("name"),
                        "source_url": annotation.get("viewerUrl"),
                        "thumbnail_url": thumbnail,
                        "category": category,
                        "license": annotation.get("license"),
                        "source_sha256": source_hash,
                        "processed_sha256": sha256(output),
                        "processing": "flatten; clean; repair; glTF Y-up to Z-up; center; max extent 0.9; binary PLY",
                        "geometry_repair": geometry_repair,
                        **metrics,
                    }
                )
                accepted_counts[(split, category)] += 1
            except Exception as exc:  # retain a complete rejection audit
                rejected.append({"uid": uid, "reason": f"{type(exc).__name__}: {exc}"})
        print(f"mesh curation: accepted={len(accepted)}/{desired} rejected={len(rejected)}")
    if len(accepted) < desired:
        missing = {
            f"{split}:{category}": target - accepted_counts[(split, category)]
            for split, targets in (("train", train_targets), ("test", test_targets))
            for category, target in targets.items()
            if accepted_counts[(split, category)] < target
        }
        print(
            f"warning: strict filters accepted {len(accepted)}/{desired} public "
            f"meshes; missing={missing}; filling with procedural solids",
            file=sys.stderr,
        )

    split_counts = defaultdict(int)
    for item in accepted:
        split_counts[item["split_pool"]] += 1
    procedural_index = 0
    for split, target in (("train", train_count), ("test", test_count)):
        while split_counts[split] < target:
            mesh, family, parameters = _procedural_mesh(
                seed + (0 if split == "train" else 10_000_000),
                procedural_index,
            )
            filename = f"procedural_{split}_{procedural_index:04d}_{family}.ply"
            output = shape_dir / filename
            mesh.export(output, file_type="ply", encoding="binary")
            metrics = _mesh_metrics(mesh)
            accepted.append(
                {
                    "file": filename,
                    "split_pool": split,
                    "source": "PRISM-procedural-v1",
                    "source_uid": f"{split}:{seed}:{procedural_index}",
                    "source_name": family,
                    "source_url": None,
                    "thumbnail_url": None,
                    "category": family,
                    "license": "CC0-1.0",
                    "source_sha256": None,
                    "processed_sha256": sha256(output),
                    "processing": "deterministic closed solid; center; max extent 0.9; binary PLY",
                    "parameters": parameters,
                    **metrics,
                }
            )
            split_counts[split] += 1
            procedural_index += 1

    train_files = [item["file"] for item in accepted if item["split_pool"] == "train"]
    test_files = [item["file"] for item in accepted if item["split_pool"] == "test"]
    write_lines(shape_dir / "train_shape.txt", train_files)
    write_lines(shape_dir / "test_shape.txt", test_files)
    with (shape_dir / "ATTRIBUTION.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("file", "source_uid", "source_name", "source_url", "license"),
        )
        writer.writeheader()
        for item in accepted:
            writer.writerow({key: item.get(key) for key in writer.fieldnames})
    manifest["sources"]["Objaverse-1.0"] = {
        "homepage": OBJAVERSE_DATASET,
        "dataset_license": "ODC-By-1.0",
        "accepted_object_licenses": sorted(LICENSES),
        "api_package": "objaverse==0.1.7",
    }
    manifest["selection"]["meshes"] = {
        "seed": seed,
        "categories": list(categories),
        "train_count": train_count,
        "test_count": test_count,
        "requested_public_count": public_meshes,
        "accepted_public_count": sum(
            item["source"] == "Objaverse-1.0" for item in accepted
        ),
        "procedural_count": sum(
            item["source"] == "PRISM-procedural-v1" for item in accepted
        ),
        "train_category_targets": train_targets,
        "test_category_targets": test_targets,
        "minimum_faces": min_faces,
        "minimum_procedural_faces": 12,
        "maximum_faces": max_faces,
        "overfetch": overfetch,
        "requirements": {
            "watertight": True,
            "winding_consistent": True,
            "maximum_components": 4,
            "minimum_main_component_face_fraction": 0.85,
            "maximum_aspect_ratio": 8.0,
            "allowed_repairs": ["native_or_hole_fill", "voxel_solid_pitch_0.015"],
        },
        "rejected_count": len(rejected),
        "rejected": rejected,
    }
    manifest["meshes"] = accepted
    counts = defaultdict(int)
    for item in accepted:
        counts[(item["split_pool"], item["category"])] += 1
    print("mesh category counts:")
    for (split, category), count in sorted(counts.items()):
        print(f"  {split:5s} {category:24s} {count}")


def validate(root: Path, manifest: dict[str, Any]) -> None:
    errors: list[str] = []
    seen_hashes: dict[str, dict[str, str]] = {"mesh": {}, "background": {}}
    for kind, directory, records, hash_key in (
        ("mesh", root / "shape", manifest.get("meshes", []), "processed_sha256"),
        ("background", root / "background", manifest.get("backgrounds", []), "sha256"),
    ):
        for item in records:
            path = directory / item["file"]
            if not path.is_file():
                errors.append(f"missing {kind}: {path}")
            else:
                digest = sha256(path)
                if digest != item[hash_key]:
                    errors.append(f"hash mismatch {kind}: {path}")
                previous = seen_hashes[kind].get(digest)
                if previous is not None:
                    errors.append(f"duplicate {kind} content: {previous}, {item['file']}")
                seen_hashes[kind][digest] = item["file"]
                if kind == "mesh":
                    try:
                        mesh = _as_mesh(path)
                        metrics = _mesh_metrics(mesh)
                        selection = manifest.get("selection", {}).get("meshes", {})
                        required_minimum = (
                            int(selection.get("minimum_procedural_faces", 12))
                            if item.get("source") == "PRISM-procedural-v1"
                            else int(selection.get("minimum_faces", 500))
                        )
                        if not _acceptable(
                            metrics,
                            required_minimum,
                            int(selection.get("maximum_faces", 200_000)),
                        ):
                            errors.append(f"mesh no longer meets quality contract: {path}")
                    except Exception as exc:  # noqa: BLE001 - validation must aggregate failures
                        errors.append(f"unreadable mesh {path}: {exc}")
                else:
                    try:
                        with Image.open(path) as image:
                            image.verify()
                        with Image.open(path) as image:
                            if min(image.size) < 512:
                                errors.append(
                                    f"background below 512 px: {path} size={image.size}"
                                )
                    except Exception as exc:  # noqa: BLE001 - validation must aggregate failures
                        errors.append(f"unreadable background {path}: {exc}")
    for relative, minimum in (
        ("shape/train_shape.txt", 2),
        ("shape/test_shape.txt", 1),
        ("background/train_background.txt", 2),
        ("background/test_background.txt", 1),
    ):
        path = root / relative
        values = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
        if len(values) < minimum:
            errors.append(f"{relative} has {len(values)} entries; expected >= {minimum}")
        missing = [value for value in values if not (path.parent / value).is_file()]
        if missing:
            errors.append(f"{relative} references missing files: {missing[:5]}")
    train_shapes = set((root / "shape/train_shape.txt").read_text().splitlines())
    test_shapes = set((root / "shape/test_shape.txt").read_text().splitlines())
    train_backgrounds = set((root / "background/train_background.txt").read_text().splitlines())
    test_backgrounds = set((root / "background/test_background.txt").read_text().splitlines())
    if train_shapes & test_shapes:
        errors.append("train/test mesh overlap")
    if train_backgrounds & test_backgrounds:
        errors.append("train/test background overlap")
    manifest_shapes = {
        item["file"] for item in manifest.get("meshes", [])
    }
    manifest_backgrounds = {
        item["file"] for item in manifest.get("backgrounds", [])
    }
    if train_shapes | test_shapes != manifest_shapes:
        errors.append("mesh index files do not exactly match the manifest")
    if train_backgrounds | test_backgrounds != manifest_backgrounds:
        errors.append("background index files do not exactly match the manifest")
    invalid_licenses = [
        item["file"]
        for item in manifest.get("meshes", [])
        if item.get("source") == "Objaverse-1.0"
        and item.get("license") not in LICENSES
    ]
    if invalid_licenses:
        errors.append(f"disallowed Objaverse licenses: {invalid_licenses[:5]}")
    if errors:
        raise RuntimeError("asset validation failed:\n" + "\n".join(errors))
    print(
        "ASSET PACK PASS "
        f"train_mesh={len(train_shapes)} test_mesh={len(test_shapes)} "
        f"train_background={len(train_backgrounds)} test_background={len(test_backgrounds)}"
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "command",
        choices=("backgrounds", "meshes", "all", "validate"),
    )
    result.add_argument(
        "--root",
        type=Path,
        default=Path("dataset_resources_research"),
    )
    result.add_argument(
        "--manifest",
        type=Path,
        default=Path("asset_manifests/prism_research_assets_v1.json"),
    )
    result.add_argument("--train-meshes", type=int, default=120)
    result.add_argument("--test-meshes", type=int, default=30)
    result.add_argument("--seed", type=int, default=20260821)
    result.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    result.add_argument("--min-faces", type=int, default=500)
    result.add_argument("--max-faces", type=int, default=200_000)
    result.add_argument("--overfetch", type=int, default=12)
    result.add_argument("--processes", type=int, default=min(os.cpu_count() or 1, 8))
    result.add_argument(
        "--public-meshes",
        type=int,
        default=40,
        help="maximum quality-filtered Objaverse subset; remainder is procedural",
    )
    result.add_argument("--keep-archives", action="store_true")
    return result


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.train_meshes < 2 or args.test_meshes < 1:
        raise SystemExit("train meshes must be >=2 and test meshes >=1")
    if args.processes < 1 or args.overfetch < 1:
        raise SystemExit("processes and overfetch must be positive")
    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    if args.command in ("backgrounds", "all"):
        prepare_backgrounds(root, manifest, keep_archives=args.keep_archives)
        save_manifest(manifest_path, manifest)
    if args.command in ("meshes", "all"):
        prepare_meshes(
            root,
            manifest,
            train_count=args.train_meshes,
            test_count=args.test_meshes,
            seed=args.seed,
            categories=tuple(args.categories),
            min_faces=args.min_faces,
            max_faces=args.max_faces,
            overfetch=args.overfetch,
            processes=args.processes,
            public_meshes=args.public_meshes,
        )
        save_manifest(manifest_path, manifest)
    if args.command in ("validate", "all"):
        validate(root, manifest)
    save_manifest(manifest_path, manifest)
    print(f"manifest: {manifest_path} sha256={sha256(manifest_path)}")


if __name__ == "__main__":
    main()
