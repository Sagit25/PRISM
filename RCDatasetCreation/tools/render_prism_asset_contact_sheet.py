#!/usr/bin/env python3
"""Render local contact sheets for manual PRISM asset review."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont, ImageOps


def mesh_tile(path: Path, label: str, identifier: str, size: int) -> Image.Image:
    mesh = trimesh.load(path, force="mesh", process=False)
    rotation = trimesh.transformations.euler_matrix(
        math.radians(65), 0.0, math.radians(35), axes="sxyz"
    )
    vertices = trimesh.transform_points(mesh.vertices, rotation)
    faces = mesh.faces
    if len(faces) > 3000:
        indices = np.linspace(0, len(faces) - 1, 3000, dtype=np.int64)
        faces = faces[indices]
    triangles = vertices[faces]
    xy = triangles[..., :2]
    minimum = xy.reshape(-1, 2).min(axis=0)
    maximum = xy.reshape(-1, 2).max(axis=0)
    extent = np.maximum(maximum - minimum, 1e-8)
    scale = (size - 36) / extent.max()
    xy = (xy - (minimum + maximum) / 2.0) * scale + size / 2.0
    depth = triangles[..., 2].mean(axis=1)
    order = np.argsort(depth)
    image = Image.new("RGB", (size, size + 38), "white")
    draw = ImageDraw.Draw(image)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)
    shade = np.clip(0.35 + 0.55 * np.abs(normals[:, 2]), 0, 1)
    for index in order:
        value = int(70 + 150 * shade[index])
        points = [(float(x), float(size - y)) for x, y in xy[index]]
        draw.polygon(points, fill=(value, value + 8, min(value + 18, 255)))
    draw.rectangle((0, size, size, size + 38), fill=(245, 245, 245))
    font = ImageFont.load_default()
    draw.text((5, size + 3), label[:42], fill="black", font=font)
    draw.text((5, size + 19), identifier[:42], fill="black", font=font)
    return image


def background_tile(path: Path, label: str, size: int) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.fit(source.convert("RGB"), (size, size))
    canvas = Image.new("RGB", (size, size + 24), "white")
    canvas.paste(image)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, size, size, size + 24), fill=(245, 245, 245))
    draw.text((5, size + 5), label[:42], fill="black", font=ImageFont.load_default())
    return canvas


def pages(tiles: list[Image.Image], output: Path, prefix: str, columns: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = columns
    page_size = columns * rows
    for page_index, offset in enumerate(range(0, len(tiles), page_size), start=1):
        selected = tiles[offset : offset + page_size]
        width = selected[0].width
        height = selected[0].height
        sheet = Image.new("RGB", (columns * width, rows * height), (230, 230, 230))
        for index, tile in enumerate(selected):
            sheet.paste(tile, ((index % columns) * width, (index // columns) * height))
        sheet.save(output / f"{prefix}_{page_index:02d}.jpg", quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset_resources_research"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("asset_manifests/prism_research_assets_v1.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("asset_review"))
    parser.add_argument("--tile-size", type=int, default=220)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--background-limit", type=int, default=100)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    mesh_tiles = [
        mesh_tile(
            args.root / "shape" / item["file"],
            f"{item['split_pool']} | {item['source']} | {item['category']}",
            item["file"],
            args.tile_size,
        )
        for item in manifest["meshes"]
    ]
    pages(mesh_tiles, args.output, "meshes", args.columns)
    backgrounds = manifest["backgrounds"]
    if len(backgrounds) > args.background_limit:
        indices = np.linspace(
            0, len(backgrounds) - 1, args.background_limit, dtype=np.int64
        )
        backgrounds = [backgrounds[int(index)] for index in indices]
    background_tiles = [
        background_tile(
            args.root / "background" / item["file"],
            f"{item['split_pool']} | {item['source_id']}",
            args.tile_size,
        )
        for item in backgrounds
    ]
    pages(background_tiles, args.output, "backgrounds", args.columns)
    print(
        f"review sheets written to {args.output}: "
        f"meshes={len(mesh_tiles)} backgrounds={len(background_tiles)}"
    )


if __name__ == "__main__":
    main()
