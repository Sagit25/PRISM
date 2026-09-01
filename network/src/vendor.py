"""Locations and bootstrap helpers for external, pinned research assets."""

from __future__ import annotations

import os
from pathlib import Path
import sys


NETWORK_ROOT = Path(__file__).resolve().parents[1]
SAM2_PINNED_REVISION = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_MODEL_NAME = "sam2.1_hiera_large"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_ROOT = Path(
    os.environ.get("PRISM_SAM2_ROOT", NETWORK_ROOT / "third_party" / "sam2")
).expanduser()
SAM2_CHECKPOINT = Path(
    os.environ.get(
        "PRISM_SAM2_CHECKPOINT",
        NETWORK_ROOT / "checkpoints" / f"{SAM2_MODEL_NAME}.pt",
    )
).expanduser()


def activate_vendored_sam2() -> Path | None:
    """Make the pinned source checkout importable without a global SAM2 install."""

    if not (SAM2_ROOT / "sam2" / "__init__.py").is_file():
        return None
    root = str(SAM2_ROOT.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return SAM2_ROOT


def sam2_setup_hint() -> str:
    script = NETWORK_ROOT / "scripts" / "install_official_sam2.sh"
    return f"run {script} to install the pinned official SAM2 source and checkpoint"
