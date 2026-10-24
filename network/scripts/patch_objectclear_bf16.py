"""Patch the pinned ObjectClear checkout's bf16 attention-map NumPy conversion.

ObjectClear's attention-guided fusion is inference-only, but NumPy cannot
consume bfloat16 tensors. Keep the model in bfloat16 and cast only this map.
"""

import argparse
from pathlib import Path


NEEDLE = "attn_map[i].mean(dim=0).cpu().numpy()"
REPLACEMENT = "attn_map[i].mean(dim=0).float().cpu().numpy()"


def patch_checkout(root: Path) -> Path:
    source = root / "objectclear" / "pipelines" / "pipeline_objectclear.py"
    original = source.read_text()
    if original.count(NEEDLE) != 1:
        raise RuntimeError(
            f"expected exactly one ObjectClear bf16 NumPy conversion in {source}; "
            f"found {original.count(NEEDLE)}"
        )
    source.write_text(original.replace(NEEDLE, REPLACEMENT))
    return source


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    print(f"PRISM_OBJECTCLEAR_BF16_NUMPY_PATCH_OK {patch_checkout(args.checkout)}", flush=True)
