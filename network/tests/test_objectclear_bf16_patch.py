"""Guard the one-line compatibility patch used by the real-weight smoke."""

import importlib.util
from pathlib import Path

import pytest


PATCH_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_objectclear_bf16.py"
SPEC = importlib.util.spec_from_file_location("patch_objectclear_bf16", PATCH_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH)


def test_patch_casts_only_attention_map(tmp_path: Path) -> None:
    source = tmp_path / "objectclear" / "pipelines" / "pipeline_objectclear.py"
    source.parent.mkdir(parents=True)
    source.write_text(f"other = 1\nattn_np = {PATCH.NEEDLE} * 255.\n")
    assert PATCH.patch_checkout(tmp_path) == source
    assert source.read_text() == f"other = 1\nattn_np = {PATCH.REPLACEMENT} * 255.\n"


@pytest.mark.parametrize("source_text", ["other = 1\n", f"{PATCH.NEEDLE}\n{PATCH.NEEDLE}\n"])
def test_patch_rejects_unexpected_source(tmp_path: Path, source_text: str) -> None:
    source = tmp_path / "objectclear" / "pipelines" / "pipeline_objectclear.py"
    source.parent.mkdir(parents=True)
    source.write_text(source_text)
    with pytest.raises(RuntimeError, match="exactly one"):
        PATCH.patch_checkout(tmp_path)
    assert source.read_text() == source_text
