import pytest

torch = pytest.importorskip("torch")

from refractive_mam2.trimap import BG, FG, UNKNOWN, build_transparency_trimap


def test_transparent_interior_is_unknown() -> None:
    mask = torch.tensor([[[[0.0, 1.0, 1.0]]]])
    alpha = torch.tensor([[[[0.0, 0.2, 0.99]]]])
    trimap = build_transparency_trimap(mask, alpha, foreground_threshold=0.95)
    assert trimap.tolist() == [[[BG, UNKNOWN, FG]]]
