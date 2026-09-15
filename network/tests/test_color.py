import pytest

torch = pytest.importorskip("torch")

from refractive_mam2 import linear_to_srgb, srgb_to_linear


def test_srgb_linear_round_trip() -> None:
    linear = torch.linspace(0.0, 1.0, 257)
    reconstructed = srgb_to_linear(linear_to_srgb(linear))
    assert torch.allclose(reconstructed, linear, atol=1e-6)
