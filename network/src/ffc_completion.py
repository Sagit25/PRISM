"""LaMa/GLaMa-style Fourier completion for PRISM true holes.

This is an in-tree, dependency-free adaptation of the FFC-ResNet generator
family.  It keeps local and global feature streams separate and mixes the
global stream in the Fourier domain, giving every bottleneck feature access to
the full background canvas in a single forward pass.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0 and channels // groups >= 2:
            return groups
    return 1


def _split_channels(channels: int, global_ratio: float) -> tuple[int, int]:
    global_channels = int(round(channels * global_ratio))
    global_channels = min(max(global_channels, 0), channels)
    return channels - global_channels, global_channels


class FourierUnit(nn.Module):
    """Learn a 1x1 transform over real and imaginary Fourier coefficients."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.mix = nn.Conv2d(2 * in_channels, 2 * out_channels, 1, bias=False)
        self.norm = nn.GroupNorm(_groups(2 * out_channels), 2 * out_channels)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4:
            raise ValueError("FourierUnit input must have shape [B,C,H,W]")
        dtype = value.dtype
        # FFT support for fp16/bfloat16 depends on device and image size.  The
        # transform is therefore evaluated in fp32 while the learned 1x1
        # projection remains in the model/autocast dtype.
        spectrum = torch.fft.rfft2(value.float(), norm="ortho")
        features = torch.cat((spectrum.real, spectrum.imag), dim=1).to(dtype)
        features = self.activation(self.norm(self.mix(features)))
        real, imaginary = features.chunk(2, dim=1)
        reconstructed = torch.fft.irfft2(
            torch.complex(real.float(), imaginary.float()),
            s=value.shape[-2:],
            norm="ortho",
        )
        return reconstructed.to(dtype)


class SpectralTransform(nn.Module):
    """LaMa-style global branch with spatial and Fourier residual paths."""

    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("FFC stride must be one or two")
        hidden = max(out_channels // 2, 1)
        self.downsample = (
            nn.AvgPool2d(2, stride=2) if stride == 2 else nn.Identity()
        )
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.SiLU(inplace=True),
        )
        self.fourier = FourierUnit(hidden, hidden)
        self.post = nn.Conv2d(hidden, out_channels, 1, bias=False)

    def forward(self, value: Tensor) -> Tensor:
        value = self.pre(self.downsample(value))
        return self.post(value + self.fourier(value))


class FastFourierConvolution(nn.Module):
    """Exchange information between local and globally mixed feature paths."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        ratio_gin: float,
        ratio_gout: float,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if not 0.0 <= ratio_gin <= 1.0 or not 0.0 <= ratio_gout <= 1.0:
            raise ValueError("FFC global ratios must lie in [0,1]")
        in_local, in_global = _split_channels(in_channels, ratio_gin)
        out_local, out_global = _split_channels(out_channels, ratio_gout)
        padding = kernel_size // 2

        def spatial(source: int, target: int) -> nn.Module | None:
            if source == 0 or target == 0:
                return None
            return nn.Conv2d(
                source,
                target,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )

        self.in_local = in_local
        self.in_global = in_global
        self.out_local = out_local
        self.out_global = out_global
        self.local_to_local = spatial(in_local, out_local)
        self.local_to_global = spatial(in_local, out_global)
        self.global_to_local = spatial(in_global, out_local)
        self.global_to_global = (
            SpectralTransform(in_global, out_global, stride)
            if in_global and out_global
            else None
        )

    @staticmethod
    def _sum(first: Tensor | None, second: Tensor | None) -> Tensor | None:
        if first is None:
            return second
        if second is None:
            return first
        return first + second

    def forward(
        self,
        value: Tensor | tuple[Tensor | None, Tensor | None],
    ) -> tuple[Tensor | None, Tensor | None]:
        if isinstance(value, tuple):
            local, global_ = value
        else:
            local, global_ = value, None
        if local is not None and local.shape[1] != self.in_local:
            raise ValueError("FFC local input channel mismatch")
        if global_ is not None and global_.shape[1] != self.in_global:
            raise ValueError("FFC global input channel mismatch")

        local_output = self._sum(
            None if self.local_to_local is None else self.local_to_local(local),
            None if self.global_to_local is None else self.global_to_local(global_),
        )
        global_output = self._sum(
            None if self.local_to_global is None else self.local_to_global(local),
            (
                None
                if self.global_to_global is None
                else self.global_to_global(global_)
            ),
        )
        return local_output, global_output


class _FFCNormActivation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        ratio_gin: float,
        ratio_gout: float,
        kernel_size: int = 3,
        stride: int = 1,
        activate: bool = True,
    ) -> None:
        super().__init__()
        self.ffc = FastFourierConvolution(
            in_channels,
            out_channels,
            ratio_gin=ratio_gin,
            ratio_gout=ratio_gout,
            kernel_size=kernel_size,
            stride=stride,
        )
        local_channels, global_channels = _split_channels(out_channels, ratio_gout)
        self.local_norm = (
            nn.GroupNorm(_groups(local_channels), local_channels)
            if local_channels
            else None
        )
        self.global_norm = (
            nn.GroupNorm(_groups(global_channels), global_channels)
            if global_channels
            else None
        )
        self.activate = activate

    def _finish(self, value: Tensor | None, norm: nn.Module | None) -> Tensor | None:
        if value is None:
            return None
        value = norm(value) if norm is not None else value
        return F.silu(value, inplace=True) if self.activate else value

    def forward(
        self,
        value: Tensor | tuple[Tensor | None, Tensor | None],
    ) -> tuple[Tensor | None, Tensor | None]:
        local, global_ = self.ffc(value)
        return self._finish(local, self.local_norm), self._finish(
            global_, self.global_norm
        )


class _FFCResidualBlock(nn.Module):
    def __init__(self, channels: int, global_ratio: float) -> None:
        super().__init__()
        self.first = _FFCNormActivation(
            channels,
            channels,
            ratio_gin=global_ratio,
            ratio_gout=global_ratio,
        )
        self.second = _FFCNormActivation(
            channels,
            channels,
            ratio_gin=global_ratio,
            ratio_gout=global_ratio,
            activate=False,
        )

    def forward(
        self, value: tuple[Tensor | None, Tensor | None]
    ) -> tuple[Tensor | None, Tensor | None]:
        identity_local, identity_global = value
        local, global_ = self.second(self.first(value))
        if local is not None and identity_local is not None:
            local = F.silu(local + identity_local, inplace=True)
        if global_ is not None and identity_global is not None:
            global_ = F.silu(global_ + identity_global, inplace=True)
        return local, global_


class GLaMaCompletionNet(nn.Module):
    """Deterministic FFC-ResNet completion used at every PRISM iteration.

    Inputs are linear-RGB evidence, physical evidence coverage, and the true
    hole mask.  The network predicts a full RGB proposal; the caller hard
    composites it only inside true holes, so direct and inverse evidence stay
    bit-exact.
    """

    def __init__(
        self,
        width: int = 48,
        down_blocks: int = 3,
        residual_blocks: int = 6,
        global_ratio: float = 0.5,
        max_channels: int = 384,
    ) -> None:
        super().__init__()
        if width < 1 or down_blocks < 1 or residual_blocks < 1:
            raise ValueError(
                "FFC width/down_blocks/residual_blocks must be positive"
            )
        if not 0.0 < global_ratio < 1.0:
            raise ValueError(
                "FFC global_ratio must lie strictly between zero and one"
            )
        if max_channels < width:
            raise ValueError("FFC max_channels must be at least width")
        self.down_blocks = down_blocks
        self.stem = _FFCNormActivation(
            5,
            width,
            ratio_gin=0.0,
            ratio_gout=global_ratio,
            kernel_size=7,
        )
        encoders: list[nn.Module] = []
        channels = width
        for _ in range(down_blocks):
            next_channels = min(2 * channels, max_channels)
            encoders.append(
                _FFCNormActivation(
                    channels,
                    next_channels,
                    ratio_gin=global_ratio,
                    ratio_gout=global_ratio,
                    stride=2,
                )
            )
            channels = next_channels
        self.encoders = nn.ModuleList(encoders)
        self.context = nn.ModuleList(
            [_FFCResidualBlock(channels, global_ratio) for _ in range(residual_blocks)]
        )

        decoders: list[nn.Module] = []
        for _ in range(down_blocks):
            next_channels = max(width, channels // 2)
            decoders.append(
                nn.Sequential(
                    nn.Conv2d(channels, next_channels, 3, padding=1, bias=False),
                    nn.GroupNorm(_groups(next_channels), next_channels),
                    nn.SiLU(inplace=True),
                )
            )
            channels = next_channels
        self.decoders = nn.ModuleList(decoders)
        self.head = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, 3, 7),
        )

    @staticmethod
    def _concat(value: tuple[Tensor | None, Tensor | None]) -> Tensor:
        local, global_ = value
        if local is None:
            assert global_ is not None
            return global_
        if global_ is None:
            return local
        return torch.cat((local, global_), dim=1)

    def forward(
        self,
        evidence_background: Tensor,
        evidence_coverage: Tensor,
        true_hole: Tensor,
    ) -> Tensor:
        if evidence_background.ndim != 4 or evidence_background.shape[1] != 3:
            raise ValueError("evidence_background must have shape [B,3,H,W]")
        expected = (
            evidence_background.shape[0],
            1,
            *evidence_background.shape[-2:],
        )
        if evidence_coverage.shape != expected or true_hole.shape != expected:
            raise ValueError("coverage and true_hole must have shape [B,1,H,W]")

        height, width = evidence_background.shape[-2:]
        divisor = 2**self.down_blocks
        padded_height = math.ceil(height / divisor) * divisor
        padded_width = math.ceil(width / divisor) * divisor
        pad = (0, padded_width - width, 0, padded_height - height)
        inputs = torch.cat(
            (
                evidence_background,
                evidence_coverage.clamp(0.0, 1.0),
                true_hole.to(evidence_background.dtype),
            ),
            dim=1,
        )
        if pad[1] or pad[3]:
            # Replication is defined even for one-pixel spatial dimensions.
            inputs = F.pad(inputs, pad, mode="replicate")

        features = self.stem(inputs)
        for encoder in self.encoders:
            features = encoder(features)
        for block in self.context:
            features = block(features)
        decoded = self._concat(features)
        for decoder in self.decoders:
            decoded = F.interpolate(
                decoded,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )
            decoded = decoder(decoded)
        prediction = torch.sigmoid(self.head(decoded))
        return prediction[..., :height, :width]
