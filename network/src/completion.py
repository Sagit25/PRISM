"""Optional true-hole background completion backends for PRISM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class DiffusionCompletionSettings:
    """Runtime settings for a frozen image-inpainting diffusion pipeline."""

    prompt: str = (
        "a clean static background, photorealistic, continuous texture, "
        "no foreground object"
    )
    negative_prompt: str = (
        "transparent object, foreground object, duplicate object, distortion, "
        "text, watermark"
    )
    inference_steps: int = 25
    guidance_scale: float = 7.5
    seed: int = 0
    mask_dilation: int = 8

    def __post_init__(self) -> None:
        if self.inference_steps < 1:
            raise ValueError("diffusion inference_steps must be positive")
        if self.guidance_scale < 0:
            raise ValueError("diffusion guidance_scale must be non-negative")
        if self.mask_dilation < 0:
            raise ValueError("diffusion mask_dilation must be non-negative")


def _linear_to_srgb(value: Tensor) -> Tensor:
    value = value.clamp(0.0, 1.0)
    return torch.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * value.pow(1.0 / 2.4) - 0.055,
    )


def _srgb_to_linear(value: Tensor) -> Tensor:
    value = value.clamp(0.0, 1.0)
    return torch.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055).pow(2.4),
    )


class FrozenDiffusionBackgroundCompleter(nn.Module):
    """Use a frozen Diffusers inpainting pipeline for final true holes only.

    The wrapper converts PRISM's linear-RGB evidence canvas to sRGB for the
    image diffusion model, then converts the generated result back to linear
    RGB. The caller remains responsible for exact evidence preservation by
    compositing the result only where the true-hole mask is one.

    ``pipeline`` is intentionally not registered as a trainable submodule.
    Diffusion weights are external assets and are not duplicated in PRISM's
    compact format-v5 checkpoint. A pre-trained LoRA may be loaded when the
    backend is constructed with :meth:`from_pretrained`.
    """

    def __init__(
        self,
        pipeline: Any,
        *,
        settings: DiffusionCompletionSettings | None = None,
        execution_device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_pipeline", pipeline)
        self.settings = settings or DiffusionCompletionSettings()
        self.execution_device = torch.device(execution_device)

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        adapter: str | None = None,
        revision: str | None = None,
        settings: DiffusionCompletionSettings | None = None,
        device: str | torch.device = "cuda",
        dtype: str = "float16",
    ) -> "FrozenDiffusionBackgroundCompleter":
        try:
            from diffusers import AutoPipelineForInpainting
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PRISM-Diffusion requires refractive-mam2[diffusion]"
            ) from exc

        dtype_by_name = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype not in dtype_by_name:
            raise ValueError("diffusion dtype must be float16, bfloat16, or float32")
        execution_device = torch.device(device)
        if execution_device.type == "cpu" and dtype == "float16":
            dtype = "float32"
        pipeline = AutoPipelineForInpainting.from_pretrained(
            model,
            torch_dtype=dtype_by_name[dtype],
            revision=revision,
        )
        if adapter is not None:
            if not hasattr(pipeline, "load_lora_weights"):
                raise TypeError("selected diffusion pipeline does not support LoRA")
            pipeline.load_lora_weights(adapter)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.to(execution_device)
        return cls(
            pipeline,
            settings=settings,
            execution_device=execution_device,
        )

    @property
    def pipeline(self) -> Any:
        return object.__getattribute__(self, "_pipeline")

    @staticmethod
    def _pil_image(value: Tensor):
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PRISM-Diffusion requires numpy and Pillow"
            ) from exc
        array = (
            value.detach()
            .permute(1, 2, 0)
            .mul(255.0)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        return Image.fromarray(np.asarray(array), mode="RGB")

    @staticmethod
    def _pil_mask(value: Tensor):
        try:
            import numpy as np
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PRISM-Diffusion requires numpy and Pillow"
            ) from exc
        array = (
            value.detach()
            .squeeze(0)
            .mul(255.0)
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        return Image.fromarray(np.asarray(array), mode="L")

    def _generator(self, offset: int = 0) -> torch.Generator:
        generator_device = (
            self.execution_device
            if self.execution_device.type == "cuda"
            else torch.device("cpu")
        )
        return torch.Generator(device=generator_device).manual_seed(
            self.settings.seed + offset
        )

    @torch.no_grad()
    def forward(
        self,
        evidence_background: Tensor,
        evidence_coverage: Tensor,
        true_hole: Tensor,
    ) -> Tensor:
        del evidence_coverage
        if evidence_background.ndim != 4 or evidence_background.shape[1] != 3:
            raise ValueError("evidence_background must have shape [B,3,H,W]")
        expected = (
            evidence_background.shape[0],
            1,
            *evidence_background.shape[-2:],
        )
        if true_hole.shape != expected:
            raise ValueError("true_hole must have shape [B,1,H,W]")

        outputs: list[Tensor] = []
        srgb_evidence = _linear_to_srgb(evidence_background)
        if self.settings.mask_dilation > 0:
            kernel = 2 * self.settings.mask_dilation + 1
            generation_hole = torch.nn.functional.max_pool2d(
                true_hole.float(),
                kernel,
                stride=1,
                padding=self.settings.mask_dilation,
            ) > 0
        else:
            generation_hole = true_hole
        for batch_index in range(evidence_background.shape[0]):
            if not bool(true_hole[batch_index].any()):
                outputs.append(evidence_background[batch_index])
                continue
            result = self.pipeline(
                prompt=self.settings.prompt,
                negative_prompt=self.settings.negative_prompt,
                image=self._pil_image(srgb_evidence[batch_index]),
                mask_image=self._pil_mask(generation_hole[batch_index].float()),
                num_inference_steps=self.settings.inference_steps,
                guidance_scale=self.settings.guidance_scale,
                generator=self._generator(batch_index),
            )
            if not getattr(result, "images", None):
                raise RuntimeError("diffusion inpainting pipeline returned no image")
            try:
                import numpy as np
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError("PRISM-Diffusion requires numpy") from exc
            image = torch.from_numpy(
                np.asarray(result.images[0].convert("RGB"), dtype=np.float32).copy()
            ).permute(2, 0, 1) / 255.0
            if image.shape[-2:] != evidence_background.shape[-2:]:
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    size=evidence_background.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            outputs.append(
                _srgb_to_linear(image).to(
                    device=evidence_background.device,
                    dtype=evidence_background.dtype,
                )
            )
        return torch.stack(outputs, dim=0)
