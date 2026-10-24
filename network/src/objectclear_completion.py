"""Flow-conditioned adapter around the released ObjectClear inpainting model.

ObjectClear's denoising pipeline is intentionally frozen. By default its
attention-guided-fusion output is encoded back to the released VAE's latent
space, then corrected by a small trainable module conditioned on inverse-
refracted evidence and splatted flow. This keeps the expensive image prior
reusable while allowing a PRISM reconstruction loss to update the optical
operator and the background adapter.
"""

from __future__ import annotations

import importlib
import inspect
import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .completion import _linear_to_srgb, _srgb_to_linear


def _ensure_torch_safetensors_metadata(snapshot: Path) -> list[str]:
    """Make released ObjectClear weights readable by Transformers 4.45.

    Some released safetensors archives omit ``__metadata__``. That version of
    Transformers dereferences ``safe_open(...).metadata().get`` unconditionally.
    Rewrite only affected local snapshot files, streaming tensor bytes without
    loading the (multi-GB) checkpoints into RAM or modifying Hub cache blobs.
    """
    repaired: list[str] = []
    for path in sorted(snapshot.rglob("*.safetensors")):
        with path.open("rb") as original:
            length_bytes = original.read(8)
            if len(length_bytes) != 8:
                raise ValueError(f"invalid safetensors header: {path}")
            header_length = int.from_bytes(length_bytes, "little")
            if not 2 <= header_length <= 100_000_000:
                raise ValueError(f"invalid safetensors header length: {path}")
            header_bytes = original.read(header_length)
            if len(header_bytes) != header_length:
                raise ValueError(f"truncated safetensors header: {path}")
            header = json.loads(header_bytes)
            metadata = header.get("__metadata__")
            if metadata is not None:
                if metadata.get("format") not in (None, "pt"):
                    raise ValueError(f"unexpected safetensors format in {path}: {metadata}")
                if metadata.get("format") == "pt":
                    continue
            header["__metadata__"] = {**(metadata or {}), "format": "pt"}
            encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            encoded += b" " * (-len(encoded) % 8)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".prism-safe-", delete=False) as output:
                temporary = Path(output.name)
                try:
                    output.write(len(encoded).to_bytes(8, "little"))
                    output.write(encoded)
                    shutil.copyfileobj(original, output, length=8 * 1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
                    os.replace(temporary, path)
                finally:
                    temporary.unlink(missing_ok=True)
            repaired.append(str(path.relative_to(snapshot)))
    return repaired


@dataclass(frozen=True)
class ObjectClearSettings:
    prompt: str = "remove the instance of object"
    inference_steps: int = 20
    guidance_scale: float = 1.0
    seed: int = 42
    min_working_side: int = 512
    mask_dilation: int = 0
    latent_delta: float = 0.5
    cache_entries: int = 512
    use_fused_output: bool = True
    late_denoise_steps: int = 0

    def __post_init__(self) -> None:
        if self.inference_steps < 1 or self.min_working_side < 1:
            raise ValueError("ObjectClear steps and working side must be positive")
        if self.guidance_scale < 0 or self.mask_dilation < 0:
            raise ValueError("ObjectClear guidance and dilation cannot be negative")
        if self.latent_delta <= 0 or self.cache_entries < 0:
            raise ValueError("ObjectClear latent delta must be positive; cache >= 0")
        if self.late_denoise_steps < 0 or self.late_denoise_steps >= self.inference_steps:
            raise ValueError("late denoising steps must be in [0, inference_steps)")
        if self.late_denoise_steps and self.use_fused_output:
            raise ValueError("late denoising gradients require raw latent output, not PIL fusion")


class _FlowConditionedObjectClearLatent(nn.Module):
    """Zero-initialized residual in the released SDXL VAE's four-channel space."""

    def __init__(self, max_delta: float) -> None:
        super().__init__()
        self.max_delta = max_delta
        # latent(4), direct canvas(3), inverse radiance(3), confidence(1),
        # splatted flow(2), editable mask(1).
        self.body = nn.Sequential(
            nn.Conv2d(14, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 4, 3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(
        self,
        latent: Tensor,
        canvas: Tensor,
        inverse: Tensor,
        weight: Tensor,
        flow: Tensor,
        editable: Tensor,
    ) -> Tensor:
        if latent.ndim != 4 or latent.shape[1] != 4:
            raise ValueError("ObjectClear must return [B,4,h,w] SDXL latents")
        batch, _, height, width = latent.shape
        size = canvas.shape[-2:]
        if canvas.shape != (batch, 3, *size) or inverse.shape != canvas.shape:
            raise ValueError("canvas and inverse evidence must be [B,3,H,W]")
        if weight.shape != (batch, 1, *size) or editable.shape != weight.shape:
            raise ValueError("weight and editable mask must be [B,1,H,W]")
        if flow.shape != (batch, 2, *size):
            raise ValueError("splatted flow must be [B,2,H,W]")

        def resize(value: Tensor) -> Tensor:
            return F.interpolate(value.float(), size=(height, width), mode="bilinear", align_corners=False)

        scale = flow.new_tensor((max(size[1] - 1, 1), max(size[0] - 1, 1))).view(1, 2, 1, 1)
        context = torch.cat(
            (
                latent.float(), resize(canvas), resize(inverse),
                resize(weight.clamp(0, 1)), resize(flow / scale), resize(editable),
            ),
            dim=1,
        )
        delta = torch.tanh(self.body(context)) * self.max_delta
        delta = delta * F.interpolate(editable.float(), size=(height, width), mode="nearest")
        return latent + delta.to(latent.dtype)


class FlowGuidedObjectClearCompleter(nn.Module):
    """Pretrained ObjectClear prior plus jointly trainable optical latent adapter.

    The external pipeline is not a registered ``nn.Module``: its large frozen
    weights stay out of PRISM checkpoints.  Only ``flow_latent_adapter`` is
    saved and optimized.  A fixed seed and exact input cache make the frozen
    prior repeatable across epochs; flow evidence is *not* cached. The default
    cache holds latents obtained from ObjectClear's fused final image, not its
    raw pre-fusion diffusion latents.
    """

    flow_latent_guidance = True
    requires_object_reference = True

    def __init__(
        self,
        pipeline: Any,
        *,
        settings: ObjectClearSettings | None = None,
        execution_device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if not hasattr(pipeline, "vae"):
            raise TypeError("official ObjectClear pipeline must expose its VAE")
        object.__setattr__(self, "_pipeline", pipeline)
        self.settings = settings or ObjectClearSettings()
        self.execution_device = torch.device(execution_device)
        if (
            getattr(pipeline.vae, "dtype", None) == torch.float16
            and getattr(pipeline.vae.config, "force_upcast", False)
        ):
            raise ValueError(
                "ObjectClear joint VAE decoding needs bfloat16 or float32; "
                "the released fp16 VAE requires an upcast that is unsafe "
                "between cached prior calls and autograd backward"
            )
        pipeline.vae.requires_grad_(False)
        if self.settings.late_denoise_steps:
            components = getattr(pipeline, "components", {})
            for component in components.values():
                if isinstance(component, nn.Module):
                    component.requires_grad_(False).eval()
        self.flow_latent_adapter = _FlowConditionedObjectClearLatent(
            self.settings.latent_delta
        )
        self._latent_cache: OrderedDict[bytes, Tensor] = OrderedDict()
        self.image_cache_hits = 0
        self.image_cache_misses = 0

    @property
    def pipeline(self) -> Any:
        return object.__getattribute__(self, "_pipeline")

    @property
    def image_cache_size(self) -> int:
        return len(self._latent_cache)

    @classmethod
    def from_pretrained(
        cls,
        model: str,
        *,
        source_root: str | Path | None = None,
        revision: str | None = None,
        settings: ObjectClearSettings | None = None,
        device: str | torch.device = "cuda",
        dtype: str = "bfloat16",
        cpu_offload: bool = True,
    ) -> "FlowGuidedObjectClearCompleter":
        if source_root is not None:
            root = Path(source_root).resolve()
            if not (root / "objectclear" / "pipelines" / "pipeline_objectclear.py").is_file():
                raise FileNotFoundError("source_root must be the official ObjectClear checkout")
            sys.path.insert(0, str(root))
        try:
            module = importlib.import_module("objectclear.pipelines")
        except ImportError as exc:
            raise ImportError("install the official ObjectClear package or set source_root") from exc
        finally:
            if source_root is not None:
                sys.path.remove(str(root))
        if source_root is not None and not Path(module.__file__).resolve().is_relative_to(root):
            raise RuntimeError("imported ObjectClear is not from the requested source_root")
        kind = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        if dtype not in kind:
            raise ValueError("ObjectClear dtype must be float16, bfloat16, or float32")
        if dtype == "float16":
            raise ValueError("ObjectClear joint training requires bfloat16 or float32 VAE decoding")
        target = torch.device(device)
        if target.type == "cpu" and dtype != "float32":
            dtype = "float32"
        resolved = settings or ObjectClearSettings()
        if resolved.late_denoise_steps and cpu_offload and target.type == "cuda":
            raise ValueError("late-denoising backward requires ObjectClear without CPU offload")
        source = model
        if revision is not None and not Path(model).is_dir():
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError
            from requests.exceptions import RequestException

            for attempt in range(1, 5):
                try:
                    source = snapshot_download(
                        repo_id=model, revision=revision,
                        ignore_patterns=["*.fp16.safetensors", "*.fp16.bin"],
                        max_workers=2,
                    )
                    break
                except (LocalEntryNotFoundError, RequestException) as exc:
                    if attempt == 4:
                        raise
                    delay = 10 * attempt
                    print(
                        f"PRISM_OBJECTCLEAR_DOWNLOAD_RETRY attempt={attempt} "
                        f"wait_s={delay} error={type(exc).__name__}", flush=True,
                    )
                    time.sleep(delay)
            repaired = _ensure_torch_safetensors_metadata(Path(source))
            print("PRISM_OBJECTCLEAR_SAFETENSORS_METADATA_OK", repaired, flush=True)
        pipeline = module.ObjectClearPipeline.from_pretrained_with_custom_modules(
            source,
            torch_dtype=kind[dtype],
            variant="fp16" if dtype == "float16" else None,
            apply_attention_guided_fusion=resolved.use_fused_output,
        )
        pipeline.set_progress_bar_config(disable=True)
        if cpu_offload and target.type == "cuda":
            pipeline.enable_model_cpu_offload(device=str(target))
        else:
            pipeline.to(target)
        return cls(pipeline, settings=resolved, execution_device=target)

    @staticmethod
    def _pil(value: Tensor, *, mask: bool = False):
        import numpy as np
        from PIL import Image

        array = (value.detach().clamp(0, 1) * 255).round().to(torch.uint8).cpu()
        if mask:
            return Image.fromarray(np.asarray(array[0].numpy()), mode="L")
        return Image.fromarray(np.asarray(array.permute(1, 2, 0).numpy()), mode="RGB")

    def _prior_input(self, image: Tensor, hole: Tensor, object_reference: Tensor):
        # The released ObjectClear encoder extracts an embedding from pixels
        # *inside* its removal mask. A zero-filled evidence canvas would make
        # that embedding blank, so show it the observed object there while
        # preserving the directly observed background everywhere else.
        prompted = torch.where(hole.bool().expand_as(image), object_reference, image)
        pil_image = self._pil(_linear_to_srgb(prompted))
        pil_mask = self._pil(hole.float(), mask=True)
        return pil_image, pil_mask

    def _latent(self, image: Tensor, hole: Tensor, object_reference: Tensor) -> Tensor:
        pil_image, pil_mask = self._prior_input(image, hole, object_reference)
        digest = blake2b(digest_size=20)
        digest.update(str((pil_image.size, self.settings.seed)).encode())
        digest.update(pil_image.tobytes())
        digest.update(pil_mask.tobytes())
        key = digest.digest()
        cached = self._latent_cache.get(key)
        if cached is not None:
            self.image_cache_hits += 1
            self._latent_cache.move_to_end(key)
            return cached.to(self.execution_device)
        self.image_cache_misses += 1
        generator = torch.Generator(device=self.execution_device).manual_seed(self.settings.seed)
        with torch.no_grad():
            result = self.pipeline(
                prompt=self.settings.prompt,
                image=pil_image,
                mask_image=pil_mask,
                generator=generator,
                num_inference_steps=self.settings.inference_steps,
                guidance_scale=self.settings.guidance_scale,
                height=image.shape[-2],
                width=image.shape[-1],
                return_attn_map=False,
                output_type="pil" if self.settings.use_fused_output else "latent",
            )
            if self.settings.use_fused_output:
                from PIL import Image

                images = getattr(result, "images", None)
                if not isinstance(images, (list, tuple)) or not images or not isinstance(images[0], Image.Image):
                    raise RuntimeError("official ObjectClear did not return its fused PIL output")
                if not hasattr(self.pipeline, "image_processor"):
                    raise RuntimeError("official ObjectClear image processor is missing")
                vae = self.pipeline.vae
                pixels = self.pipeline.image_processor.preprocess(images[0])
                pixels = pixels.to(self.execution_device, dtype=vae.dtype)
                distribution = vae.encode(pixels).latent_dist
                encoded = distribution.mode()
                mean = getattr(vae.config, "latents_mean", None)
                std = getattr(vae.config, "latents_std", None)
                if mean is not None and std is not None:
                    mean_value = encoded.new_tensor(mean).view(1, 4, 1, 1)
                    std_value = encoded.new_tensor(std).view(1, 4, 1, 1)
                    latent = (encoded - mean_value) / std_value * vae.config.scaling_factor
                else:
                    latent = encoded * vae.config.scaling_factor
            else:
                latent = getattr(result, "images", None)
        if not isinstance(latent, Tensor) or latent.shape[:2] != (1, 4):
            raise RuntimeError("official ObjectClear did not produce four-channel latents")
        detached = latent[0].detach()
        if self.settings.cache_entries:
            self._latent_cache[key] = detached.cpu().contiguous()
            if len(self._latent_cache) > self.settings.cache_entries:
                self._latent_cache.popitem(last=False)
        return detached

    def _late_denoise_latent(
        self,
        image: Tensor,
        hole: Tensor,
        object_reference: Tensor,
        inverse: Tensor,
        weight: Tensor,
        flow: Tensor,
    ) -> Tensor:
        """Backpropagate through the last official denoising steps to flow.

        The released pipeline wraps inference in ``torch.no_grad``. Unwrap only
        this explicitly requested training path; all released model weights
        stay frozen. Early steps have no trainable input, so they build no
        autograd graph. A small optical adapter is inserted before the final
        K denoising steps, whose exact UNet Jacobian reaches PRISM's flow.
        """

        pil_image, pil_mask = self._prior_input(image, hole, object_reference)
        generator = torch.Generator(device=self.execution_device).manual_seed(self.settings.seed)
        inject_before = self.settings.late_denoise_steps
        injected = False

        def on_step_end(pipeline, index, timestep, values):
            del timestep
            nonlocal injected
            latent = values["latents"]
            total = int(pipeline.num_timesteps)
            if index == total - inject_before - 1:
                latent = self.flow_latent_adapter(
                    latent, image[None], inverse[None], weight[None],
                    flow[None], hole[None],
                )
                injected = True
            return {"latents": latent}

        needs_gradient = torch.is_grad_enabled() and (
            flow.requires_grad or inverse.requires_grad
            or any(parameter.requires_grad for parameter in self.flow_latent_adapter.parameters())
        )
        # The official inference wrapper is still appropriate for validation.
        # Unwrap its no-grad decorator only when the optical graph needs a
        # gradient through the last denoising steps.
        pipeline_call = (
            inspect.unwrap(type(self.pipeline).__call__)
            if needs_gradient else type(self.pipeline).__call__
        )
        self.image_cache_misses += 1  # The conditioned denoising path cannot be cached.
        result = pipeline_call(
            self.pipeline,
            prompt=self.settings.prompt,
            image=pil_image,
            mask_image=pil_mask,
            generator=generator,
            num_inference_steps=self.settings.inference_steps,
            guidance_scale=self.settings.guidance_scale,
            height=image.shape[-2],
            width=image.shape[-1],
            return_attn_map=False,
            output_type="latent",
            callback_on_step_end=on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        latent = getattr(result, "images", None)
        if not injected or not isinstance(latent, Tensor) or latent.shape[:2] != (1, 4):
            raise RuntimeError("ObjectClear did not run the flow-guided late denoising path")
        if needs_gradient and not latent.requires_grad:
            raise RuntimeError("ObjectClear late denoising detached the optical gradient")
        return latent

    def forward(
        self,
        evidence_background: Tensor,
        evidence_coverage: Tensor,
        true_hole: Tensor,
        *,
        flow_evidence: tuple[Tensor, Tensor, Tensor] | None = None,
        object_reference: Tensor | None = None,
    ) -> Tensor:
        del evidence_coverage
        if evidence_background.ndim != 4 or evidence_background.shape[1] != 3:
            raise ValueError("evidence_background must be [B,3,H,W]")
        batch, _, height, width = evidence_background.shape
        if true_hole.shape != (batch, 1, height, width):
            raise ValueError("true_hole must be [B,1,H,W]")
        if object_reference is None or object_reference.shape != evidence_background.shape:
            raise ValueError("ObjectClear requires an observed object-reference frame [B,3,H,W]")
        if flow_evidence is None:
            raise ValueError("ObjectClear joint mode requires inverse B, confidence, and flow")
        inverse, weight, flow = flow_evidence
        if inverse.shape != evidence_background.shape or weight.shape != true_hole.shape:
            raise ValueError("inverse evidence shape mismatch")
        if flow.shape != (batch, 2, height, width):
            raise ValueError("flow canvas shape mismatch")
        if not bool(true_hole.any()):
            return evidence_background

        side = self.settings.min_working_side
        scale = max(1.0, side / min(height, width))
        work_size = (math.ceil(height * scale / 16) * 16, math.ceil(width * scale / 16) * 16)
        known = (~true_hole.bool()).float()
        work_known = F.interpolate(known, size=work_size, mode="bilinear", align_corners=False)
        work_canvas = F.interpolate(
            evidence_background * known, size=work_size, mode="bilinear", align_corners=False
        ) / work_known.clamp_min(1e-6)
        work_hole = F.interpolate(true_hole.float(), size=work_size, mode="nearest")
        if self.settings.mask_dilation:
            radius = self.settings.mask_dilation
            work_hole = F.max_pool2d(work_hole, radius * 2 + 1, stride=1, padding=radius)
        work_inverse = F.interpolate(inverse, size=work_size, mode="bilinear", align_corners=False)
        work_reference = F.interpolate(
            object_reference, size=work_size, mode="bilinear", align_corners=False
        )
        work_weight = F.interpolate(weight, size=work_size, mode="bilinear", align_corners=False)
        flow_scale = flow.new_tensor((work_size[1] / width, work_size[0] / height)).view(1, 2, 1, 1)
        work_flow = F.interpolate(flow, size=work_size, mode="bilinear", align_corners=False) * flow_scale

        results: list[Tensor] = []
        vae = self.pipeline.vae
        for index in range(batch):
            if not bool(true_hole[index].any()):
                results.append(evidence_background[index])
                continue
            if self.settings.late_denoise_steps:
                adapted = self._late_denoise_latent(
                    work_canvas[index], work_hole[index], work_reference[index],
                    work_inverse[index], work_weight[index], work_flow[index],
                )
            else:
                latent = self._latent(
                    work_canvas[index], work_hole[index], work_reference[index]
                ).unsqueeze(0)
                adapted = self.flow_latent_adapter(
                    latent, work_canvas[index:index + 1], work_inverse[index:index + 1],
                    work_weight[index:index + 1], work_flow[index:index + 1],
                    work_hole[index:index + 1],
                )
            mean = getattr(vae.config, "latents_mean", None)
            std = getattr(vae.config, "latents_std", None)
            if mean is not None and std is not None:
                mean_value = adapted.new_tensor(mean).view(1, 4, 1, 1)
                std_value = adapted.new_tensor(std).view(1, 4, 1, 1)
                decoded_latent = adapted * std_value / vae.config.scaling_factor + mean_value
            else:
                decoded_latent = adapted / vae.config.scaling_factor
            decoded = vae.decode(
                decoded_latent.to(getattr(vae, "dtype", adapted.dtype)),
                return_dict=False,
            )[0]
            image = _srgb_to_linear((decoded.float() / 2 + 0.5).clamp(0, 1))
            image = F.interpolate(image, size=(height, width), mode="area")[0]
            results.append(image.to(evidence_background.dtype))
        return torch.stack(results, dim=0)
