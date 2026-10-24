"""Optional masked background completion backends for PRISM."""

from __future__ import annotations

import inspect
import math
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any

import torch
import torch.nn.functional as F
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
    mask_dilation: int = 0
    # FLUX Fill's native examples are much larger than PRISM's 64px frames.
    # Zero keeps legacy behavior; a positive value runs inpainting on an
    # upscaled canvas and downsamples the linear-RGB result afterward.
    min_working_side: int = 0
    # Cache only exact, quantized FLUX inputs. A frozen model with a fixed seed
    # otherwise repeats the same expensive generation every training epoch.
    image_cache_entries: int = 2048
    # Keep the FLUX transformer frozen, but jointly train a flow-conditioned
    # residual in its denoised latent before the official VAE decoder.
    flow_latent_guidance: bool = False
    flow_latent_max_delta: float = 0.5

    def __post_init__(self) -> None:
        if self.inference_steps < 1:
            raise ValueError("diffusion inference_steps must be positive")
        if self.guidance_scale < 0:
            raise ValueError("diffusion guidance_scale must be non-negative")
        if self.mask_dilation < 0:
            raise ValueError("diffusion mask_dilation must be non-negative")
        if self.min_working_side < 0:
            raise ValueError("diffusion min_working_side must be non-negative")
        if self.image_cache_entries < 0:
            raise ValueError("diffusion image_cache_entries must be non-negative")
        if self.flow_latent_max_delta <= 0:
            raise ValueError("flow_latent_max_delta must be positive")


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


def _is_flux_fill_model(model: str) -> bool:
    """Return whether ``model`` requires Diffusers' dedicated FLUX Fill loader."""

    normalized = model.rstrip("/").lower()
    return "flux.1-fill" in normalized or "flux1-fill" in normalized


def _pipeline_accepts_keyword(pipeline: Any, keyword: str) -> bool:
    """Inspect a Diffusers call signature without assuming one pipeline family."""

    try:
        parameters = inspect.signature(pipeline.__call__).parameters.values()
    except (TypeError, ValueError):
        # Some third-party callables do not expose a Python signature. Keep the
        # legacy behavior for those pipelines rather than rejecting valid args.
        return True
    return any(
        parameter.name == keyword
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


class _FlowGuidedFluxLatentAdapter(nn.Module):
    """Trainable optical correction *inside* FLUX's decoded latent path."""

    def __init__(self, max_delta: float) -> None:
        super().__init__()
        self.max_delta = max_delta
        # FLUX Fill's official VAE has 16 latent channels. Conditioning:
        # direct canvas RGB, inverse RGB, inverse coverage, splatted flow,
        # and editable mask. The last layer is zero initialized.
        self.body = nn.Sequential(
            nn.Conv2d(26, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 16, 3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(
        self,
        latents: Tensor,
        evidence: Tensor,
        inverse_background: Tensor,
        inverse_weight: Tensor,
        flow_canvas: Tensor,
        editable: Tensor,
    ) -> Tensor:
        if latents.ndim != 4 or latents.shape[1] != 16:
            raise ValueError("FLUX Fill latent adapter requires 16-channel VAE latents")
        batch, _, height, width = latents.shape
        if evidence.shape[0] != batch or evidence.shape[1] != 3:
            raise ValueError("evidence must have shape [B,3,H,W]")
        if inverse_background.shape != evidence.shape:
            raise ValueError("inverse_background must match evidence")
        if inverse_weight.shape != (batch, 1, *evidence.shape[-2:]):
            raise ValueError("inverse_weight must have shape [B,1,H,W]")
        if flow_canvas.shape != (batch, 2, *evidence.shape[-2:]):
            raise ValueError("flow_canvas must have shape [B,2,H,W]")
        if editable.shape != inverse_weight.shape:
            raise ValueError("editable must match inverse_weight")
        size = (height, width)
        resize = lambda value: F.interpolate(
            value.float(), size=size, mode="bilinear", align_corners=False
        )
        flow_scale = flow_canvas.new_tensor(
            (max(evidence.shape[-1] - 1, 1), max(evidence.shape[-2] - 1, 1))
        ).view(1, 2, 1, 1)
        context = torch.cat(
            (
                latents.float(),
                resize(evidence),
                resize(inverse_background),
                resize(inverse_weight.clamp(0, 1)),
                resize(flow_canvas / flow_scale),
                resize(editable),
            ),
            dim=1,
        )
        correction = torch.tanh(self.body(context)) * self.max_delta
        correction = correction * F.interpolate(editable.float(), size=size, mode="nearest")
        return latents + correction.to(latents.dtype)


class FrozenDiffusionBackgroundCompleter(nn.Module):
    """Use a frozen Diffusers inpainting pipeline for editable background pixels.

    The wrapper converts PRISM's linear-RGB evidence canvas to sRGB for the
    image diffusion model, then converts the generated result back to linear
    RGB. The caller remains responsible for exact direct-evidence preservation
    by compositing the result only where the editable mask is one.

    ``pipeline`` is intentionally not registered as a trainable submodule.
    Diffusion weights are external assets and are not duplicated in PRISM's
    compact format-v8 checkpoint. A pre-trained LoRA may be loaded when the
    backend is constructed with :meth:`from_pretrained`.
    """

    def __init__(
        self,
        pipeline: Any,
        *,
        settings: DiffusionCompletionSettings | None = None,
        execution_device: str | torch.device = "cpu",
        cache_prompt_embeddings: bool = False,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_pipeline", pipeline)
        self.settings = settings or DiffusionCompletionSettings()
        self.execution_device = torch.device(execution_device)
        self.cache_prompt_embeddings = cache_prompt_embeddings
        self._prompt_embeddings: tuple[Tensor, Tensor] | None = None
        self._image_cache: OrderedDict[bytes, Tensor] = OrderedDict()
        self._latent_cache: OrderedDict[bytes, Tensor] = OrderedDict()
        self.image_cache_hits = 0
        self.image_cache_misses = 0
        self.flow_latent_guidance = self.settings.flow_latent_guidance
        self.flow_latent_adapter = None
        if self.flow_latent_guidance:
            if not all(hasattr(pipeline, name) for name in ("vae", "_unpack_latents", "vae_scale_factor")):
                raise TypeError("flow latent guidance requires the official FLUX Fill latent/VAE API")
            pipeline.vae.requires_grad_(False)
            self.flow_latent_adapter = _FlowGuidedFluxLatentAdapter(
                self.settings.flow_latent_max_delta
            )

    @staticmethod
    def _image_cache_key(image: Any, mask: Any, seed: int) -> bytes:
        """Hash exactly what the image model receives, including its RNG seed."""

        digest = blake2b(digest_size=20)
        digest.update(str((image.mode, image.size, mask.mode, mask.size, seed)).encode())
        digest.update(image.tobytes())
        digest.update(mask.tobytes())
        return digest.digest()

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
        cpu_offload: bool = True,
    ) -> "FrozenDiffusionBackgroundCompleter":
        try:
            import diffusers
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "PRISM-Diffusion requires refractive-mam2[diffusion]"
            ) from exc

        if _is_flux_fill_model(model):
            pipeline_class = getattr(diffusers, "FluxFillPipeline", None)
            if pipeline_class is None:
                raise ImportError(
                    "FLUX.1-Fill requires a Diffusers release that provides "
                    "FluxFillPipeline"
                )
        else:
            pipeline_class = diffusers.AutoPipelineForInpainting

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
        pipeline = pipeline_class.from_pretrained(
            model,
            torch_dtype=dtype_by_name[dtype],
            revision=revision,
        )
        if adapter is not None:
            if not hasattr(pipeline, "load_lora_weights"):
                raise TypeError("selected diffusion pipeline does not support LoRA")
            pipeline.load_lora_weights(adapter)
        pipeline.set_progress_bar_config(disable=True)
        if (
            cpu_offload
            and execution_device.type == "cuda"
            and hasattr(pipeline, "enable_model_cpu_offload")
        ):
            pipeline.enable_model_cpu_offload(device=str(execution_device))
        else:
            pipeline.to(execution_device)
        return cls(
            pipeline,
            settings=settings,
            execution_device=execution_device,
            cache_prompt_embeddings=_is_flux_fill_model(model),
        )

    @property
    def pipeline(self) -> Any:
        return object.__getattribute__(self, "_pipeline")

    @property
    def image_cache_size(self) -> int:
        return len(self._latent_cache if self.flow_latent_guidance else self._image_cache)

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

    def _generator(self) -> torch.Generator:
        """Keep a clip's frozen prior independent of its batch position.

        The prior input can be reused across epochs, but a shuffled batch
        index is not a stable image identity. Using seed+index made the same
        scene alternate between different completed backgrounds during
        training, adding avoidable noise to F/T/flow supervision.
        """

        generator_device = (
            self.execution_device
            if self.execution_device.type == "cuda"
            else torch.device("cpu")
        )
        return torch.Generator(device=generator_device).manual_seed(
            self.settings.seed
        )

    def _fixed_prompt_embeddings(self, batch_size: int) -> tuple[Tensor, Tensor] | None:
        """Encode FLUX Fill's unchanged prompt once, then broadcast per image."""

        if not self.cache_prompt_embeddings:
            return None
        encode = getattr(self.pipeline, "encode_prompt", None)
        if encode is None or not all(
            _pipeline_accepts_keyword(self.pipeline, name)
            for name in ("prompt_embeds", "pooled_prompt_embeds")
        ):
            return None
        if self._prompt_embeddings is None:
            result = encode(
                prompt=self.settings.prompt,
                device=self.execution_device,
                num_images_per_prompt=1,
            )
            if not isinstance(result, (tuple, list)) or len(result) < 2:
                raise RuntimeError("FLUX Fill returned invalid prompt embeddings")
            prompt, pooled = result[:2]
            if prompt.shape[0] != 1 or pooled.shape[0] != 1:
                raise RuntimeError("FLUX Fill prompt encoding must have batch size one")
            self._prompt_embeddings = (prompt.detach(), pooled.detach())
        prompt, pooled = self._prompt_embeddings
        return prompt.expand(batch_size, *prompt.shape[1:]), pooled.expand(
            batch_size, *pooled.shape[1:]
        )

    def forward(
        self,
        evidence_background: Tensor,
        evidence_coverage: Tensor,
        true_hole: Tensor,
        *,
        flow_evidence: tuple[Tensor, Tensor, Tensor] | None = None,
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

        outputs: list[Tensor] = [
            evidence_background[index]
            for index in range(evidence_background.shape[0])
        ]
        original_height, original_width = evidence_background.shape[-2:]
        if self.settings.min_working_side > min(original_height, original_width):
            scale = self.settings.min_working_side / min(
                original_height, original_width
            )
            working_size = (
                math.ceil(original_height * scale / 16) * 16,
                math.ceil(original_width * scale / 16) * 16,
            )
            # The hole canvas is zero-filled. Plain bilinear enlargement would
            # mix those zeros into pixels *outside* the enlarged hole mask,
            # making FLUX preserve a dark/object-shaped rim as known context.
            # Normalize by the interpolated known-pixel support instead.
            known = (~true_hole).float()
            working_known = torch.nn.functional.interpolate(
                known,
                size=working_size,
                mode="bilinear",
                align_corners=False,
            )
            working_evidence = torch.nn.functional.interpolate(
                evidence_background.float() * known,
                size=working_size,
                mode="bilinear",
                align_corners=False,
            ) / working_known.clamp_min(1e-6)
            working_evidence = working_evidence.to(evidence_background.dtype)
            working_hole = torch.nn.functional.interpolate(
                true_hole.float(), size=working_size, mode="nearest"
            ).bool()
        else:
            working_size = (original_height, original_width)
            working_evidence = evidence_background
            working_hole = true_hole
        srgb_evidence = _linear_to_srgb(working_evidence)
        if self.settings.mask_dilation > 0:
            kernel = 2 * self.settings.mask_dilation + 1
            generation_hole = torch.nn.functional.max_pool2d(
                working_hole.float(),
                kernel,
                stride=1,
                padding=self.settings.mask_dilation,
            ) > 0
        else:
            generation_hole = working_hole
        active = [
            index
            for index in range(evidence_background.shape[0])
            if bool(true_hole[index].any())
        ]
        if not active:
            return torch.stack(outputs, dim=0)

        if self.flow_latent_guidance:
            if flow_evidence is None:
                raise ValueError("flow-guided FLUX requires inverse B, weight and flow canvas")
            return self._forward_flow_latent(
                outputs,
                active,
                srgb_evidence,
                working_evidence,
                generation_hole,
                working_size,
                (original_height, original_width),
                evidence_background,
                flow_evidence,
            )

        missing: list[int] = []
        images = []
        masks = []
        cache_keys: list[bytes] = []
        for index in active:
            image = self._pil_image(srgb_evidence[index])
            mask = self._pil_mask(generation_hole[index].float())
            key = self._image_cache_key(image, mask, self.settings.seed)
            cached = (
                self._image_cache.get(key)
                if self.settings.image_cache_entries else None
            )
            if cached is not None:
                self.image_cache_hits += 1
                self._image_cache.move_to_end(key)
                outputs[index] = cached.to(
                    device=evidence_background.device,
                    dtype=evidence_background.dtype,
                ).clone()
            else:
                self.image_cache_misses += 1
                missing.append(index)
                images.append(image)
                masks.append(mask)
                cache_keys.append(key)
        if not missing:
            return torch.stack(outputs, dim=0)

        generators = [self._generator() for _ in missing]
        batched = len(missing) > 1
        call_kwargs = {
            "image": images if batched else images[0],
            "mask_image": masks if batched else masks[0],
            "num_inference_steps": self.settings.inference_steps,
            "guidance_scale": self.settings.guidance_scale,
            "generator": generators if batched else generators[0],
        }
        prompt_embeddings = self._fixed_prompt_embeddings(len(missing))
        if prompt_embeddings is None:
            call_kwargs["prompt"] = (
                [self.settings.prompt] * len(missing)
                if batched else self.settings.prompt
            )
        else:
            call_kwargs["prompt_embeds"], call_kwargs["pooled_prompt_embeds"] = (
                prompt_embeddings
            )
        height, width = working_size
        if _pipeline_accepts_keyword(self.pipeline, "height"):
            call_kwargs["height"] = height
        if _pipeline_accepts_keyword(self.pipeline, "width"):
            call_kwargs["width"] = width
        if _pipeline_accepts_keyword(self.pipeline, "negative_prompt"):
            call_kwargs["negative_prompt"] = (
                [self.settings.negative_prompt] * len(active)
                if batched
                else self.settings.negative_prompt
            )
        with torch.no_grad():
            result = self.pipeline(**call_kwargs)
        result_images = getattr(result, "images", None)
        if not result_images or len(result_images) != len(missing):
            raise RuntimeError(
                "diffusion inpainting pipeline returned an invalid image batch"
            )
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("PRISM-Diffusion requires numpy") from exc
        for batch_index, generated, cache_key in zip(
            missing, result_images, cache_keys, strict=True
        ):
            image = torch.from_numpy(
                np.asarray(generated.convert("RGB"), dtype=np.float32).copy()
            ).permute(2, 0, 1) / 255.0
            if image.shape[-2:] != working_size:
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    size=working_size,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            linear_image = _srgb_to_linear(image)
            if working_size != (original_height, original_width):
                linear_image = torch.nn.functional.interpolate(
                    linear_image.unsqueeze(0),
                    size=(original_height, original_width),
                    mode="area",
                ).squeeze(0)
            outputs[batch_index] = linear_image.to(
                device=evidence_background.device,
                dtype=evidence_background.dtype,
            )
            if self.settings.image_cache_entries:
                self._image_cache[cache_key] = linear_image.cpu().contiguous()
                self._image_cache.move_to_end(cache_key)
                if len(self._image_cache) > self.settings.image_cache_entries:
                    self._image_cache.popitem(last=False)
        return torch.stack(outputs, dim=0)

    def _forward_flow_latent(
        self,
        outputs: list[Tensor],
        active: list[int],
        srgb_evidence: Tensor,
        working_evidence: Tensor,
        generation_hole: Tensor,
        working_size: tuple[int, int],
        original_size: tuple[int, int],
        original_evidence: Tensor,
        flow_evidence: tuple[Tensor, Tensor, Tensor],
    ) -> Tensor:
        """Cache frozen FLUX samples, then decode a trainable optical latent."""
        if self.flow_latent_adapter is None:
            raise RuntimeError("FLUX flow latent adapter was not initialized")
        inverse_background, inverse_weight, flow_canvas = flow_evidence
        if inverse_background.shape != original_evidence.shape:
            raise ValueError("inverse background must match original evidence")
        if inverse_weight.shape != (original_evidence.shape[0], 1, *original_size):
            raise ValueError("inverse weight must match original evidence")
        if flow_canvas.shape != (original_evidence.shape[0], 2, *original_size):
            raise ValueError("flow canvas must match original evidence")

        cached_latents: dict[int, Tensor] = {}
        missing: list[int] = []
        images: list[Any] = []
        masks: list[Any] = []
        keys: list[bytes] = []
        for index in active:
            image = self._pil_image(srgb_evidence[index])
            mask = self._pil_mask(generation_hole[index].float())
            key = self._image_cache_key(image, mask, self.settings.seed)
            cached = self._latent_cache.get(key) if self.settings.image_cache_entries else None
            if cached is None:
                self.image_cache_misses += 1
                missing.append(index)
                images.append(image)
                masks.append(mask)
                keys.append(key)
            else:
                self.image_cache_hits += 1
                self._latent_cache.move_to_end(key)
                cached_latents[index] = cached.to(self.execution_device)

        if missing:
            batched = len(missing) > 1
            kwargs: dict[str, Any] = {
                "image": images if batched else images[0],
                "mask_image": masks if batched else masks[0],
                "num_inference_steps": self.settings.inference_steps,
                "guidance_scale": self.settings.guidance_scale,
                "generator": [self._generator() for _ in missing] if batched else self._generator(),
                "output_type": "latent",
            }
            with torch.no_grad():
                prompt_embeddings = self._fixed_prompt_embeddings(len(missing))
                if prompt_embeddings is None:
                    kwargs["prompt"] = [self.settings.prompt] * len(missing) if batched else self.settings.prompt
                else:
                    kwargs["prompt_embeds"], kwargs["pooled_prompt_embeds"] = prompt_embeddings
                if _pipeline_accepts_keyword(self.pipeline, "height"):
                    kwargs["height"] = working_size[0]
                if _pipeline_accepts_keyword(self.pipeline, "width"):
                    kwargs["width"] = working_size[1]
                if _pipeline_accepts_keyword(self.pipeline, "negative_prompt"):
                    kwargs["negative_prompt"] = [self.settings.negative_prompt] * len(missing) if batched else self.settings.negative_prompt
                result = self.pipeline(**kwargs)
            latents = getattr(result, "images", None)
            if not isinstance(latents, Tensor) or latents.ndim != 3 or latents.shape[0] != len(missing):
                raise RuntimeError("FLUX Fill did not return packed latent tensors")
            for index, key, latent in zip(missing, keys, latents.detach(), strict=True):
                cached_latents[index] = latent
                if self.settings.image_cache_entries:
                    self._latent_cache[key] = latent.cpu().contiguous()
                    self._latent_cache.move_to_end(key)
                    if len(self._latent_cache) > self.settings.image_cache_entries:
                        self._latent_cache.popitem(last=False)

        packed = torch.stack([cached_latents[index] for index in active], dim=0)
        height, width = working_size
        unpacked = self.pipeline._unpack_latents(
            packed, height, width, self.pipeline.vae_scale_factor
        )
        if unpacked.ndim != 4 or unpacked.shape[1] != 16:
            raise RuntimeError("FLUX Fill returned unexpected VAE latent channels")
        work_inverse = F.interpolate(inverse_background[active], size=working_size, mode="bilinear", align_corners=False)
        work_weight = F.interpolate(inverse_weight[active], size=working_size, mode="bilinear", align_corners=False)
        # Flow is in source-image pixel units. Scale it to the working canvas
        # before the latent adapter normalizes by that canvas's dimensions.
        scale = flow_canvas.new_tensor(
            (width / original_size[1], height / original_size[0])
        ).view(1, 2, 1, 1)
        work_flow = F.interpolate(flow_canvas[active], size=working_size, mode="bilinear", align_corners=False) * scale
        adapted = self.flow_latent_adapter(
            unpacked,
            working_evidence[active],
            work_inverse,
            work_weight,
            work_flow,
            generation_hole[active].float(),
        )
        vae = self.pipeline.vae
        latent_input = adapted / vae.config.scaling_factor + vae.config.shift_factor
        latent_input = latent_input.to(dtype=getattr(vae, "dtype", latent_input.dtype))
        decoded = vae.decode(latent_input, return_dict=False)[0]
        srgb_images = (decoded.float() / 2.0 + 0.5).clamp(0.0, 1.0)
        if srgb_images.shape[-2:] != working_size:
            srgb_images = F.interpolate(srgb_images, size=working_size, mode="bilinear", align_corners=False)
        linear_images = _srgb_to_linear(srgb_images)
        if working_size != original_size:
            linear_images = F.interpolate(linear_images, size=original_size, mode="area")
        for position, index in enumerate(active):
            outputs[index] = linear_images[position].to(
                device=original_evidence.device, dtype=original_evidence.dtype
            )
        return torch.stack(outputs, dim=0)
