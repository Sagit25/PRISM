"""One-step real-weight GPU smoke for the SAM2Matting/ObjectClear PRISM path.

This is a diagnostic, not a training job. It deliberately fails if the
ObjectClear prior is bypassed or either trainable branch receives no gradient.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam2matting-root", type=Path, required=True)
    parser.add_argument("--sam2matting-checkpoint", type=Path, required=True)
    parser.add_argument("--objectclear-root", type=Path, required=True)
    parser.add_argument("--objectclear-model", default="jixin0101/ObjectClear")
    parser.add_argument("--objectclear-revision", default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--late-denoise-steps", type=int, default=0)
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("ObjectClear AGF needs at least two denoising steps")
    if args.late_denoise_steps < 0 or args.late_denoise_steps >= args.steps:
        parser.error("late denoising steps must be in [0, steps)")
    if args.late_denoise_steps and args.cpu_offload:
        parser.error("late denoising backward cannot use CPU offload")
    os.environ["PRISM_SAM2_ROOT"] = str(args.sam2matting_root.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    import torch

    from refractive_mam2 import (
        BackgroundConfig, MaskedTemporalBackground, MatterConfig,
        PipelineConfig, RefractiveMAM2,
    )
    from refractive_mam2.objectclear_completion import (
        FlowGuidedObjectClearCompleter, ObjectClearSettings,
    )
    from refractive_mam2.official_sam2matting import OfficialSAM2MattingClipBackbone
    from refractive_mam2.training import configure_stage4

    if not torch.cuda.is_available():
        raise SystemExit("This real-weight smoke requires a CUDA GPU")
    device = torch.device("cuda")
    torch.manual_seed(19)
    predictor = OfficialSAM2MattingClipBackbone.from_pretrained(
        args.sam2matting_root, args.sam2matting_checkpoint, device=device,
    )
    completion = FlowGuidedObjectClearCompleter.from_pretrained(
        args.objectclear_model, source_root=args.objectclear_root,
        revision=args.objectclear_revision,
        settings=ObjectClearSettings(
            inference_steps=args.steps,
            use_fused_output=args.late_denoise_steps == 0,
            late_denoise_steps=args.late_denoise_steps,
        ),
        device=device, dtype="bfloat16", cpu_offload=args.cpu_offload,
    )
    config = PipelineConfig(
        background=BackgroundConfig(
            evidence_flux_only=True, completion_variant="diffusion",
            inverse_min_reliable_weight=0.0,
        ),
        matter=MatterConfig(feature_channels=256, width=16),
        joint_refinement_steps=1,
    )
    model = RefractiveMAM2(
        predictor, config,
        background_model=MaskedTemporalBackground(
            config.background, diffusion_completion=completion,
        ),
    ).to(device).train()
    trainable = configure_stage4(predictor, model)
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    torch.cuda.reset_peak_memory_stats(device)

    height = width = 64
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device), indexing="ij",
    )
    circle = ((xx - 32).square() + (yy - 32).square()) <= 12**2
    x = xx.float() / (width - 1)
    y = yy.float() / (height - 1)
    background = torch.stack((0.2 + 0.4 * x, 0.25 + 0.3 * y, 0.5 - 0.2 * x))
    foreground = torch.tensor((0.65, 0.7, 0.8), device=device)[:, None, None]
    frame = torch.where(circle[None], foreground, background).clamp(0, 1)
    frames = frame[None, None].repeat(1, 2, 1, 1, 1)
    mean = frame.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
    std = frame.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
    normalized = (frames - mean) / std
    semantic = predictor.forward_mam2_clip(
        normalized, first_frame_mask_inputs=circle[None, None].float(),
        original_size=(height, width),
    )
    optimizer.zero_grad(set_to_none=True)
    output = model.forward_from_backbone(frames, semantic)
    if completion.image_cache_misses != 1:
        raise RuntimeError("ObjectClear was bypassed; the synthetic clip had no editable hole")
    # A single shared render objective must reach both released-model
    # branches and the two-channel PRISM refractive-flow head.
    loss = (output.reconstructed_frames - frames).square().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("the joint render loss is not finite")
    loss.backward()

    def grad_sum(module: torch.nn.Module) -> float:
        return sum(
            float(parameter.grad.detach().abs().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )

    alpha_parameter = next(
        (
            parameter for parameter in predictor.main.alpha_pred3.parameters()
            if parameter.requires_grad and parameter.grad is not None
            and bool(parameter.grad.detach().abs().sum() > 0)
        ),
        None,
    )
    if alpha_parameter is None:
        raise RuntimeError("released SAM2Matting alpha head has no usable gradient")
    selected = {
        "sam2matting_alpha": alpha_parameter,
        "sam2matting_flow_adapter": predictor.mam2_matter.flow_feature_modulator[-1].weight,
        "objectclear_flow_adapter": completion.flow_latent_adapter.body[-1].weight,
        "prism_flow_head": model.matter.head.weight,
    }
    trainable_ids = {id(parameter) for parameter in trainable}
    if any(id(parameter) not in trainable_ids for parameter in selected.values()):
        raise RuntimeError("Stage 4 optimizer omitted a joint-trainable branch")
    report = {
        "sam2matting_alpha_trainable_parameters": sum(
            parameter.numel() for parameter in predictor.main.alpha_pred3.parameters()
            if parameter.requires_grad
        ),
        "objectclear_unet_trainable_parameters": sum(
            parameter.numel() for parameter in completion.pipeline.unet.parameters()
            if parameter.requires_grad
        ),
        "objectclear_adapter_trainable_parameters": sum(
            parameter.numel() for parameter in completion.flow_latent_adapter.parameters()
            if parameter.requires_grad
        ),
        "sam2matting_alpha_gradient": grad_sum(predictor.main.alpha_pred3),
        "sam2matting_flow_adapter_gradient": grad_sum(
            predictor.mam2_matter.flow_feature_modulator
        ),
        "objectclear_flow_adapter_gradient": grad_sum(completion.flow_latent_adapter),
        "prism_pam_gradient": grad_sum(model.matter),
        "prism_flow_head_gradient": float(
            model.matter.head.weight.grad[4:6].detach().abs().sum()
        ),
        "objectclear_prior_calls": completion.image_cache_misses,
        "objectclear_late_denoise_steps": args.late_denoise_steps,
        "editable_hole_pixels": int(output.background.true_hole.sum()),
        "loss": float(loss.detach()),
        "peak_gpu_reserved_gib": round(
            torch.cuda.max_memory_reserved(device) / 1024**3, 3
        ),
    }
    if report["objectclear_unet_trainable_parameters"] != 0:
        raise RuntimeError("the released ObjectClear UNet must stay frozen in this adapter experiment")
    if not all(value > 0 for key, value in report.items() if key.endswith("_gradient")):
        raise RuntimeError("the real-weight joint graph has a disconnected trainable branch")
    before = {name: parameter.detach().clone() for name, parameter in selected.items()}
    optimizer.step()
    for name, parameter in selected.items():
        after_value = parameter.detach()
        before_value = before[name]
        if name == "prism_flow_head":
            after_value = after_value[4:6]
            before_value = before_value[4:6]
        report[f"{name}_update"] = float((after_value - before_value).abs().sum())
    print(json.dumps(report, sort_keys=True), flush=True)
    if not all(value > 0 for key, value in report.items() if key.endswith("_update")):
        raise RuntimeError("a joint-trainable branch did not update in the same optimizer step")


if __name__ == "__main__":
    main()
