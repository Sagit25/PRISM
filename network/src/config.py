from dataclasses import dataclass, field
from typing import Literal


@dataclass
class PDDConfig:
    """Promptable Dual-mode Decoder configuration.

    ``feature_channels`` must match SAM2's ``hidden_dim`` (256 for the
    released SAM2/SAM2.1 Hiera checkpoints).
    """

    feature_channels: int = 256
    width: int = 256
    depth: int = 3
    prompt_heads: int = 8
    refinement_width: int = 64
    mask_delta_scale: float = 1.0
    detach_mask_pseudo_prompt: bool = False


@dataclass
class SAM2IntegrationConfig:
    """Controls the official-SAM2 to MAM2 upgrade."""

    pdd: PDDConfig = field(default_factory=PDDConfig)
    replace_sam_mask_for_memory: bool = True
    cache_inference_features_on_cpu: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_patterns: tuple[str, ...] = ("attn.qkv", "attn.proj")


@dataclass
class BackgroundConfig:
    """Configuration for one sequence-level counterfactual background asset."""

    completion_width: int = 48
    completion_dilations: tuple[int, ...] = (1, 2, 4, 8)
    completion_variant: Literal["base", "diffusion"] = "base"
    diffusion_model: str | None = None
    diffusion_adapter: str | None = None
    diffusion_revision: str | None = None
    diffusion_prompt: str = (
        "a clean static background, photorealistic, continuous texture, "
        "no foreground object"
    )
    diffusion_negative_prompt: str = (
        "transparent object, foreground object, duplicate object, distortion, "
        "text, watermark"
    )
    diffusion_inference_steps: int = 25
    diffusion_guidance_scale: float = 7.5
    diffusion_seed: int = 0
    diffusion_mask_dilation: int = 8
    diffusion_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    object_threshold: float = 0.5
    coverage_scale: float = 1.0
    exclusion_dilation: int = 7
    robust_color_delta: float = 0.08
    variance_scale: float = 12.0
    minimum_observation_weight: float = 1e-3
    use_temporal_completion: bool = True
    inverse_min_transmittance: float = 0.05
    inverse_weight_scale: float = 1.0
    inverse_max_radiance: float = 2.0
    preserve_direct_observations: bool = True
    use_inverse_evidence: bool = True
    eps: float = 1e-6


@dataclass
class MatterConfig:
    """Configuration for the reusable colored refractive operator."""

    feature_channels: int = 256
    width: int = 64
    flow_parameterization: Literal["resolution_fraction", "fixed_pixels"] = (
        "resolution_fraction"
    )
    max_refractive_flow_fraction: float = 0.25
    max_refractive_flow: float = 64.0
    hard_support_at_inference: bool = True
    straight_foreground_eps: float = 1e-3
    straight_foreground_min_alpha: float = 1e-3
    residual_scale: float = 0.25
    neutral_transmission_bias: float = 4.0
    use_rgb_transmission: bool = True
    use_residual: bool = True


@dataclass
class LossWeights:
    mask: float = 1.0
    trimap: float = 1.0
    alpha: float = 2.0
    alpha_gradient: float = 0.5
    premultiplied_foreground: float = 1.0
    color_transmission: float = 1.0
    transmittance: float = 1.0
    residual: float = 0.5
    residual_sparsity: float = 0.02
    refractive_flow: float = 1.0
    source_coordinates: float = 0.25
    background: float = 1.0
    render: float = 2.0
    render_multiscale: float = 0.5
    temporal: float = 0.25
    flow_smoothness: float = 0.05
    flow_out_of_bounds: float = 0.1
    confidence: float = 0.1
    observed_background: float = 1.0
    inverse_background: float = 0.5
    background_true_hole: float = 1.0
    operator_reuse: float = 0.5


@dataclass
class PipelineConfig:
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    matter: MatterConfig = field(default_factory=MatterConfig)
    # The final joint stage keeps both paths connected. Earlier stages can
    # still freeze modules explicitly through the training helpers.
    detach_semantics_for_background: bool = False
    detach_background_for_matter: bool = False
    joint_refinement_steps: int = 2
