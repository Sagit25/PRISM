from dataclasses import dataclass, field
from typing import Literal


# Official FBA Matting's penultimate full-resolution decoder activation.
FBA_PRE_FUSION_CHANNELS = 16


@dataclass
class PDDConfig:
    """Paper PDD configuration in the official SAM2 decoder feature space.

    ``feature_channels`` must match SAM2's ``hidden_dim`` (256 for the
    released SAM2/SAM2.1 Hiera checkpoints).
    """

    feature_channels: int = 256
    width: int = 256
    depth: int = 2
    prompt_heads: int = 8
    transformer_mlp_dim: int = 2048
    trimap_classes: int = 3
    detach_mask_pseudo_prompt: bool = False


@dataclass
class MAM2MatteConfig:
    """Lightweight trimap-guided alpha matter used by the MAM2 front end.

    The public MAM2 paper uses MEMatte for this role.  This in-tree module is
    a dependency-light, trainable implementation of the same RGB+trimap
    contract; an official MEMatte/MAM2 backend can replace it without changing
    the downstream PRISM interface.
    """

    width: int = 48
    depth: int = 3
    hard_trimap_at_inference: bool = True
    activation_checkpointing: bool = True
    full_activation_checkpointing: bool = True
    frame_chunk_size: int = 4
    backend: Literal["builtin", "external_mematte", "official_mam2", "official_sam2matting"] = "builtin"
    external_root: str | None = None
    external_config: str | None = None
    external_checkpoint: str | None = None
    external_max_tokens: int = 12000
    external_patch_decoder: bool = True
    external_train_decoder: bool = True
    external_train_backbone: bool = True
    # Modulate the official MEMatte backbone feature with a flow-aligned
    # background/render-error signal before its pretrained decoder.
    flow_feature_guidance: bool = False


@dataclass
class SAM2IntegrationConfig:
    """Controls the official-SAM2 to MAM2 upgrade."""

    pdd: PDDConfig = field(default_factory=PDDConfig)
    matte: MAM2MatteConfig = field(default_factory=MAM2MatteConfig)
    replace_sam_mask_for_memory: bool = True
    cache_inference_features_on_cpu: bool = False
    temporal_activation_checkpointing: bool = True
    temporal_checkpoint_chunk_size: int = 4
    temporal_detach_interval: int = 0
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_patterns: tuple[str, ...] = ("attn.qkv", "attn.proj")


@dataclass
class BackgroundConfig:
    """Configuration for one sequence-level counterfactual background asset."""

    completion_backbone: Literal["ffc", "dilated"] = "ffc"
    completion_width: int = 64
    completion_dilations: tuple[int, ...] = (1, 2, 4, 8)
    completion_down_blocks: int = 3
    completion_residual_blocks: int = 9
    completion_global_ratio: float = 0.5
    completion_max_channels: int = 512
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
    diffusion_mask_dilation: int = 0
    diffusion_min_working_side: int = 0
    diffusion_dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    object_threshold: float = 0.5
    # A predicted partly-transparent/object-colored pixel is not immutable
    # direct background evidence. FLUX-only runs use this tighter cutoff;
    # uncertain pixels are completed rather than preserved as measurements.
    flux_direct_observation_threshold: float = 0.1
    # At a fixed camera pixel, conflicting supposedly-visible frames may be
    # missed object pixels. Never freeze their majority color as direct B in
    # the FLUX-only variant; let the inpainting prior edit that location.
    flux_require_unanimous_direct_consensus: bool = True
    # One supposedly visible frame cannot distinguish true background from
    # a missed object pixel. Require independent temporal confirmation before
    # freezing that color in the FLUX-only background.
    flux_min_consistent_observations: int = 2
    coverage_scale: float = 1.0
    # A 1x1 kernel excludes only the predicted object pixel at 64x64.
    exclusion_dilation: int = 1
    robust_color_delta: float = 0.08
    variance_scale: float = 12.0
    minimum_observation_weight: float = 1e-3
    # A single very weak inverse splat is not a measured background pixel.
    inverse_min_reliable_weight: float = 0.05
    # Disagreeing inverse samples are uncertain context, not known pixels.
    inverse_variance_scale: float = 12.0
    use_temporal_completion: bool = True
    # Use observed/inverse evidence during fixed-point updates, then frozen
    # FLUX for non-direct pixels (or true holes when inverse refinement is
    # disabled). The learned completion module is bypassed.
    evidence_flux_only: bool = False
    inverse_min_transmittance: float = 0.05
    inverse_weight_scale: float = 1.0
    inverse_max_radiance: float = 2.0
    inverse_radiance_tolerance: float = 0.02
    preserve_direct_observations: bool = True
    use_inverse_evidence: bool = True
    # Inverse samples are predictions, not immutable observations. Let the
    # completion backend correct them while preserving direct observations.
    refine_inverse_evidence: bool = True
    # In FLUX-only mode, keep a small differentiable, confidence-gated inverse
    # correction around the frozen FLUX prior. This preserves a route from the
    # final render through B to the refractive operator without accepting an
    # arbitrary object-colored inverse sample as a background observation.
    diffusion_inverse_feedback_strength: float = 0.5
    diffusion_inverse_feedback_max_delta: float = 0.1
    diffusion_inverse_feedback_agreement_scale: float = 5.0
    # A flow-inverted sample can correct FLUX more confidently when the same
    # background neighborhood is also directly visible elsewhere in the clip.
    diffusion_inverse_overlap_kernel: int = 7
    diffusion_inverse_overlap_min_support: float = 0.02
    diffusion_inverse_validated_strength: float = 0.75
    diffusion_inverse_validated_max_delta: float = 0.25
    eps: float = 1e-6


@dataclass
class MatterConfig:
    """Configuration for the full reusable colored refractive operator.

    The default low-resolution head predicts the dataset's supervised single
    correspondence ``Phi=x+u`` directly. A deformable refractive kernel remains
    available as an explicit higher-resolution ablation (size >= 3).
    """

    feature_channels: int = 256
    width: int = 96
    encoder_depths: tuple[int, int, int] = (2, 3, 4)
    max_channels: int = 512
    temporal_blocks: int = 2
    flow_parameterization: Literal["resolution_fraction", "fixed_pixels"] = (
        "resolution_fraction"
    )
    max_refractive_flow_fraction: float = 0.5
    max_refractive_flow: float = 64.0
    # Optional zero-initialized, photometric-error-conditioned flow correction.
    # Disabled by default so existing v17/v4 checkpoints keep their contract.
    refine_refractive_flow: bool = False
    flow_refinement_width: int = 32
    flow_refinement_max_fraction: float = 0.0625
    # Optional full-resolution foreground correction conditioned on the
    # refracted background and the current physical reconstruction error.
    refine_straight_foreground: bool = False
    foreground_refinement_width: int = 32
    foreground_refinement_logit_scale: float = 2.0
    # Optional shared feature bridge: sample background features at x+u,
    # then jointly refine F, RGB transmission, and refractive flow. The
    # background itself is updated by the outer inverse-splat/FLUX loop.
    flow_guided_latent_bridge: bool = False
    # Supply the bridge with a pretrained FBA pre-fusion decoder feature, not
    # the conventional alpha-composited F/B outputs that mix refracted pixels.
    use_fba_foreground_prior: bool = False
    # Reuse the current pretrained SAM2/MEMatte visual features directly in
    # the flow-guided bridge instead of loading the older FBA matting model.
    use_mam2_foreground_prior: bool = False
    latent_bridge_width: int = 32
    latent_bridge_flow_fraction: float = 0.0625
    latent_bridge_foreground_logit_scale: float = 1.0
    latent_bridge_transmission_logit_scale: float = 1.0
    refractive_kernel_size: int = 1
    refractive_kernel_radius_fraction: float = 0.015625
    refractive_kernel_center_bias: float = 4.0
    hard_support_at_inference: bool = True
    # Gate all object-attached outputs by the predicted binary object mask.
    # The straight-through gradient keeps Stage 4 semantic refinement trainable.
    hard_object_support: bool = True
    # Keep the one-pixel antialiased silhouette reported by MAM2 while still
    # forbidding object-attached outputs on certain far-background pixels.
    antialiased_boundary_support: bool = True
    boundary_alpha_threshold: float = 0.01
    # Retained for loading older configurations; F is no longer obtained by
    # dividing G by alpha. Direct F supervision has its own GT-alpha gate.
    straight_foreground_eps: float = 1e-3
    straight_foreground_min_alpha: float = 1e-3
    residual_scale: float = 0.5
    # Legacy independent-sigmoid head initialization. The canonical RGB
    # head below instead starts from zero relative logits, since a common
    # bias cancels when its maximum channel is subtracted.
    neutral_transmission_bias: float = 4.0
    # The renderer defines T as tau / max_rgb(tau), so its largest RGB
    # component is one wherever transmission is identifiable.
    canonical_rgb_transmission: bool = True
    use_rgb_transmission: bool = True
    use_residual: bool = True
    refine_mam2_alpha: bool = True
    # The former 0.1 * trimap-unknown correction could not remove a filled
    # alpha interior when MAM2's matte was wrong there.
    alpha_refinement_scale: float = 0.5
    # Optional v18 geometry extension. Keeping this disabled preserves the
    # exact v17 PAM head shape and checkpoint contract.
    predict_geometry: bool = False
    geometry_min_depth: float = 1e-3


@dataclass
class LossWeights:
    mask: float = 1.0
    trimap: float = 1.0
    mam2_alpha: float = 2.0
    mam2_alpha_gradient: float = 0.5
    alpha: float = 2.0
    alpha_gradient: float = 0.5
    premultiplied_foreground: float = 1.0
    straight_foreground: float = 0.5
    premultiplied_foreground_gradient: float = 0.25
    straight_foreground_gradient: float = 0.1
    straight_foreground_laplacian: float = 0.0
    foreground_background_leakage: float = 1.0
    flow_background_leakage: float = 0.25
    residual_background_leakage: float = 0.25
    color_transmission: float = 1.0
    color_transmission_chroma: float = 0.0
    transmittance: float = 1.0
    temporal_visible_transmission: float = 0.25
    temporal_visible_joint_recomposition: float = 0.25
    residual: float = 0.75
    residual_sparsity: float = 0.05
    residual_excess: float = 0.0
    refractive_flow: float = 1.0
    source_coordinates: float = 0.25
    background: float = 1.0
    render: float = 2.0
    render_object: float = 2.0
    render_multiscale: float = 0.5
    render_oracle_background: float = 1.0
    temporal: float = 0.25
    temporal_holdout_render: float = 0.25
    flow_smoothness: float = 0.05
    flow_out_of_bounds: float = 0.1
    refractive_kernel_spread: float = 0.002
    refractive_kernel_smoothness: float = 0.01
    confidence: float = 0.1
    surface_normal: float = 1.0
    depth: float = 1.0
    depth_gradient: float = 0.25
    geometry_confidence: float = 0.1
    observed_background: float = 1.0
    inverse_background: float = 0.5
    background_inverse_refined: float = 2.0
    background_true_hole: float = 2.0
    background_frequency: float = 0.1
    operator_reuse: float = 0.5


@dataclass
class PipelineConfig:
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    matter: MatterConfig = field(default_factory=MatterConfig)
    # The final joint stage keeps both paths connected. Earlier stages can
    # still freeze modules explicitly through the training helpers.
    detach_semantics_for_background: bool = False
    detach_background_for_matter: bool = False
    joint_refinement_steps: int = 3
