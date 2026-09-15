"""Physics-aware extensions for MAM2-style video matting."""

from .background import (
    BackgroundOutput,
    DirectBackgroundEvidence,
    MaskedTemporalBackground,
    RefractiveBackgroundEvidence,
)
from .config import (
    BackgroundConfig,
    LossWeights,
    MatterConfig,
    MAM2MatteConfig,
    PDDConfig,
    PipelineConfig,
    SAM2IntegrationConfig,
)
from .completion import (
    DiffusionCompletionSettings,
    FrozenDiffusionBackgroundCompleter,
)
from .ffc_completion import GLaMaCompletionNet
from .color import linear_to_srgb, srgb_to_linear
from .dataset import (
    PairedBackgroundBatchSampler,
    RCTransBatch,
    RCTransPRISMDataset,
    build_paired_prism_dataloader,
    prism_collate,
)
from .inverse import inverse_refractive_splat
from .losses import (
    RefractiveGroundTruth,
    RefractiveLoss,
    reusable_operator_consistency,
    reusable_operator_consistency_in_batch,
    source_coordinates_from_flow,
)
from .matter import PhysicsAwareMatter, PhysicsMatterOutput
from .mam2_matte import ExternalMEMatteMatter, MAM2TrimapMatter, build_mam2_matter
from .mss import MSSOutput, MemorySeparableSiamese
from .pdd import PDDOutput, PromptableDualModeDecoder
from .pipeline import RefractiveMAM2, RefractiveMAM2Output
from .renderer import recompose, warp_background
from .semantic_dataset import ManifestSemanticDataset, SemanticBatch, semantic_collate
from .runner import (
    SAM2RefractiveRunner,
    build_physics_pipeline_for_sam2,
    load_refractive_checkpoint,
    propagate_mam2_backbone,
    save_refractive_checkpoint,
)
from .sam2_integration import (
    MAM2FrameOutput,
    MAM2VideoPredictor,
    build_mam2_video_predictor,
    mark_only_mam2_trainable,
)
from .trimap import BG, FG, UNKNOWN, build_transparency_trimap
from .types import MAM2Backbone, MAM2BackboneOutput

__all__ = [
    "BG",
    "FG",
    "UNKNOWN",
    "BackgroundConfig",
    "BackgroundOutput",
    "DirectBackgroundEvidence",
    "DiffusionCompletionSettings",
    "LossWeights",
    "FrozenDiffusionBackgroundCompleter",
    "GLaMaCompletionNet",
    "ExternalMEMatteMatter",
    "MAM2FrameOutput",
    "MAM2MatteConfig",
    "MAM2TrimapMatter",
    "MAM2Backbone",
    "MAM2BackboneOutput",
    "MAM2VideoPredictor",
    "MSSOutput",
    "MaskedTemporalBackground",
    "MatterConfig",
    "MemorySeparableSiamese",
    "PDDConfig",
    "PDDOutput",
    "PairedBackgroundBatchSampler",
    "PhysicsAwareMatter",
    "PhysicsMatterOutput",
    "PipelineConfig",
    "PromptableDualModeDecoder",
    "RefractiveGroundTruth",
    "RefractiveBackgroundEvidence",
    "RefractiveLoss",
    "RefractiveMAM2",
    "RefractiveMAM2Output",
    "RCTransBatch",
    "RCTransPRISMDataset",
    "SAM2IntegrationConfig",
    "SAM2RefractiveRunner",
    "build_mam2_video_predictor",
    "build_mam2_matter",
    "build_physics_pipeline_for_sam2",
    "build_paired_prism_dataloader",
    "build_transparency_trimap",
    "inverse_refractive_splat",
    "linear_to_srgb",
    "recompose",
    "load_refractive_checkpoint",
    "mark_only_mam2_trainable",
    "propagate_mam2_backbone",
    "save_refractive_checkpoint",
    "reusable_operator_consistency",
    "reusable_operator_consistency_in_batch",
    "source_coordinates_from_flow",
    "srgb_to_linear",
    "prism_collate",
    "warp_background",
    "ManifestSemanticDataset",
    "SemanticBatch",
    "semantic_collate",
]
