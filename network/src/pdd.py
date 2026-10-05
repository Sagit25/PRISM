from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import PDDConfig
from .vendor import activate_vendored_sam2, sam2_setup_hint


activate_vendored_sam2()

try:
    from sam2.modeling.sam.transformer import TwoWayTransformer
    from sam2.modeling.sam2_utils import LayerNorm2d, MLP
except ImportError as exc:  # pragma: no cover - installation guard
    raise ImportError(sam2_setup_hint()) from exc


@dataclass
class PDDOutput:
    mask_logits: Tensor
    trimap_logits: Tensor
    decoded_features: Tensor


@dataclass
class _TokenDecode:
    segmentation_feature: Tensor
    trimap_feature: Tensor
    mask_token: Tensor
    trimap_tokens: Tensor


class _UpscaleBranch(nn.Module):
    """SAM2 mask-decoder upscaling with an independent feature branch.

    Official SAM2 projects the two high-resolution FPN levels in
    ``forward_image`` before ``_track_step`` returns them. Consequently these
    skip tensors already have ``width // 8`` and ``width // 4`` channels and
    must be added directly, exactly as SAM2's mask decoder does.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        if width % 8:
            raise ValueError("PDD width must be divisible by 8")
        self.deconv1 = nn.ConvTranspose2d(width, width // 4, 2, stride=2)
        self.norm1 = LayerNorm2d(width // 4)
        self.activation1 = nn.GELU()
        self.deconv2 = nn.ConvTranspose2d(width // 4, width // 8, 2, stride=2)
        self.activation2 = nn.GELU()

    @staticmethod
    def _add_skip(value: Tensor, skip: Tensor, *, level: str) -> Tensor:
        if skip.shape != value.shape:
            raise ValueError(
                f"SAM2 {level} skip must match the upscaled tensor; "
                f"got skip={tuple(skip.shape)} and value={tuple(value.shape)}"
            )
        return value + skip

    def forward(
        self,
        features: Tensor,
        high_res_features: Sequence[Tensor] | None,
    ) -> Tensor:
        first = self.deconv1(features)
        if high_res_features:
            if len(high_res_features) != 2:
                raise ValueError("PDD expects two SAM2 high-resolution feature levels")
            feature_s0, feature_s1 = high_res_features
            first = self._add_skip(first, feature_s1, level="s1")
        first = self.activation1(self.norm1(first))
        second = self.deconv2(first)
        if high_res_features:
            second = self._add_skip(second, high_res_features[0], level="s0")
        return self.activation2(second)

    def initialize_from_sam2(self, decoder: nn.Module) -> None:
        source = decoder.output_upscaling
        self.deconv1.load_state_dict(source[0].state_dict())
        self.norm1.load_state_dict(source[1].state_dict())
        self.deconv2.load_state_dict(source[3].state_dict())


class PromptableDualModeDecoder(nn.Module):
    """Paper-faithful clean-room implementation of MAM2's PDD.

    The decoder preserves SAM2's two-way transformer and mask decoding flow.
    Three trimap output tokens and a parallel upscaling branch are added beside
    the mask token. The predicted mask becomes a mask-augmentation feature and
    is fused with both output feature branches before a token-feature dot
    product produces the three trimap logits.

    Two auxiliary tokens retain the object-score/IoU token positions of the
    released SAM2.1 video model. This lets :meth:`initialize_from_sam2` copy the
    official transformer, tokens, mask hypernetwork and upscaler without
    changing the token sequence seen by the pretrained decoder.
    """

    ARCHITECTURE = "mam2-paper-pdd-v1"

    def __init__(self, config: PDDConfig | None = None) -> None:
        super().__init__()
        self.config = config or PDDConfig()
        width = self.config.width
        if self.config.feature_channels != width:
            raise ValueError(
                "paper PDD requires feature_channels == width to preserve the "
                "SAM2 two-way-transformer feature space"
            )
        if width % self.config.prompt_heads:
            raise ValueError("PDD width must be divisible by prompt_heads")
        if self.config.trimap_classes != 3:
            raise ValueError("MAM2 PDD requires exactly three trimap classes")

        self.transformer = TwoWayTransformer(
            depth=self.config.depth,
            embedding_dim=width,
            mlp_dim=self.config.transformer_mlp_dim,
            num_heads=self.config.prompt_heads,
        )
        self.object_score_token = nn.Embedding(1, width)
        self.iou_token = nn.Embedding(1, width)
        self.mask_output_token = nn.Embedding(1, width)
        self.trimap_output_tokens = nn.Embedding(self.config.trimap_classes, width)

        self.segmentation_upscale = _UpscaleBranch(width)
        self.trimap_upscale = _UpscaleBranch(width)
        output_width = width // 8
        self.mask_hypernetwork = MLP(width, width, output_width, 3)
        self.trimap_hypernetworks = nn.ModuleList(
            [MLP(width, width, output_width, 3) for _ in range(self.config.trimap_classes)]
        )

        self.mask_augmentation = nn.Sequential(
            nn.Conv2d(1, output_width, 3, padding=1),
            LayerNorm2d(output_width),
            nn.GELU(),
        )
        self.trimap_fusion = nn.Sequential(
            nn.Conv2d(output_width * 3, output_width, 3, padding=1),
            LayerNorm2d(output_width),
            nn.GELU(),
            nn.Conv2d(output_width, output_width, 3, padding=1),
            LayerNorm2d(output_width),
            nn.GELU(),
        )
        self._sam2_initialized = False

    @property
    def sam2_initialized(self) -> bool:
        return self._sam2_initialized

    def initialize_from_sam2(self, decoder: nn.Module) -> None:
        """Initialize the unchanged PDD mask path from official SAM2.1."""

        if getattr(decoder, "transformer_dim", None) != self.config.width:
            raise ValueError("SAM2 mask decoder width does not match PDD width")
        if getattr(decoder, "num_mask_tokens", 0) < 4:
            raise ValueError("SAM2 decoder must expose four mask tokens")
        self.transformer.load_state_dict(decoder.transformer.state_dict())
        with torch.no_grad():
            self.iou_token.weight.copy_(decoder.iou_token.weight)
            object_token = getattr(decoder, "obj_score_token", None)
            if object_token is None:
                self.object_score_token.weight.zero_()
            else:
                self.object_score_token.weight.copy_(object_token.weight)
            self.mask_output_token.weight.copy_(decoder.mask_tokens.weight[0:1])
            self.trimap_output_tokens.weight.copy_(decoder.mask_tokens.weight[1:4])

        self.segmentation_upscale.initialize_from_sam2(decoder)
        self.trimap_upscale.initialize_from_sam2(decoder)
        self.mask_hypernetwork.load_state_dict(
            decoder.output_hypernetworks_mlps[0].state_dict()
        )
        for target, source in zip(
            self.trimap_hypernetworks,
            decoder.output_hypernetworks_mlps[1:4],
            strict=True,
        ):
            target.load_state_dict(source.state_dict())
        self._sam2_initialized = True

    @staticmethod
    def _resize(value: Tensor, size: tuple[int, int]) -> Tensor:
        if value.shape[-2:] == size:
            return value
        return F.interpolate(value, size=size, mode="bilinear", align_corners=False)

    def _decode_tokens(
        self,
        features: Tensor,
        *,
        sparse_prompt_embeddings: Tensor | None,
        dense_prompt_embeddings: Tensor | None,
        image_pe: Tensor | None,
        high_res_features: Sequence[Tensor] | None,
    ) -> _TokenDecode:
        if features.ndim != 4 or features.shape[1] != self.config.feature_channels:
            raise ValueError(
                "features must have shape [B, PDDConfig.feature_channels, h, w]"
            )
        batch = features.shape[0]
        if sparse_prompt_embeddings is None:
            sparse_prompt_embeddings = features.new_zeros((batch, 0, self.config.width))
        if sparse_prompt_embeddings.ndim != 3:
            raise ValueError("sparse SAM2 prompts must have shape [B,N,C]")
        if sparse_prompt_embeddings.shape[0] != batch:
            raise ValueError("sparse prompt batch must match image features")

        tokens = torch.cat(
            (
                self.object_score_token.weight,
                self.iou_token.weight,
                self.mask_output_token.weight,
                self.trimap_output_tokens.weight,
            ),
            dim=0,
        ).unsqueeze(0).expand(batch, -1, -1)
        tokens = torch.cat((tokens, sparse_prompt_embeddings), dim=1)

        source = features
        if dense_prompt_embeddings is not None:
            if dense_prompt_embeddings.shape[1] != self.config.feature_channels:
                raise ValueError("dense SAM2 prompt embedding has an invalid channel count")
            source = source + self._resize(dense_prompt_embeddings, source.shape[-2:])
        if image_pe is None:
            image_pe = torch.zeros_like(source)
        else:
            image_pe = self._resize(image_pe, source.shape[-2:])
            if image_pe.shape[0] == 1 and batch > 1:
                image_pe = image_pe.expand(batch, -1, -1, -1)
            if image_pe.shape != source.shape:
                raise ValueError("SAM2 dense positional encoding does not match features")

        token_outputs, encoded = self.transformer(source, image_pe, tokens)
        encoded = encoded.transpose(1, 2).reshape_as(source)
        segmentation_feature = self.segmentation_upscale(encoded, high_res_features)
        trimap_feature = self.trimap_upscale(encoded, high_res_features)
        return _TokenDecode(
            segmentation_feature=segmentation_feature,
            trimap_feature=trimap_feature,
            mask_token=token_outputs[:, 2],
            trimap_tokens=token_outputs[:, 3 : 3 + self.config.trimap_classes],
        )

    @staticmethod
    def _token_dot(tokens: Tensor, features: Tensor) -> Tensor:
        batch, channels, height, width = features.shape
        if tokens.shape != (batch, channels):
            raise ValueError("hypernetwork token width does not match decoded features")
        return (tokens[:, None] @ features.reshape(batch, channels, height * width)).reshape(
            batch, 1, height, width
        )

    def _mask_from_decode(self, decoded: _TokenDecode) -> Tensor:
        return self._token_dot(
            self.mask_hypernetwork(decoded.mask_token),
            decoded.segmentation_feature,
        )

    def decode_mask(
        self,
        features: Tensor,
        seed_mask_logits: Tensor,
        *,
        sparse_prompt_embeddings: Tensor | None = None,
        dense_prompt_embeddings: Tensor | None = None,
        image_pe: Tensor | None = None,
        high_res_features: Sequence[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        decoded = self._decode_tokens(
            features,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            image_pe=image_pe,
            high_res_features=high_res_features,
        )
        mask = self._resize(
            self._mask_from_decode(decoded),
            seed_mask_logits.shape[-2:],
        )
        return mask, decoded.segmentation_feature

    def decode_trimap(
        self,
        features: Tensor,
        mask_logits: Tensor,
        *,
        sparse_prompt_embeddings: Tensor | None = None,
        dense_prompt_embeddings: Tensor | None = None,
        image_pe: Tensor | None = None,
        high_res_features: Sequence[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        decoded = self._decode_tokens(
            features,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            image_pe=image_pe,
            high_res_features=high_res_features,
        )
        mask = mask_logits.detach() if self.config.detach_mask_pseudo_prompt else mask_logits
        mask_feature = self.mask_augmentation(
            self._resize(mask.sigmoid(), decoded.trimap_feature.shape[-2:])
        )
        fused = self.trimap_fusion(
            torch.cat(
                (decoded.segmentation_feature, decoded.trimap_feature, mask_feature),
                dim=1,
            )
        )
        trimap_hyper = torch.stack(
            [
                network(decoded.trimap_tokens[:, index])
                for index, network in enumerate(self.trimap_hypernetworks)
            ],
            dim=1,
        )
        batch, channels, height, width = fused.shape
        trimap = (
            trimap_hyper @ fused.reshape(batch, channels, height * width)
        ).reshape(batch, self.config.trimap_classes, height, width)
        return self._resize(trimap, mask_logits.shape[-2:]), fused

    def forward(
        self,
        features: Tensor,
        seed_mask_logits: Tensor,
        *,
        sparse_prompt_embeddings: Tensor | None = None,
        dense_prompt_embeddings: Tensor | None = None,
        image_pe: Tensor | None = None,
        high_res_features: Sequence[Tensor] | None = None,
    ) -> PDDOutput:
        mask, decoded = self.decode_mask(
            features,
            seed_mask_logits,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            image_pe=image_pe,
            high_res_features=high_res_features,
        )
        trimap, _ = self.decode_trimap(
            features,
            mask,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            image_pe=image_pe,
            high_res_features=high_res_features,
        )
        return PDDOutput(mask_logits=mask, trimap_logits=trimap, decoded_features=decoded)
