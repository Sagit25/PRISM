import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from refractive_mam2.compare import compare_metric_files
from refractive_mam2.completion import (
    DiffusionCompletionSettings,
    FrozenDiffusionBackgroundCompleter,
)
from refractive_mam2.semantic_dataset import (
    ManifestSemanticDataset,
    semantic_collate,
)
from refractive_mam2.train import _prompt_points_from_mask
from refractive_mam2.config import (
    BackgroundConfig,
    MatterConfig,
    PipelineConfig,
    SAM2IntegrationConfig,
)
from refractive_mam2.pipeline import RefractiveMAM2
from refractive_mam2.runner import (
    load_refractive_checkpoint,
    load_refractive_training_state,
    save_refractive_checkpoint,
)


class _PipelineResult:
    def __init__(self, image):
        self.images = [image]


class RecordingInpaintPipeline:
    def __init__(self):
        self.masks = []
        self.seeds = []

    def __call__(self, *, image, mask_image, generator, **kwargs):
        del kwargs
        self.masks.append(np.asarray(mask_image).copy())
        self.seeds.append(generator.initial_seed())
        return _PipelineResult(image)


class StrictFluxFillPipeline:
    def __init__(self):
        self.calls = []

    def __call__(
        self,
        *,
        prompt,
        image,
        mask_image,
        num_inference_steps,
        guidance_scale,
        generator,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "mask_image": mask_image,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": generator.initial_seed(),
            }
        )
        return _PipelineResult(image)


def test_diffusion_wrapper_dilates_generation_mask_and_offsets_batch_seed() -> None:
    pipeline = RecordingInpaintPipeline()
    completer = FrozenDiffusionBackgroundCompleter(
        pipeline,
        settings=DiffusionCompletionSettings(seed=17, mask_dilation=1),
    )
    evidence = torch.full((2, 3, 5, 5), 0.25)
    hole = torch.zeros(2, 1, 5, 5, dtype=torch.bool)
    hole[:, :, 2, 2] = True
    output = completer(evidence, torch.ones(2, 1, 5, 5), hole)
    assert output.shape == evidence.shape
    assert pipeline.seeds == [17, 18]
    assert all((mask > 0).sum() == 9 for mask in pipeline.masks)
    assert torch.allclose(output, evidence, atol=2e-2)


def test_diffusion_wrapper_omits_unsupported_negative_prompt() -> None:
    pipeline = StrictFluxFillPipeline()
    completer = FrozenDiffusionBackgroundCompleter(
        pipeline,
        settings=DiffusionCompletionSettings(
            negative_prompt="must not be passed",
            inference_steps=2,
            guidance_scale=30.0,
        ),
    )
    evidence = torch.full((1, 3, 4, 4), 0.25)
    hole = torch.zeros(1, 1, 4, 4, dtype=torch.bool)
    hole[:, :, 1:3, 1:3] = True
    output = completer(evidence, torch.ones(1, 1, 4, 4), hole)
    assert output.shape == evidence.shape
    assert pipeline.calls[0]["num_inference_steps"] == 2
    assert pipeline.calls[0]["guidance_scale"] == 30.0


def test_flux_fill_uses_dedicated_diffusers_loader(monkeypatch) -> None:
    loaded = []

    class _LoadedPipeline:
        def load_lora_weights(self, adapter):
            loaded.append(("adapter", adapter))

        def set_progress_bar_config(self, *, disable):
            loaded.append(("progress", disable))

        def to(self, device):
            loaded.append(("device", str(device)))
            return self

    class _FluxLoader:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            loaded.append(("flux", model, kwargs))
            return _LoadedPipeline()

    class _AutoLoader:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            loaded.append(("auto", model, kwargs))
            return _LoadedPipeline()

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.FluxFillPipeline = _FluxLoader
    fake_diffusers.AutoPipelineForInpainting = _AutoLoader
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    completer = FrozenDiffusionBackgroundCompleter.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        adapter="example/adapter",
        device="cpu",
        dtype="float16",
    )
    assert isinstance(completer.pipeline, _LoadedPipeline)
    assert loaded[0][0:2] == ("flux", "black-forest-labs/FLUX.1-Fill-dev")
    assert loaded[0][2]["torch_dtype"] is torch.float32
    assert ("adapter", "example/adapter") in loaded
    assert not any(event[0] == "auto" for event in loaded)


def test_cuda_diffusion_uses_model_cpu_offload(monkeypatch) -> None:
    loaded = []

    class _LoadedPipeline:
        def set_progress_bar_config(self, *, disable):
            loaded.append(("progress", disable))

        def enable_model_cpu_offload(self, *, device):
            loaded.append(("offload", device))

        def to(self, device):
            loaded.append(("device", str(device)))
            return self

    class _FluxLoader:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            loaded.append(("flux", model, kwargs))
            return _LoadedPipeline()

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.FluxFillPipeline = _FluxLoader
    fake_diffusers.AutoPipelineForInpainting = _FluxLoader
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    FrozenDiffusionBackgroundCompleter.from_pretrained(
        "black-forest-labs/FLUX.1-Fill-dev",
        device="cuda",
        dtype="bfloat16",
    )
    assert ("offload", "cuda") in loaded
    assert not any(event[0] == "device" for event in loaded)


def test_point_and_box_prompt_protocol_stays_on_object() -> None:
    mask = torch.zeros(1, 1, 10, 20)
    mask[:, :, 2:8, 5:15] = 1
    point, labels = _prompt_points_from_mask(mask, size=100, mode="point")
    assert labels.tolist() == [[1]]
    original_x = int(point[0, 0, 0] * 20 / 100)
    original_y = int(point[0, 0, 1] * 10 / 100)
    assert mask[0, 0, original_y, original_x] == 1
    box, box_labels = _prompt_points_from_mask(mask, size=100, mode="box")
    assert box.shape == (1, 2, 2)
    assert box_labels.tolist() == [[2, 3]]


def test_manifest_semantic_dataset_mixes_vos_and_matting(tmp_path: Path) -> None:
    frame = np.full((8, 10, 3), 127, np.uint8)
    mask = np.zeros((8, 10), np.uint8)
    mask[2:7, 3:9] = 255
    alpha = mask.copy()
    Image.fromarray(frame).save(tmp_path / "frame.png")
    Image.fromarray(mask).save(tmp_path / "mask.png")
    Image.fromarray(alpha).save(tmp_path / "alpha.png")
    manifest = tmp_path / "semantic.jsonl"
    manifest.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "id": "vos",
                        "dataset_kind": "vos",
                        "frames": ["frame.png"],
                        "object_masks": ["mask.png"],
                    }
                ),
                json.dumps(
                    {
                        "id": "matting",
                        "dataset_kind": "image_matting",
                        "frames": ["frame.png"],
                        "alpha": ["alpha.png"],
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = ManifestSemanticDataset(
        [manifest],
        image_size=16,
        clip_length=2,
        random_horizontal_flip=False,
    )
    batch = semantic_collate([dataset[0], dataset[1]])
    assert batch.ground_truth.frames.shape == (2, 2, 3, 16, 16)
    assert batch.ground_truth.object_mask.shape == (2, 2, 1, 16, 16)
    assert batch.ground_truth.trimap.max() == 2
    assert batch.dataset_kinds == ["vos", "image_matting"]

    alpha_only = ManifestSemanticDataset(
        [manifest],
        image_size=16,
        require_alpha=True,
        random_horizontal_flip=False,
    )
    assert len(alpha_only) == 1
    assert alpha_only[0]["dataset_kind"] == "image_matting"


def test_base_diffusion_comparison_requires_matched_checkpoint(tmp_path: Path) -> None:
    common = {
        "dataset": "test",
        "metrics": {
            "background_true_hole_psnr": 20.0,
            "evidence_preservation_l1": 0.0,
        },
        "reproducibility": {
            "checkpoint_sha256": "abc",
            "sam2_checkpoint_sha256": "sam",
            "dataset_artifact_manifest_sha256": "data",
            "prompt_mode": "point",
            "seed": 7,
        },
    }
    base = tmp_path / "base.json"
    diffusion = tmp_path / "diffusion.json"
    base.write_text(json.dumps(common), encoding="utf-8")
    changed = json.loads(json.dumps(common))
    changed["metrics"]["background_true_hole_psnr"] = 23.0
    diffusion.write_text(json.dumps(changed), encoding="utf-8")
    report = compare_metric_files(base, diffusion)
    assert (
        report["metrics"]["background_true_hole_psnr"]["diffusion_minus_base"]
        == 3.0
    )


class _CheckpointPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.extension = torch.nn.Linear(2, 2)
        self.mam2_integration_config = SAM2IntegrationConfig()

    def mam2_extension_state_dict(self):
        return self.extension.state_dict()

    def load_mam2_extension_state_dict(self, state, strict=True):
        self.extension.load_state_dict(state, strict=strict)


class _CheckpointPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PipelineConfig()
        self.head = torch.nn.Linear(2, 2)


def test_format_v6_checkpoint_carries_resume_state(tmp_path: Path) -> None:
    predictor = _CheckpointPredictor()
    pipeline = _CheckpointPipeline()
    path = tmp_path / "checkpoint.pt"
    save_refractive_checkpoint(
        path,
        predictor,
        pipeline,
        metadata={"stage": 2},
        training_state={"epoch": 4, "global_step": 19},
    )
    metadata = load_refractive_checkpoint(path, predictor, pipeline)
    state = load_refractive_training_state(path)
    assert metadata == {"stage": 2}
    assert state == {"epoch": 4, "global_step": 19}


def test_legacy_dilated_checkpoint_initializes_new_ffc_completion(tmp_path: Path) -> None:
    predictor = _CheckpointPredictor()
    legacy = RefractiveMAM2(
        predictor,
        PipelineConfig(
            background=BackgroundConfig(
                completion_backbone="dilated",
                completion_width=8,
                completion_dilations=(1,),
            ),
            matter=MatterConfig(feature_channels=8, width=8),
        ),
    )
    with torch.no_grad():
        legacy.matter.head.weight.fill_(0.125)
    path = tmp_path / "legacy-dilated.pt"
    save_refractive_checkpoint(path, predictor, legacy)

    upgraded = RefractiveMAM2(
        predictor,
        PipelineConfig(
            background=BackgroundConfig(
                completion_backbone="ffc",
                completion_width=8,
                completion_down_blocks=1,
                completion_residual_blocks=1,
                completion_max_channels=16,
            ),
            matter=MatterConfig(feature_channels=8, width=8),
        ),
    )
    with pytest.warns(UserWarning, match="completion backbone changed"):
        load_refractive_checkpoint(path, predictor, upgraded)
    assert torch.allclose(
        upgraded.matter.head.weight,
        torch.full_like(upgraded.matter.head.weight, 0.125),
    )
