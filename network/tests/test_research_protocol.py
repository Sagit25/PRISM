import json
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
from refractive_mam2.config import PipelineConfig
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

    def mam2_extension_state_dict(self):
        return self.extension.state_dict()

    def load_mam2_extension_state_dict(self, state, strict=True):
        self.extension.load_state_dict(state, strict=strict)


class _CheckpointPipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = PipelineConfig()
        self.head = torch.nn.Linear(2, 2)


def test_format_v5_checkpoint_carries_resume_state(tmp_path: Path) -> None:
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
