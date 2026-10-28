import json
import subprocess
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
    load_stage1_mam2_checkpoint,
    save_refractive_checkpoint,
)


class _PipelineResult:
    def __init__(self, image):
        self.images = image if isinstance(image, list) else [image]


class RecordingInpaintPipeline:
    def __init__(self):
        self.masks = []
        self.images = []
        self.seeds = []
        self.call_count = 0

    def __call__(self, *, image, mask_image, generator, **kwargs):
        del kwargs
        self.call_count += 1
        masks = mask_image if isinstance(mask_image, list) else [mask_image]
        images = image if isinstance(image, list) else [image]
        generators = generator if isinstance(generator, list) else [generator]
        self.masks.extend(np.asarray(mask).copy() for mask in masks)
        self.images.extend(np.asarray(item).copy() for item in images)
        self.seeds.extend(item.initial_seed() for item in generators)
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


def test_diffusion_wrapper_dilates_generation_mask_and_uses_stable_seed() -> None:
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
    assert pipeline.seeds == [17, 17]
    assert pipeline.call_count == 1
    assert all((mask > 0).sum() == 9 for mask in pipeline.masks)
    assert torch.allclose(output, evidence, atol=2e-2)


def test_diffusion_wrapper_can_inpaint_on_larger_canvas_then_downsample() -> None:
    pipeline = RecordingInpaintPipeline()
    completer = FrozenDiffusionBackgroundCompleter(
        pipeline,
        settings=DiffusionCompletionSettings(
            mask_dilation=2, min_working_side=128,
        ),
    )
    evidence = torch.full((1, 3, 64, 64), 0.25)
    hole = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
    hole[:, :, 30:34, 30:34] = True

    output = completer(evidence, (~hole).float(), hole)

    assert pipeline.call_count == 1
    assert pipeline.masks[0].shape == (128, 128)
    assert output.shape == evidence.shape
    # The recording stub echoes the masked input instead of completing the
    # hole; only pixels that FLUX must preserve are expected to match.
    known = (~hole).expand_as(evidence)
    assert torch.allclose(output[known], evidence[known], atol=2e-2)


def test_frozen_diffusion_caches_only_matching_image_mask_and_seed() -> None:
    pipeline = RecordingInpaintPipeline()
    completer = FrozenDiffusionBackgroundCompleter(
        pipeline,
        settings=DiffusionCompletionSettings(seed=17, image_cache_entries=2),
    )
    evidence = torch.full((2, 3, 8, 8), 0.25)
    evidence[1] = 0.5
    hole = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    hole[:, :, 3, 3] = True
    coverage = (~hole).float()

    original = completer(evidence, coverage, hole)
    repeated = completer(evidence, coverage, hole)
    assert pipeline.call_count == 1
    assert pipeline.seeds == [17, 17]
    assert (completer.image_cache_hits, completer.image_cache_misses) == (2, 2)
    assert completer.image_cache_size == 2
    torch.testing.assert_close(repeated, original)

    changed = evidence.clone()
    changed[1, :, 0, 0] = 0.75
    completer(changed, coverage, hole)
    assert pipeline.call_count == 2
    assert pipeline.seeds == [17, 17, 17]
    assert completer.image_cache_size == 2

    changed_mask = hole.clone()
    changed_mask[0, :, 4, 4] = True
    completer(changed, (~changed_mask).float(), changed_mask)
    assert pipeline.call_count == 3
    assert pipeline.seeds[-1] == 17


def test_flux_pixel_refiner_learns_only_inside_hole_with_cached_prior() -> None:
    pipeline = RecordingInpaintPipeline()
    completer = FrozenDiffusionBackgroundCompleter(
        pipeline,
        settings=DiffusionCompletionSettings(pixel_refinement=True),
    )
    evidence = torch.full((1, 3, 16, 16), 0.4)
    evidence[:, :, 5:11, 5:11] = 0
    hole = torch.zeros((1, 1, 16, 16), dtype=torch.bool)
    hole[:, :, 5:11, 5:11] = True
    inverse = torch.full_like(evidence, 0.6, requires_grad=True)
    weight = torch.ones_like(hole, dtype=evidence.dtype)
    flow = torch.zeros((1, 2, 16, 16), requires_grad=True)
    args = (inverse, weight, flow)

    first = completer(evidence, (~hole).float(), hole, flow_evidence=args)
    second = completer(evidence, (~hole).float(), hole, flow_evidence=args)
    assert pipeline.call_count == 1
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[~hole.expand_as(first)], evidence[~hole.expand_as(evidence)])
    assert first.grad_fn is not None
    loss = (second[hole.expand_as(second)] - 0.7).square().mean()
    loss.backward()
    assert completer.pixel_refiner.body[-1].weight.grad.abs().sum() > 0
    optimizer = torch.optim.SGD(completer.pixel_refiner.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad()
    inverse.grad = None
    flow.grad = None
    refined = completer(evidence, (~hole).float(), hole, flow_evidence=args)
    (refined[hole.expand_as(refined)] - 0.7).square().mean().backward()
    assert inverse.grad is not None and inverse.grad.abs().sum() > 0
    assert flow.grad is not None and flow.grad.abs().sum() > 0


def test_diffusion_prior_is_invariant_to_batch_reordering() -> None:
    pipeline = RecordingInpaintPipeline()
    completer = FrozenDiffusionBackgroundCompleter(
        pipeline,
        settings=DiffusionCompletionSettings(seed=23, image_cache_entries=4),
    )
    evidence = torch.stack(
        (torch.full((3, 8, 8), 0.2), torch.full((3, 8, 8), 0.7))
    )
    hole = torch.zeros(2, 1, 8, 8, dtype=torch.bool)
    hole[..., 3, 3] = True
    coverage = (~hole).float()
    original = completer(evidence, coverage, hole)
    reordered = completer(evidence.flip(0), coverage.flip(0), hole.flip(0))
    assert pipeline.call_count == 1
    assert pipeline.seeds == [23, 23]
    torch.testing.assert_close(reordered, original.flip(0))


def test_diffusion_image_cache_can_be_disabled() -> None:
    pipeline = RecordingInpaintPipeline()
    completer = FrozenDiffusionBackgroundCompleter(
        pipeline,
        settings=DiffusionCompletionSettings(image_cache_entries=0),
    )
    evidence = torch.full((1, 3, 8, 8), 0.25)
    hole = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    hole[..., 3, 3] = True
    completer(evidence, (~hole).float(), hole)
    completer(evidence, (~hole).float(), hole)
    assert pipeline.call_count == 2


def test_upscaled_flux_context_does_not_mix_zero_hole_into_known_pixels() -> None:
    pipeline = RecordingInpaintPipeline()
    completer = FrozenDiffusionBackgroundCompleter(
        pipeline,
        settings=DiffusionCompletionSettings(min_working_side=128),
    )
    evidence = torch.full((1, 3, 64, 64), 0.25)
    evidence[:, :, 30:34, 30:34] = 0.0
    hole = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
    hole[:, :, 30:34, 30:34] = True

    output = completer(evidence, (~hole).float(), hole)

    assert pipeline.masks[0][59, 64] == 0  # FLUX must preserve this pixel
    assert pipeline.images[0][59, 64, 0] == pipeline.images[0][58, 64, 0]
    assert torch.allclose(output[0, :, 29, 32], torch.full((3,), 0.25), atol=0.01)


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


def test_flux_fill_reuses_fixed_prompt_embeddings_across_batches() -> None:
    class CachedPromptPipeline:
        def __init__(self) -> None:
            self.encode_calls = 0
            self.batch_sizes = []

        def encode_prompt(self, *, prompt, device, num_images_per_prompt):
            assert prompt == DiffusionCompletionSettings().prompt
            assert num_images_per_prompt == 1
            self.encode_calls += 1
            return (
                torch.ones(1, 4, 8, device=device),
                torch.ones(1, 8, device=device),
                torch.zeros(4, 3, device=device),
            )

        def __call__(self, *, image, prompt_embeds, pooled_prompt_embeds, **kwargs):
            assert "prompt" not in kwargs
            assert prompt_embeds.shape[0] == pooled_prompt_embeds.shape[0]
            self.batch_sizes.append(prompt_embeds.shape[0])
            return _PipelineResult(image)

    pipeline = CachedPromptPipeline()
    completer = FrozenDiffusionBackgroundCompleter(
        pipeline, cache_prompt_embeddings=True,
    )
    for batch_size in (2, 1):
        # Change the image so the output cache cannot hide this second call;
        # this test is specifically about prompt-embedding reuse.
        evidence = torch.full((batch_size, 3, 4, 4), 0.25 * batch_size)
        hole = torch.ones(batch_size, 1, 4, 4, dtype=torch.bool)
        output = completer(evidence, hole.float(), hole)
        assert output.shape == evidence.shape
    assert pipeline.encode_calls == 1
    assert pipeline.batch_sizes == [2, 1]


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


def test_cuda_diffusion_can_offload_each_module_for_small_gpus(monkeypatch) -> None:
    loaded = []

    class _LoadedPipeline:
        def set_progress_bar_config(self, *, disable):
            loaded.append(("progress", disable))

        def enable_model_cpu_offload(self, *, device):
            loaded.append(("model_offload", device))

        def enable_sequential_cpu_offload(self, *, device):
            loaded.append(("sequential_offload", device))

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
        sequential_cpu_offload=True,
    )
    assert ("sequential_offload", "cuda") in loaded
    assert not any(event[0] in ("model_offload", "device") for event in loaded)


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


def test_format_v8_checkpoint_carries_resume_state(tmp_path: Path) -> None:
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


def test_stage2_checkpoint_adds_new_objectclear_flow_and_pixel_adapters(tmp_path: Path) -> None:
    predictor = _CheckpointPredictor()
    stage2 = _CheckpointPipeline()
    stage2.background_model = torch.nn.Module()
    stage2.background_model.completion = torch.nn.Linear(2, 2)
    path = tmp_path / "stage2.pt"
    save_refractive_checkpoint(path, predictor, stage2)

    stage3 = _CheckpointPipeline()
    stage3.config.background.completion_variant = "diffusion"
    stage3.background_model = torch.nn.Module()
    stage3.background_model.completion = torch.nn.Linear(2, 2)
    stage3.background_model.diffusion_completion = torch.nn.Module()
    stage3.background_model.diffusion_completion.flow_latent_adapter = torch.nn.Linear(2, 2)
    stage3.background_model.diffusion_completion.pixel_refiner = torch.nn.Linear(2, 2)
    original_adapter = stage3.background_model.diffusion_completion.flow_latent_adapter.weight.detach().clone()
    original_pixel = stage3.background_model.diffusion_completion.pixel_refiner.weight.detach().clone()
    load_refractive_checkpoint(path, _CheckpointPredictor(), stage3)
    torch.testing.assert_close(
        stage3.background_model.completion.weight,
        stage2.background_model.completion.weight,
    )
    torch.testing.assert_close(
        stage3.background_model.diffusion_completion.flow_latent_adapter.weight,
        original_adapter,
    )
    torch.testing.assert_close(
        stage3.background_model.diffusion_completion.pixel_refiner.weight,
        original_pixel,
    )


def test_old_flux_checkpoint_can_initialize_new_pixel_refiner(tmp_path: Path) -> None:
    old = _CheckpointPipeline()
    old.config.background.completion_variant = "diffusion"
    old.background_model = torch.nn.Module()
    old.background_model.completion = torch.nn.Linear(2, 2)
    old.background_model.diffusion_completion = torch.nn.Module()
    path = tmp_path / "old-flux.pt"
    save_refractive_checkpoint(path, _CheckpointPredictor(), old)

    new = _CheckpointPipeline()
    new.config.background.completion_variant = "diffusion"
    new.config.background.diffusion_pixel_refinement = True
    new.background_model = torch.nn.Module()
    new.background_model.completion = torch.nn.Linear(2, 2)
    new.background_model.diffusion_completion = torch.nn.Module()
    new.background_model.diffusion_completion.pixel_refiner = torch.nn.Linear(2, 2)
    initial = new.background_model.diffusion_completion.pixel_refiner.weight.detach().clone()
    load_refractive_checkpoint(path, _CheckpointPredictor(), new)
    torch.testing.assert_close(
        new.background_model.diffusion_completion.pixel_refiner.weight, initial,
    )


def test_checkpoint_can_add_only_foreground_residual_refiner(tmp_path: Path) -> None:
    old = _CheckpointPipeline()
    old.matter = torch.nn.Module()
    old.matter.ARCHITECTURE = "prism-flow-residual-v1"
    old.matter.head = torch.nn.Linear(2, 2)
    path = tmp_path / "old-foreground.pt"
    save_refractive_checkpoint(path, _CheckpointPredictor(), old)

    upgraded = _CheckpointPipeline()
    upgraded.config.matter.refine_straight_foreground = True
    upgraded.matter = torch.nn.Module()
    upgraded.matter.ARCHITECTURE = old.matter.ARCHITECTURE + "-foreground-residual-v1"
    upgraded.matter.head = torch.nn.Linear(2, 2)
    upgraded.matter.foreground_refiner = torch.nn.Linear(2, 2)
    initial_refiner = upgraded.matter.foreground_refiner.weight.detach().clone()
    with pytest.warns(UserWarning, match="adding the foreground residual refiner"):
        load_refractive_checkpoint(path, _CheckpointPredictor(), upgraded)
    torch.testing.assert_close(upgraded.matter.head.weight, old.matter.head.weight)
    torch.testing.assert_close(
        upgraded.matter.foreground_refiner.weight, initial_refiner,
    )

    incompatible = _CheckpointPipeline()
    incompatible.config.matter.refine_straight_foreground = True
    incompatible.matter = torch.nn.Module()
    incompatible.matter.ARCHITECTURE = old.matter.ARCHITECTURE + "-other-change"
    incompatible.matter.head = torch.nn.Linear(2, 2)
    incompatible.matter.foreground_refiner = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        load_refractive_checkpoint(path, _CheckpointPredictor(), incompatible)


def test_checkpoint_can_add_only_coarse_to_fine_foreground_detail(tmp_path: Path) -> None:
    old = _CheckpointPipeline()
    old.matter = torch.nn.Module()
    old.matter.ARCHITECTURE = "prism-mam2-bridge-v1"
    old.matter.head = torch.nn.Linear(2, 2)
    path = tmp_path / "old-detail.pt"
    save_refractive_checkpoint(path, _CheckpointPredictor(), old)

    upgraded = _CheckpointPipeline()
    upgraded.config.matter.refine_straight_foreground_detail = True
    upgraded.matter = torch.nn.Module()
    upgraded.matter.ARCHITECTURE = old.matter.ARCHITECTURE + "-foreground-coarse-fine-v1"
    upgraded.matter.head = torch.nn.Linear(2, 2)
    upgraded.matter.foreground_detail_refiner = torch.nn.Linear(2, 2)
    detail_initial = upgraded.matter.foreground_detail_refiner.weight.detach().clone()
    with pytest.warns(UserWarning, match="coarse-to-fine foreground detail"):
        load_refractive_checkpoint(path, _CheckpointPredictor(), upgraded)
    torch.testing.assert_close(upgraded.matter.head.weight, old.matter.head.weight)
    torch.testing.assert_close(upgraded.matter.foreground_detail_refiner.weight, detail_initial)


def test_checkpoint_can_add_only_mam2_flow_guided_bridge(tmp_path: Path) -> None:
    old = _CheckpointPipeline()
    old.matter = torch.nn.Module()
    old.matter.ARCHITECTURE = "prism-foreground-residual-v1"
    old.matter.head = torch.nn.Linear(2, 2)
    path = tmp_path / "old-bridge.pt"
    save_refractive_checkpoint(path, _CheckpointPredictor(), old)

    upgraded = _CheckpointPipeline()
    upgraded.config.matter.flow_guided_latent_bridge = True
    upgraded.config.matter.use_mam2_foreground_prior = True
    upgraded.matter = torch.nn.Module()
    upgraded.matter.ARCHITECTURE = (
        old.matter.ARCHITECTURE
        + "-flow-guided-latent-bridge-v1-pretrained-mam2-flow-bridge-v1"
    )
    upgraded.matter.head = torch.nn.Linear(2, 2)
    upgraded.matter.latent_bridge = torch.nn.Linear(2, 2)
    upgraded.matter.mam2_bridge_projection = torch.nn.Linear(2, 2)
    initial_bridge = upgraded.matter.latent_bridge.weight.detach().clone()
    initial_projection = upgraded.matter.mam2_bridge_projection.weight.detach().clone()
    with pytest.warns(UserWarning, match="adding the MAM2 flow-guided foreground bridge"):
        load_refractive_checkpoint(path, _CheckpointPredictor(), upgraded)
    torch.testing.assert_close(upgraded.matter.head.weight, old.matter.head.weight)
    torch.testing.assert_close(upgraded.matter.latent_bridge.weight, initial_bridge)
    torch.testing.assert_close(
        upgraded.matter.mam2_bridge_projection.weight, initial_projection,
    )

    incompatible = _CheckpointPipeline()
    incompatible.config.matter.flow_guided_latent_bridge = True
    incompatible.config.matter.use_mam2_foreground_prior = True
    incompatible.config.matter.latent_bridge_width = 64
    incompatible.matter = torch.nn.Module()
    incompatible.matter.ARCHITECTURE = upgraded.matter.ARCHITECTURE
    incompatible.matter.head = torch.nn.Linear(2, 2)
    incompatible.matter.latent_bridge = torch.nn.Linear(2, 2)
    incompatible.matter.mam2_bridge_projection = torch.nn.Linear(2, 2)
    with pytest.warns(UserWarning, match="adding the MAM2 flow-guided foreground bridge"):
        with pytest.raises(RuntimeError, match="latent_bridge_width"):
            load_refractive_checkpoint(path, _CheckpointPredictor(), incompatible)


def test_pretrained_foreground_prior_survives_stage_checkpoint_handoff(
    tmp_path: Path,
) -> None:
    """A Stage 2/3 handoff must restore the fine-tuned prior, not reset it."""

    predictor = _CheckpointPredictor()
    trained = _CheckpointPipeline()
    trained.config.matter.use_fba_foreground_prior = True
    trained.matter = torch.nn.Module()
    trained.matter.ARCHITECTURE = "test-flow-guided-pretrained-fba-features-v2"
    trained.foreground_prior_model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        trained.foreground_prior_model.weight.fill_(0.375)
        trained.foreground_prior_model.bias.fill_(-0.125)
    path = tmp_path / "stage2.pt"
    save_refractive_checkpoint(path, predictor, trained)

    resumed = _CheckpointPipeline()
    resumed.config.matter.use_fba_foreground_prior = True
    resumed.matter = torch.nn.Module()
    resumed.matter.ARCHITECTURE = trained.matter.ARCHITECTURE
    resumed.foreground_prior_model = torch.nn.Linear(2, 2)
    load_refractive_checkpoint(path, _CheckpointPredictor(), resumed)
    assert torch.equal(
        resumed.foreground_prior_model.weight,
        trained.foreground_prior_model.weight,
    )
    assert torch.equal(
        resumed.foreground_prior_model.bias,
        trained.foreground_prior_model.bias,
    )

    incompatible = _CheckpointPipeline()
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        load_refractive_checkpoint(path, _CheckpointPredictor(), incompatible)


def test_completed_stage_reuse_rejects_a_different_fba_variant(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "stage2.pt"
    torch.save(
        {
            "format_version": 8,
            "pipeline_config": {
                "matter": {
                    "flow_guided_latent_bridge": True,
                    "use_fba_foreground_prior": True,
                },
                "background": {"evidence_flux_only": False},
            },
            "metadata": {
                "arguments": {
                    "stage": "2",
                    "clip_length": 1,
                    "loss_profile": "optical_priority",
                }
            },
        },
        checkpoint,
    )
    script = Path(__file__).parents[1] / "scripts" / "verify_prism_stage_checkpoint.py"
    common = [
        sys.executable, str(script), str(checkpoint),
        "--expect", "arguments", "stage", "2",
        "--expect", "arguments", "clip_length", "1",
        "--expect", "matter", "flow_guided_latent_bridge", "true",
        "--expect", "background", "evidence_flux_only", "false",
    ]
    compatible = subprocess.run(
        [*common, "--expect", "matter", "use_fba_foreground_prior", "true"],
        capture_output=True, text=True, check=False,
    )
    assert compatible.returncode == 0, compatible.stderr
    incompatible = subprocess.run(
        [*common, "--expect", "matter", "use_fba_foreground_prior", "false"],
        capture_output=True, text=True, check=False,
    )
    assert incompatible.returncode != 0
    assert "use a new output root" in incompatible.stderr


def test_stage1_transfer_loads_only_mam2_and_rejects_other_sources(tmp_path: Path) -> None:
    trained = _CheckpointPredictor()
    with torch.no_grad():
        trained.extension.weight.fill_(0.25)
    source_pipeline = _CheckpointPipeline()
    source = tmp_path / "prism_stage1b_best.pt"
    save_refractive_checkpoint(source, trained, source_pipeline)

    recipient = _CheckpointPredictor()
    recipient_pipeline = _CheckpointPipeline()
    original_head = recipient_pipeline.head.weight.detach().clone()
    load_stage1_mam2_checkpoint(source, recipient)
    assert torch.allclose(recipient.extension.weight, trained.extension.weight)
    assert torch.equal(recipient_pipeline.head.weight, original_head)

    wrong_name = tmp_path / "prism_stage2_best.pt"
    source.rename(wrong_name)
    with pytest.raises(RuntimeError, match="Stage 1B best"):
        load_stage1_mam2_checkpoint(wrong_name, recipient)


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
