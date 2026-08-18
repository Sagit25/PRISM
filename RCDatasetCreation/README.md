# RCTrans-based PRISM Dataset Generator

RCTrans의 scene/material/ray-tracing pipeline을 fixed-camera moving-object
video와 PRISM decomposition GT 생성용으로 확장한 버전입니다. 모든 명령은
repository root인 `RCDatasetCreation/`에서 실행합니다.

## 1. Installation

Linux 또는 Windows WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-macos.txt
```

OpenEXR I/O를 활성화합니다.

```bash
export OPENCV_IO_ENABLE_OPENEXR=1
```

CPU Mitsuba 확인:

```bash
python -c "import mitsuba as mi; mi.set_variant('llvm_ad_rgb'); print(mi.__version__, mi.variant())"
```

CUDA GPU를 사용할 경우 추가 확인:

```bash
python -c "import mitsuba as mi; mi.set_variant('cuda_ad_rgb'); print(mi.__version__, mi.variant())"
```

## 2. Config and source preflight

렌더링 전에 Python source와 모든 YAML config를 검사합니다.

```bash
python -m compileall -q projects scene_builder camera_poses utils tools

python - <<'PY'
from pathlib import Path
from omegaconf import OmegaConf

for path in sorted(Path("configs").glob("*.yaml")):
    OmegaConf.load(path)
    print("OK", path)
PY
```

## 3. Recommended first run: PRISM main paired smoke test

최신 PRISM main 규정을 한 번에 확인하는 기본 테스트입니다.

- fixed camera/background
- moving transparent object
- reflection off
- RGB transmission randomization
- 두 개의 서로 다른 background
- 동일 operator seed를 사용하는 paired-background group
- `Phi`, `u`, `R`, confidence와 모든 decomposition GT

```bash
python render_dataset.py \
  --conf configs/dataset_prism_main_smoke.yaml \
  --device cpu
```

Train과 test split을 모두 검증합니다.

```bash
python tools/validate_prism_contract.py \
  result/prism_main_smoke/train \
  --require-pairs

python tools/validate_prism_contract.py \
  result/prism_main_smoke/test \
  --require-pairs
```

정상적인 경우 마지막 줄에 다음과 같은 결과가 출력됩니다.

```text
PASS sequences=2 frames=4 paired_groups=1 ...
```

## 4. Reflection diagnostic smoke test

Fresnel reflection을 켠 별도 diagnostic split입니다. Main training data와
섞지 않습니다.

```bash
python render_dataset.py \
  --conf configs/dataset_prism_reflection_smoke.yaml \
  --device cpu

python tools/validate_prism_contract.py \
  result/prism_reflection_smoke/train

python tools/validate_prism_contract.py \
  result/prism_reflection_smoke/test
```

이 config에서는 `sequence_meta.json`의 `split_kind`가
`diagnostic_reflection`이고 `reflection_scale`은 1이어야 합니다.

## 5. 3D-background correspondence smoke test

Planar background가 아니라 editable 3D scene manifest를 사용하여 refracted
3D hit, clean-view visibility와 projected source coordinate를 검사합니다.

```bash
python render_dataset.py \
  --conf configs/dataset_prism_3d_smoke.yaml \
  --device cpu

python tools/validate_prism_contract.py \
  result/prism_3d_smoke/train \
  --formation-tol 5e-3

python tools/validate_prism_contract.py \
  result/prism_3d_smoke/test \
  --formation-tol 5e-3
```

추가로 다음 파일이 존재하는지 확인합니다.

```bash
find result/prism_3d_smoke -type f | \
  grep -E 'Bg_hit_xyz|Bg_hit_normal|Bg_object_id|Bg_clean_visible'
```

## 6. Minimal legacy renderer smoke test

한 frame만 렌더링하여 Mitsuba, EXR writing과 기존 RCTrans integration만 빠르게
확인합니다. 이 config는 reflection이 켜져 있으므로 PRISM main split 검증을
대신하지 않습니다.

```bash
python render_dataset.py \
  --conf configs/dataset_cpu_smoke_background.yaml \
  --device cpu \
  --project_name legacy_cpu_smoke
```

## 7. Full PRISM main generation

먼저 `dataset_resources/shape/`와 `dataset_resources/background/`에 실제
training asset을 준비하고 train/test list가 mesh/background identity 기준으로
분리되었는지 확인합니다. 그다음 GPU generation을 실행합니다.

```bash
python render_dataset.py \
  --conf configs/dataset_prism_main.yaml \
  --device gpu
```

결과 검증:

```bash
python tools/validate_prism_contract.py \
  result/prism_main/train \
  --require-pairs

python tools/validate_prism_contract.py \
  result/prism_main/test \
  --require-pairs
```

CPU에서 같은 full config를 실행할 수도 있지만 시간이 오래 걸립니다.

```bash
python render_dataset.py \
  --conf configs/dataset_prism_main.yaml \
  --device cpu
```

## 8. Full reflection diagnostic generation

```bash
python render_dataset.py \
  --conf configs/dataset_prism_diagnostic_reflection.yaml \
  --device gpu

python tools/validate_prism_contract.py \
  result/prism_diagnostic_reflection/train

python tools/validate_prism_contract.py \
  result/prism_diagnostic_reflection/test
```

## 9. Config summary

| Config | 목적 | 기본 device | 규모 |
| --- | --- | --- | --- |
| `dataset_prism_main_smoke.yaml` | 최신 main 및 paired-background 검증 | CPU | background 2개, frame 4개/split |
| `dataset_prism_reflection_smoke.yaml` | Reflection diagnostic 검증 | CPU | frame 2개/split |
| `dataset_prism_3d_smoke.yaml` | 3D background hit/projection 검증 | CPU | frame 2개/split |
| `dataset_cpu_smoke_background.yaml` | 기존 planar renderer 최소 확인 | CPU | frame 1개/split |
| `dataset_cpu_smoke_tree_scene.yaml` | 기존 3D renderer 최소 확인 | CPU | frame 1개/split |
| `dataset_prism_main.yaml` | Full refraction/transmission main split | GPU | config 값에 따름 |
| `dataset_prism_diagnostic_reflection.yaml` | Full reflection diagnostic split | GPU | config 값에 따름 |

## 10. Canonical output contract

Sequence-level outputs:

```text
*_background.exr          reusable counterfactual background Bcf
*_camera_intrinsic.npy
*_camera_extrinsic.npy
*_sequence_meta.json      seeds, material, pair group and split metadata
```

Frame-level outputs:

```text
*_I.exr                   observed frame I
*_object_mask.png         geometric object mask
*_alpha.npy               scalar alpha
*_CF.exr                  G = alpha * F_std
*_F.exr                   standard straight foreground F_std
*_T.exr                   normalized RGB color transmission C
*_A.exr                   tau = (1-alpha) * C
*_Phi.npy                 absolute source coordinate Phi
*_u.npy                   displacement u = Phi - x
*_R.exr                   signed R = I - G - tau * B(Phi)
*_confidence.npy          correspondence/model-fit confidence
*_phi_valid.png           hard correspondence validity
*_object_pose.npy         frame-level object pose
```

`*_Phi_src.npy`는 `*_Phi.npy`의 compatibility alias입니다. v15부터
`*_Phi.npy`는 displacement가 아니므로 flow가 필요하면 반드시 `*_u.npy`를
사용합니다.

## 11. Validation performed by `validate_prism_contract.py`

검증 스크립트는 다음 조건을 검사하고 하나라도 실패하면 non-zero exit code를
반환합니다.

```text
Phi = x + u                         on valid pixels
tau = (1-alpha) * C
I = G + tau * B(Phi) + R
Phi_src = Phi
0 <= confidence <= 1
main split reflection_scale = 0
paired group camera/material/operator seed equality
paired group alpha/Phi/u/object-pose equality
paired group background diversity
```

## 12. Resume and rerun behavior

Generator는 frame checkpoint와 `generator_version`을 확인하여 완료된 결과를
건너뜁니다. 기존 결과를 보존하면서 새로 실행하려면 `--project_name`으로 새로운
output name을 지정합니다.

```bash
python render_dataset.py \
  --conf configs/dataset_prism_main_smoke.yaml \
  --device cpu \
  --project_name prism_main_smoke_retry01
```

결과 위치는 항상 다음 형식입니다.

```text
result/<project_name>/train/
result/<project_name>/test/
```

## 13. Troubleshooting

### OpenCV가 EXR을 읽거나 쓰지 못하는 경우

Python process가 시작되기 전에 환경 변수를 설정합니다.

```bash
export OPENCV_IO_ENABLE_OPENEXR=1
```

### `cuda_ad_rgb`를 사용할 수 없는 경우

먼저 CPU smoke test를 실행하고 NVIDIA driver/CUDA 지원 환경에서 다음을 다시
확인합니다.

```bash
nvidia-smi
python -c "import mitsuba as mi; mi.set_variant('cuda_ad_rgb'); print(mi.variant())"
```

### `No paired-background group` 오류

`PairedBackground.enabled: true`, `backgrounds_per_shape >= 2`인지 확인하고
background list에 서로 다른 asset이 최소 두 개 있는지 확인합니다. 제공된 smoke
resource에는 `smoke_background.png`와 `smoke_background_alt.ppm`이 포함됩니다.

### 새로운 수정이 반영되지 않고 모두 skip되는 경우

기존 결과를 지우는 대신 새로운 project name으로 실행합니다.

```bash
python render_dataset.py \
  --conf configs/dataset_prism_main_smoke.yaml \
  --device cpu \
  --project_name prism_main_smoke_new
```
