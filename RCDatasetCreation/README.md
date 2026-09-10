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

Mitsuba의 CPU Dr.Jit backend에는 Homebrew LLVM이 필요합니다.

```bash
brew install llvm
export DRJIT_LIBLLVM_PATH="$(brew --prefix llvm)/lib/libLLVM.dylib"
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
  result/prism_main_smoke/train

python tools/validate_prism_contract.py \
  result/prism_main_smoke/test
```

정상적인 경우 마지막 줄에 다음과 같은 결과가 출력됩니다.

```text
PASS sequences=2 frames=4 ...
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

## 6. Minimal renderer smoke test

한 frame만 렌더링하여 Mitsuba, EXR writing과 PRISM renderer integration을 빠르게
확인합니다. 이 config는 reflection이 켜져 있으므로 PRISM main split 검증을
대신하지 않습니다.

```bash
python render_dataset.py \
  --conf configs/dataset_cpu_smoke_background.yaml \
  --device cpu
```

## 7. Full PRISM main generation

Full config는 smoke fixture와 분리된 `dataset_resources_research/`를 사용합니다.
공식 DIV2K 800/100 background와 총 120/30 mesh를 구성합니다. Mesh pack은
Objaverse CC0/CC-BY의 선별·closed-solid repair 보조군과, 벽 두께를 명시한
deterministic procedural vessel/lens/prism 주력군의 혼합입니다. 모든 mesh는
watertight/normal/component/aspect/face-count 검사를 통과해야 합니다.

```bash
python -m pip install -r requirements-assets.txt
python tools/prepare_prism_assets.py all
python tools/prepare_prism_assets.py validate
```

`asset_manifests/prism_research_assets_v1.json`에는 source UID/URL/license,
원본·가공본 SHA-256, mesh 품질 수치, rejection audit와 split pool이 저장됩니다.
대용량 binary asset은 Git에서 제외되며 같은 명령으로 재구축합니다. 실제
validation split은 full generator가 train pool에서 별도 mesh/background를
deterministically reserve합니다. 준비가 끝나면 GPU generation을 실행합니다.

현재 고정된 `prism-research-assets-v1` 구성은 다음과 같습니다.

| pool | mesh | background | 용도 |
| --- | ---: | ---: | --- |
| train resource pool | 120 | 800 | generator가 train/validation으로 재분할 |
| held-out test pool | 30 | 100 | 개발 중 선택·튜닝에 사용하지 않는 최종 평가 |

Mesh 150개 중 40개는 Objaverse의 개별 CC0/CC-BY 자산이며, 110개는
재현 가능한 PRISM procedural closed solid입니다. 공개 mesh는 필요한 경우
0.015-unit voxel solid repair를 거친 뒤 watertight, winding, component,
aspect-ratio 검사를 다시 통과해야 합니다. Procedural hollow vessel은 열린
표면이 아니라 안쪽 벽과 바닥을 가진 watertight shell입니다. Attribution은
`dataset_resources_research/shape/ATTRIBUTION.csv`, 전수 수치와 hash는 asset
manifest에서 확인합니다.

사람이 빠르게 검수할 contact sheet는 다음 명령으로 다시 생성합니다.

```bash
python tools/render_prism_asset_contact_sheet.py
```

기본 full main config는 train 3,456, validation 384, test 960 sequence, 즉
총 38,400 frame을 만듭니다. 각 frame에 여러 HDR/GT/debug pass를 저장하므로
렌더링 전에 대용량 scratch storage를 확보해야 합니다. 빠른 기능 확인에는
반드시 smoke config를 먼저 사용하십시오.

연구용 asset index 자체를 통과하는 8-frame CPU smoke test는 다음과 같습니다.
`ResourceSubset`은 원본 index를 수정하거나 복사하지 않고 seed 기반으로
train/test mesh 각 1개와 background 각 2개만 고릅니다.

```bash
export DRJIT_LIBLLVM_PATH="$(brew --prefix llvm)/lib/libLLVM.dylib"

python render_dataset.py \
  --conf configs/dataset_prism_research_asset_smoke.yaml \
  --device cpu

python tools/validate_prism_contract.py \
  result/prism_research_asset_smoke/train
python tools/validate_prism_contract.py \
  result/prism_research_asset_smoke/test

python tools/freeze_prism_manifest.py result/prism_research_asset_smoke
python tools/freeze_prism_manifest.py result/prism_research_asset_smoke --verify
```

`freeze_prism_manifest.py`는 `dataset_manifest.json`의 `run_splits`를 따릅니다.
따라서 train shard나 validation/test 단독 Run도 해당 Run이 실제 생성한 split만
독립적으로 content-hash하고 검증할 수 있습니다.

### VESSL에서 재현 가능한 7개 Run 실행

VESSL에서는 브랜치의 최신 상태를 clone한 뒤 특정 커밋인지 비교하지 않습니다.
대신 실행할 전체 commit hash를 직접 fetch하고 detached HEAD로 checkout합니다.
따라서 `main`에 후속 커밋이 추가되어도 이미 만든 Run이 오래된 hash 검사 때문에
실패하지 않으며, 각 Run이 사용한 코드는 `PRISM_GIT_COMMIT` 로그로 남습니다.

Cloud command는 공통 실행기 `tools/vessl_generate.py`만 호출합니다. 이 실행기는
의존성 설치, research asset 압축 해제, 렌더링, 물리 계약 검증, content manifest
생성·재검증, 완료 marker 기록을 순서대로 수행합니다. POSIX shell에서 동작하지
않는 Bash array는 사용하지 않습니다.

VESSL의 `export`는 실시간 mount가 아니라 Run command가 종료된 뒤 실행됩니다.
따라서 cloud command에서는 생성기의 종료 코드를 별도로 기록하고 마지막 shell
종료 코드는 0으로 돌려 VESSL export phase가 항상 실행되게 해야 합니다. 생성기가
실패하면 dataset root에 traceback을 담은 `.generation_failed`가 남고, 성공한
경우에만 `.generation_complete`가 생깁니다. 모니터링과 후속 학습은 VESSL의
표면상 상태가 아니라 이 marker를 기준으로 성공 여부를 판정합니다. 이 방식은
렌더링 또는 후처리 오류가 발생해도 완료된 frame을 회수하고, 재실행 시 generator의
sequence/frame checkpoint 기능으로 이어서 생성하기 위한 것입니다.

```bash
# 아래 COMMIT은 반드시 실행하려는 GitHub commit의 전체 40자리 hash로 지정합니다.
COMMIT=<full-commit-hash>
git init /root/workspace/PRISM
cd /root/workspace/PRISM
git remote add origin https://github.com/Sagit25/PRISM.git
git fetch --depth 1 origin "$COMMIT"
git checkout --detach FETCH_HEAD
cd RCDatasetCreation
export PRISM_GIT_COMMIT="$COMMIT"

# 5개의 train Run에서 각각 index 0, 1, 2, 3, 4를 사용합니다.
python tools/vessl_generate.py train 0 5

# 별도 Run 두 개입니다.
python tools/vessl_generate.py validation
python tools/vessl_generate.py test
```

실제 VESSL command에서는 위 호출을 다음 POSIX wrapper로 감쌉니다.

```sh
set +e
python tools/vessl_generate.py train 0 5
generation_status=$?
set -e
sync
if [ "$generation_status" -ne 0 ]; then
  echo "PRISM generation failed; exporting recoverable partial output"
fi
exit 0
```

이는 Python/렌더러/검증 오류에 대한 결과 회수 장치입니다. 노드가 즉시 사라지는
강제 종료까지 실시간 보존하는 mount는 아니므로, Run을 수동 종료할 때는 export가
완료됐는지 Files 탭에서 확인해야 합니다.

기본 입력 archive는
`/input/assets/prism-research-assets-v1.tar`, 출력 volume 경로는
`/root/workspace/persistent_export`입니다. 완료된 데이터셋 root에는 검증까지
통과했다는 의미의 `.generation_complete`가 생성됩니다. 필요하면
`PRISM_ASSET_TAR`, `PRISM_OUTPUT_ROOT`, `PRISM_RUN_VERSION` 환경변수로 경로와
버전을 바꿀 수 있습니다.

```bash
python render_dataset.py \
  --conf configs/dataset_prism_main.yaml \
  --device gpu
```

결과 검증:

```bash
python tools/validate_prism_contract.py \
  result/prism_main/train

python tools/validate_prism_contract.py \
  result/prism_main/test
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
| `dataset_prism_research_asset_smoke.yaml` | 실제 DIV2K/Objaverse asset 경로 검증 | CPU | frame 4개/split |
| `dataset_prism_reflection_smoke.yaml` | Reflection diagnostic 검증 | CPU | frame 2개/split |
| `dataset_prism_3d_smoke.yaml` | 3D background hit/projection 검증 | CPU | frame 2개/split |
| `dataset_cpu_smoke_background.yaml` | 기존 planar renderer 최소 확인 | CPU | frame 1개/split |
| `dataset_cpu_smoke_tree_scene.yaml` | 기존 3D renderer 최소 확인 | CPU | frame 1개/split |
| `dataset_prism_main.yaml` | Full refraction/transmission main split | GPU | 4,800 sequences / 38,400 frames |
| `dataset_prism_diagnostic_reflection.yaml` | Full reflection diagnostic split | GPU | 1,200 sequences / 9,600 frames |

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
