# 서울대 CSE GPU 워크스페이스에서 PRISM 사용

## 현재 배치

- Workspace: `ws-wkuoxg0il92t0` (`sagit25`), SGS 화면: <https://sgs.snucse.org/ws/wkuoxg0il92t0>
- 영구 볼륨: PVC `prism-storage`, 200 GiB, `ReadWriteOnce`, `/data`에 마운트
- 관리용 CPU Pod: `prism-transfer` (GPU를 점유하지 않음)
- 코드: `/data/code/PRISM` (커밋 `38019fd7410770ad39a9814ecbb6d312be5682b0`)
- 데이터: `/data/datasets/prism-full-64-singleflow-pack-v2-369367271785-output/`
- 학습 출력 권장 경로: `/data/checkpoints/<실험명>/`, 캐시: `/data/cache/`, 로그: `/data/logs/`
- 공식 MEMatte ViT-S AIM-500 가중치: `/data/models/MEMatte_AIM500.pth` (SHA-256 `538cf871933dd0663a6bc28a4bec9b870cc2de909c7431b857b72ce427e82a09`)

컨테이너의 `/root`나 `/tmp`에만 저장한 파일은 Pod 교체 시 사라질 수 있습니다. 보존할 데이터와 체크포인트는 반드시 PVC의 `/data` 아래에 저장하세요. 로컬 kubeconfig는 `/Users/yangsukhun/Downloads/kubeconfig.txt`에 있으며 **저장소에 커밋하거나 공유하면 안 됩니다.**

## 접속과 확인

아래 명령에서 `<KUBECONFIG>`는 로컬의 실제 kubeconfig 경로로 바꾸세요.

```bash
kubectl --kubeconfig=<KUBECONFIG> -n ws-wkuoxg0il92t0 get pod,pvc,job
kubectl --kubeconfig=<KUBECONFIG> -n ws-wkuoxg0il92t0 exec -it prism-transfer -- sh
```

Pod 안에서 `ls /data/code/PRISM`, `ls /data/datasets`, `df -h /data`로 코드·데이터·남은 공간을 확인합니다. `prism-transfer`는 파일 관리용이므로 이 Pod에서 GPU 학습을 실행하지 않습니다.

## 로컬과 영구 볼륨 간 파일 이동

```bash
# 로컬 파일을 영구 볼륨으로 복사
kubectl --kubeconfig=<KUBECONFIG> -n ws-wkuoxg0il92t0 cp ./my-file prism-transfer:/data/my-file

# 결과를 로컬로 가져오기
kubectl --kubeconfig=<KUBECONFIG> -n ws-wkuoxg0il92t0 cp prism-transfer:/data/checkpoints/my-run ./my-run
```

대용량 데이터는 `network/scripts/migrate_vessl_archive_to_sgs.py`의 `--relay` 옵션으로 VESSL에서 한 파일씩 중계하고 원본 manifest의 크기 및 SHA-256을 검증합니다. 검증이 완료되기 전에는 데이터셋 전체가 이전되었다고 간주하지 마세요. 영구 볼륨을 삭제하거나 `kubectl delete pvc prism-storage`를 실행하면 데이터 복구가 어려울 수 있습니다.

## GPU 작업 실행 원칙

이 워크스페이스의 GPU 할당량은 **1개**입니다. 학습은 관리용 CPU Pod와 별개의 Job/Pod에서 `nvidia.com/gpu: 1`을 요청하고 같은 PVC를 `/data`에 마운트해야 합니다. 학습 프로세스의 데이터 경로를 `/data/datasets/...`, 출력 경로를 `/data/checkpoints/<실험명>`으로 지정하세요. 기존 VESSL 체크포인트를 자동 복원하지 말고, 새 실험의 복원 경로를 명시적으로 확인하세요.

현재 PVC는 `openebs-hostpath`의 `ReadWriteOnce`입니다. `network/k8s/sgs-prism-gpu-smoke.yaml`의 Job이 2026-10-16에 노드 `ford`에서 완료되어 A100-SXM4-40GB와 동일 PVC 마운트를 확인했습니다(`PRISM_SGS_GPU_PVC_SMOKE_OK`). 이후 노드 배치나 볼륨 구성이 바뀌면 smoke를 다시 확인하세요. GPU 세션은 유휴 상태로 오래 두지 말고 Job으로 실행해 로그와 종료 상태를 남기는 편이 안전합니다.

```bash
kubectl --kubeconfig=<KUBECONFIG> -n ws-wkuoxg0il92t0 get jobs,pods -o wide
kubectl --kubeconfig=<KUBECONFIG> -n ws-wkuoxg0il92t0 logs job/<작업명> --tail=100
kubectl --kubeconfig=<KUBECONFIG> -n ws-wkuoxg0il92t0 describe pod <Pod명>
```

64×64 데이터의 archive tar는 `PRISM_ARCHIVE_ROOT=/data/datasets/prism-full-64-singleflow-pack-v2-369367271785-output`으로 로컬 materializer에 전달할 수 있습니다. `PRISM_MEMATTE_CHECKPOINT=/data/models/MEMatte_AIM500.pth`를 설정하면 이미 검증된 가중치를 사용합니다. 다만 VESSL용 `volume://` 체크포인트 경로, W&B/Hugging Face 토큰, FLUX/SAM2 설치 절차는 서울대 Job에 맞춰 바꾸고 작은 샘플 학습을 통과시킨 뒤 전체 학습을 옮기세요. 현재 VESSL v7r1과 서울대 이전은 독립적이며, 서울대에서 전체 학습을 새로 시작한 상태는 아닙니다.
