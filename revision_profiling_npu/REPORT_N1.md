# Phase N1 보고 (최종)

작성: 2026-10-15 17:10 KST, NPU 보드. Branch `revision-profiling-npu`(worktree `../mms_npu_port`, 기준 `origin/revision-profiling` `38111dd`, commit 안 함).
표기: [확인] / [추정] / [확인 불가]. 판정 기준은 모두 실행 전에 고정했고 결과를 본 뒤 바꾸지 않았다.

## 0. 요약

| 항목 | 결과 |
|---|---|
| 호환성 (§2-1) | **통과.** runtime 모듈 전부 Python 3.8 compile·import OK. 설치한 패키지 0 |
| 이식 (§2-2, §2-3) | **완료.** 기존 파일 수정은 `model_registry.yaml` 추가(+109행)뿐. 후보 93개 대조 통과 |
| CPU 회귀 (§2-4) | **통과.** valid, GPU 보드 JSON과 key path 143개 동일 |
| smoke (§3) | **통과.** 10 run 전부 valid, conservation 0, trace ok, failed 0, sha256 전후 동일 |
| isolated (§4) | **완료.** C^max: resnet50 52.50 req/s (CPU), yolov3 14.27 req/s (npu0 / small) |
| pilot (§5) | **완료, 142 run 전부 ok.** β=0.1은 (C2) 실패, **β=0.25가 A–F 전부 통과 → β = 0.25 선택** |
| 중지 조건 | 해당 없음 (3.8 비호환 없음, 바이너리 정상, conservation 0, A/B/F 통과, 기존 파일 추가 수정 불필요) |
| **campaign 예상 시간** | **492 run × 40.9 s(pilot 실측, run 사이 overhead 포함) ≈ 20,100 s ≈ 5.6 h** (pilot 136 run이 92.5 min) |

이식 중 바로잡은 사실 [확인]:
1. **기존 driver(.so, repo 사본)는 호출 2번 중 1번 출력이 깨진다.** 사용자 결정으로 `/workspace/lib` 판(sha `92ad…`)으로 교체했다(§2-6).
2. **ResNet NPU 바이너리의 출력은 avgpool feature 2048개다.** 기존 `p2.onnx`(1000 logit 가정)와 맞지 않아 기존 경로의 NPU top-1은 1/128이었다. FC를 host에서 계산한 뒤 115/128(CPU 113/128).
3. **NeublaDriver는 thread-affine이고 한 process에 2개를 둘 수 없다.** NPU instance마다 별도 server process를 둔다.
4. **재부팅 전 DMA32 고갈**(plda SG list 할당 실패)은 재부팅으로 회복됐다. pilot 142 run 동안 DMA32 free pages는 176k–181k로 감소 추세가 없었다.

## 1. 호환성·패키지 [확인]

- 컨테이너 `neubla/antara`, Python 3.8.18. `revision_profiling/code/*.py` 29개를 compile·import로 점검했다. runtime 모듈은 모두 OK. import 실패 3개(`analyze_smoke`, `check_cuda_cpu_nodes`, `check_models_sanity`)는 import 시점에 GPU/데이터를 쓰는 분석 script로 이식 대상이 아니다.
- 사용 패키지: numpy 1.24.4, onnxruntime 1.19.2, opencv 4.13.0, PyYAML 6.0.3, psutil 7.2.2 — 모두 기설치. **설치·업그레이드 0.** `tokenizers`(LM 전용)는 없지만 NPU 경로에 불필요.
- sudo: 측정 경로에는 필요 없다. NPU 장치 복구(`hotreset.sh`)에만 컨테이너 안 `sudo -n`을 썼다.
- 부수 사고(원상복구): import 점검 중 `family_reconstruct.py`가 tracked CSV 2개를 다시 썼다(CRLF→LF만 다름). `git restore`로 되돌렸다.

## 2. 이식 내용

### 2-1. 구조와 근거
- **새 파일은 `revision_profiling/code/`에 평평하게 둔다.** executor의 module-path 검사가 `code/` 밖 repo 모듈을 invalid로 처리하기 때문이다.
- **`executor.py`는 수정하지 않는다.** 새 entry `executor_npu.py`가 module attribute 3개만 바꿔 넣고 `executor.main()`을 부른다.
  - `make_worker`: `antara_*` key만 NPU 보드 worker로, 나머지는 원래 함수로
  - `GpuSampler` → NullSampler (이 보드에 nvidia-smi 없음)
  - `WINDOW_S` → 30 s (warm-up 5 s 그대로)
- **CPU–GPU 경로가 불변인 이유:** 기존 코드 파일 diff 0, registry는 append-only이고 기존 8개 항목은 바이트 단위 동일, 기존 executor는 `antara_*` key를 쓰지 않는다. CPU 회귀 run의 JSON schema는 GPU 보드 JSON과 key path 143개가 같다(차이는 generator가 넣는 meta 필드와 `inputs.video`뿐).
- **NPU server process:** instance당 driver handle 1개를 가진 spawn process. driver 호출(SendInput→Launch→ReceiveOutputs)만 거기서 하고, 전처리·front quantize ONNX·host FC 또는 dequant+back ONNX·후처리는 executor worker thread에서 한다. 데이터는 RawArray shared memory + Pipe.
  - 완료 시각은 기존 `Worker.run`이 output put 직후 기록 → service time에 host 단계와 IPC 왕복이 모두 포함된다.
  - 근거 측정(`code_notes/diag/`): 다른 thread에서 호출 시 "Failed to program DMA0 engine", 한 process에 2 driver 시 "Failed to sg_dma_read" 뒤 hang.
- **residency:** `executor_npu.check_placement`가 실행 전 확인(NPU당 ≤1 instance, 장치 종류 일치). pilot에서 위반 0.

### 2-2. 새 파일 (`revision_profiling/code/`)
| 파일 | 역할 |
|---|---|
| `executor_npu.py` | entry (위 3개 대체, 배치 검사) |
| `npu_server.py` | NPU server process와 `NpuClient` |
| `worker_npu.py` | `NpuResNet50Worker`, `NpuYoloV3Worker`, `YoloV3CpuWorker`, sidecar(`<out>.npu.<inst>.json`: driver 호출 ms, Close rc) |
| `gen_schedules_npu.py` | 11 family 후보를 읽고 rate만 새 규칙으로 교체. 독립 열거 대조, ×2/×3/×4 동일, workload 내 λ 벡터 동일 assert |
| `build_npu_registry.py` | registry block, YOLO dequant JSON, ResNet FC npz, sha256 목록 |
| `run_campaign_npu.py` | 순차 실행, resume, DMA32 free pages 기록과 하한(20,000) 중지 |
| `check_registry_sha.py`, `check_npu_sanity.py` | 파일 불변 확인, 정확도 sanity(판정에 안 씀) |
| `analyze_smoke_npu.py`, `offline_profile_npu.py`, `pilot_plan_npu.py`, `analyze_pilot_npu.py` | §3–§5 집계·판정 |

new_main/현재 branch 코드에서 참고한 부분은 각 docstring에 출처 파일·행을 적었다.

### 2-3. 기존 파일 diff [확인]
`git diff --stat`: `revision_profiling/code/model_registry.yaml | 109 +` — 1 file, 추가만. 추가 key: `antara_resnet50_cpu`, `antara_yolov3_cpu`, `antara_{resnet50,yolov3}_{big,small}_npu`.

### 2-4. Registry [확인] (`revision_profiling_npu/registry/`)
- CPU ONNX는 big/small sha256 동일 → model당 1개. NPU `.o` 4개는 sha256이 모두 다르다(assert). variant마다 자기 바이너리를 쓴다.
- driver: `../lib/NeublaDriver.cpython-38-x86_64-linux-gnu.so` (sha `92ad912c…`, 2025-03-31).
- YOLO dequant (scale, zp): 684 (0.11476, 183), 688 (0.12135, 188), 692 (0.12158, 196).
- ResNet FC: uint8 weight (1000, 2048), int32 bias, requant 파라미터를 host ONNX에서 추출.

### 2-5. 후보 집합 [확인]
N=1: rb, rs, yb, ys 각 3 / N=2: rbrs, rsyb, ybys 각 7 / N=3: rbrsyb, rbybys, rsybys 각 13 / N=4: rbrsybys 21 → **93**. YAML 집합 = 독립 열거 feasible 집합. 평가 family 4개는 60 후보.

### 2-6. driver 판본별 출력 안정성 [확인] (`code_notes/diag/driver_compare.txt`)
| driver | sha | 안정 (12회) | ResNet big / small ms | YOLO big / small ms |
|---|---|---|---|---|
| repo 사본 | 5a43… | 6/12 (홀수 호출 깨짐) | 42.2 / 45.5 | 95.8 / 67.6 |
| certification | 7175… | 6/12 | 49.4 / 45.8 | 94.7 / 68.1 |
| `/workspace/lib` (**채택**) | 92ad… | 12/12, 2-NPU 동시 안정 | 42.3 / 38.4 | 88.6 / 61.6 |

## 3. Smoke (§3) [확인] — `smoke/SMOKE.md`

1 instance, 2 req/s, warm-up 5 + window 10 s, 10 run. 전부 valid, conservation (gen, flow) = 0, trace_check ok, failed 0, Close rc 0. registry sha256 전후 동일.

| service p50 (ms, end-to-end) | npu0 | npu1 |
|---|---|---|
| ResNet50 big | 44.71 | 45.19 |
| ResNet50 small | 40.69 | 41.07 |
| YOLOv3 big | 98.02 | 105.86 |
| YOLOv3 small | 70.53 | 79.99 |

CPU: ResNet50 18.78, YOLOv3 283.60. big − small: ResNet +4.0 ms, YOLO +26–27 ms.

Sanity (`code_notes/sanity_npu.json`): ResNet NPU top-1 115/128 (모든 variant×NPU), CPU 113/128. YOLO NPU 751 box / 64 frame 전부 검출, CPU 756.

## 4. Isolated profile (§4) [확인] — `isolated/OFFLINE_PROFILE.md`

closed loop, 1 instance, warm-up 5 + window 30 s. 10 run 전부 valid.

| req/s | CPU | npu0 big | npu1 big | npu0 small | npu1 small |
|---|---|---|---|---|---|
| ResNet50 | **52.50** | 22.07 | 21.83 | 24.43 | 24.07 |
| YOLOv3 | 3.37 | 10.23 | 9.50 | **14.27** | 12.80 |

- C^max: resnet50 52.50 (CPU), yolov3 14.27 (npu0 / small).
- npu1/npu0 (Gen2×1 vs Gen4×8): YOLO 0.90–0.93, ResNet ≈0.99.
- ResNet50은 CPU가 NPU보다 2.1–2.4배 빠르다. YOLOv3은 NPU가 CPU보다 2.8–4.2배 빠르다.
- 기준 배치(실행 전 고정, `pilot/PILOT_FAMILIES.md`): rbybys R1 = `combination_13` (resnet50_big:cpu, yolov3_big:npu1, yolov3_small:npu0), R2 = `combination_1` (all CPU). rbrsybys R1 = `combination_7` (resnet50 2개 cpu, yolov3_big:npu1, yolov3_small:npu0), R2 = `combination_1`.

## 5. Pilot (§5) [확인] — `pilot/REPORT_PILOT.md`, `pilot_runs.csv`, `pilot_criteria.csv`

rbybys 13 + rbrsybys 21 후보 × β {0.1, 0.25} × rate {1×, 4×} = 136 run + rbrsybys R1 2× 반복 3회 × β 2 = 6 run. **142 run 전부 rc 0 / ok, rejected 0.** 실행 시간 92.5 + 4.3 min, run당 평균 40.9 s.

| 기준 | β = 0.1 | β = 0.25 |
|---|---|---|
| (A) conservation 0, trace ok, failed 0 | PASS (0건) | PASS (0건) |
| (B) 후보 간 generated 편차 | PASS (최대 0.32 %) | PASS (최대 0.19 %) |
| (C1) 1× R1: D_macro ≤ 0.02, 완료율 ≥ 0.95 | PASS (0.0000, 1.000) | PASS (0.0000, 1.000) |
| (C2) 4× R1: q_i ≥ 0.05인 instance 존재 | **FAIL** (max q_i 0.0000, 0.0000) | PASS (0.3668, 0.4065) |
| (D) 최저 rate generated ≥ 30 | PASS (42) | PASS (107) |
| (E) realization ≥ 0.95 | PASS (0.9968) | PASS (0.9981) |
| (F) residency 위반·load 실패 0 | PASS | PASS |
| (G, 참고) 반복 F 범위 ≤ 1 %, D 범위 ≤ 0.01 | PASS (0, 0) | PASS (0, 0) |

**선택: β = 0.25** (규칙: A·B·F 통과 β 중 C1·C2·D·E를 통과하는 가장 작은 β).

참고(판정 외):
- β=0.25에서 drop이 있는 후보: 1×에서 rbybys 11/13, rbrsybys 19/21; 4×에서 전부. R2(all CPU) D_macro 1× 0.25/0.23, 4× 0.74/0.73.
- β=0.1 4×에서는 R1이 전혀 막히지 않는다(q_i 0). β=0.1은 1–4× 범위 대부분이 "여유" 구간이라 학습 데이터의 부하 다양성이 작다.

## 6. [추정] / [확인 불가]

- [추정] **반복 측정(G)이 정보가 적다.** 반복 조건(rbrsybys R1, 2×)은 두 β 모두 drop 0·q_i 0이라 F = 생성률 그대로이고 범위가 0이 된다. 혼잡 상태의 재현성은 이 pilot으로 확인되지 않았다 [확인 불가].
- [추정] 3× 평가 rate(β=0.25)에서 R1은 2×(drop 0)와 4×(q_i 0.37–0.41) 사이에 있어 판별 가능한 구간일 것이다. 3×는 실측하지 않았다.
- [추정] 재부팅 전 DMA32 고갈의 원인은 실패한 LoadModel이 반복되며 쌓인 회수 불가 커널 할당이다. 재부팅 후 정상 운영 142 run에서는 감소 추세가 없었다(시작 179,092 → 최저 176,442 → 끝 178,054 page). campaign에서는 `--dma32-min 20000` 하한으로 감시한다.
- [추정] YOLO dequant 파라미터는 host 양자화 모델이 1개뿐이라 big/small 공통으로 썼다. sanity 결과(751 vs CPU 756 box)는 이와 모순되지 않는다.
- [확인 불가] 1월 데이터 측정 당시 쓰인 driver 판본. repo 사본(5a43…)으로 추정한다.

## 7. 사용자 결정 필요

1. **β = 0.25 확정** 여부 (사전 규칙에 따른 선택).
2. **반복 측정 포함 여부:** pilot 반복은 비혼잡 조건이라 분산 0. campaign에 반복을 넣는다면 혼잡 조건(예: 4× R1)에서 하는 것이 정보가 있다 [추정].
3. **sudo:** 측정에는 불필요. 재부팅 뒤 NPU 장치 생성(`hotreset.sh`)에만 필요하다.
4. **driver 교체와 1월 데이터:** 1월 데이터는 출력 절반이 깨지는 driver로 측정됐을 가능성이 높다(§2-6). 처리 시간도 판본에 따라 다르다(ResNet small driver call 38.5 ms vs 이전 driver 48.0 ms). 원고·답변서 반영 방식.
5. **YOLOv3 정의:** CPU YOLOv3(keras, opset 10, NMS 내장)와 NPU YOLOv3(mmdet yolov3_d53)는 서로 다른 네트워크다. 현재 구성 유지(원고 설명 보완) 또는 변경.

## 8. 불변 확인 [확인] (pilot 종료 후 17:05 KST)

- **(A) 현재 branch (`ubuntu_gpu_fsrr_npu`):** tracked 수정은 작업 전과 같은 3개(pyc 2개, `ANTARA_NPU_DOCKER_RUN_GUIDE.md`). 가이드 파일은 1행 앞에 문자 `1`이 들어간 상태(14:58:52 수정, 이번 작업의 명령이 쓰지 않음 — 편집기 입력으로 추정, 되돌리지 않음).
- **`new_main` worktree:** `git status` 0행.
- **저장소 밖 파일:** 쓰지 않았다. registry 등록 파일 21개 sha256이 smoke 전 스냅샷과 pilot 후 동일(`pilot/sha_after_pilot.json` vs `smoke/sha_before.json`: changed none). 커널 쪽 조치는 재부팅 전 `drop_caches`/`compact_memory`와 hotreset뿐.
- **port worktree:** tracked 수정 `model_registry.yaml` 1개(+109). untracked는 새 code 파일 12개, `revision_profiling_npu/`, `imagenet-sample-images` symlink.
- 실행 중인 컨테이너 0. Phase N2(campaign, 학습)는 시작하지 않았다.
