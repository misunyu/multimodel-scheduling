# GPU(sm_120) vision 실측 검증 보고

날짜: 2026-07-21 · 성격: **검증만(코드·환경 불변)** · 대상 GPU: RTX 5090 (Blackwell, sm_120)

## 한 줄 결론

**onnxruntime-gpu는 sm_120에서 vision conv를 실제로 GPU 실행한다(증명됨, 3~7x 가속).** fsrr에서
관측된 "조용한 CPU 폴백"은 onnxruntime 버전 문제가 아니라 **같은 프로세스에서 torch와
onnxruntime-gpu가 서로 다른 cuDNN을 로드해 충돌**하기 때문이다. → onnxruntime 유지, **vision 워커를
torch로부터 분리**하는 것이 해법(torch 전환 불필요).

## 환경 특정 (§1)

| venv | onnxruntime | providers | 비고 |
|---|---|---|---|
| `multimodel-scheduling-video/.venv` | **gpu 1.20.1** | TensorRT, CUDA, CPU | GPU 신호의 출처 |
| `multimodel-scheduling-mobilint/.venv` | **gpu 1.20.1** | TensorRT, CUDA, CPU | 동일 |
| `fsrr/.venv` | **gpu 1.23.2** | TensorRT, CUDA, CPU | 현행 |
| `MobilintTest/.venv` | onnxruntime(CPU) 1.23.2 | Azure, CPU | (이전 진단이 잘못 참조한 venv) |

- 이전 진단이 "mobilint는 CPU 전용"이라 한 것은 **엉뚱한 venv(MobilintTest)**를 본 오류. 실제
  mobilint/video venv는 onnxruntime-**gpu**를 갖고 있다.
- 모든 venv에 torch 2.12.0+cu130 (sm_120 지원).

## 실측 결과 (§2~4)

### A. CUDA-only 세션 생성 (1.20.1, video)
- yolo11s / resnet50 / mobilenet_v2 **모두 성공** — 로드 시 conv 미지원 오류 없음.

### B. 노드 provider 프로파일 (resnet50, torch **미로드**)
| venv | 전체 노드 | Conv 노드 |
|---|---|---|
| video 1.20.1 | 610 CUDA | **265 CUDA** |
| fsrr 1.23.2 | 610 CUDA | **265 CUDA** |
- **두 버전 모두** conv 포함 전 노드가 CUDA. cuDNN 오류 없음.

### C. GPU vs CPU 속도 (워밍업 후 반복 평균, torch **미로드**)
| 모델 | GPU | CPU | speedup |
|---|---|---|---|
| yolo11s | 3.86ms | 24.08ms | **6.25x** |
| resnet50 | 2.71ms | 9.47ms | **3.49x** |
| mobilenet_v2 | 1.06ms | 1.10ms | 1.04x (모델이 작아 오버헤드 지배) |
- fsrr 1.23.2도 동일 조건에서 yolo11s 7.29x / resnet50 3.11x. **CPU 폴백으로는 불가능한 가속 → 실제 GPU 실행 확정.**

### D. 결정적 대조 — torch를 **먼저 로드**했을 때 (fsrr 실제 런타임 재현)
| venv | 현상 |
|---|---|
| **fsrr 1.23.2** | `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` → Conv 265개 **전부 CPU**, speedup **0.47x**(GPU가 CPU보다 느림) |
| **video 1.20.1** | `RuntimeError: cuDNN version incompatibility: PyTorch compiled against (9,20,0) but found runtime (9,12,0)` — torch가 예외 |

- **핵심**: onnxruntime-gpu 1.23.2는 cuDNN 9.x(≈)를, torch는 cuDNN **9.20**을 로드. 한 프로세스에서
  둘이 만나면 cuDNN 서브라이브러리 버전이 어긋난다.
  - torch 먼저 → onnxruntime이 9.20을 물어 `SUBLIBRARY_VERSION_MISMATCH` → onnxruntime **CPU 폴백**.
  - onnxruntime(1.20.1, cuDNN 9.12) 먼저 → torch가 9.12를 물어 torch 예외.

## 근본 원인

fsrr vision 워커는 `_limit_cpu_threads()`(`import torch`, model_processors.py:107)와
`_yolo_decode()`(`from torchvision.ops import nms`, :200)로 **torch를 로드**한다. 워커가
`threading.Thread`라 onnxruntime-gpu와 **동일 프로세스**를 공유 → cuDNN 충돌 → conv가 조용히 CPU로
폴백. (mobilint는 워커를 `multiprocessing.Process`로 돌려 vision 워커 프로세스에 torch가 없어 충돌이
없다.)

## 판정 (§5)

- onnxruntime GPU vision: **동작(PASS)**. torch를 co-load하지 않는 한 sm_120 conv를 GPU에서 실행.
- fsrr 현행 런타임: **CPU 폴백(FAIL)** — 원인은 onnxruntime 버전이 아니라 **torch 공존**.
- 따라서 **torch 경로 전환은 불필요**. onnxruntime-gpu를 유지하되 vision 워커를 torch로부터 분리한다.

## 권장 (다음 작업, 이 보고는 검증만)

1. **vision 워커를 torch-free로** (권장, 최소 변경, Thread 유지):
   - `_limit_cpu_threads()`: torch 대신 `OMP_NUM_THREADS`/onnxruntime `intra_op_num_threads`로 스레드 제한.
   - `_yolo_decode()`: `torchvision.ops.nms` 대신 **numpy NMS**(현재 fallback은 naive top-300이라 정식 IoU NMS로 교체 필요).
   - 효과: vision 워커 프로세스에 torch 미로드 → onnxruntime-gpu가 conv를 GPU 실행. **onnxruntime 다운그레이드·torch 전환 불필요.**
2. **혼합 셋(vision+LLM) 대비**: LLM/VLM 워커는 torch가 필수다. vision(onnxruntime)과 LLM(torch)이
   같은 프로세스 스레드로 공존하면 다시 충돌한다. → 장기적으로 **워커를 프로세스 분리**(mobilint 방식)
   하거나, GPU에서 vision·LLM을 같은 프로세스에 두지 않는 배치가 필요.
3. **조용한 폴백을 에러로 승격**(오염 방지): vision GPU 워커에서 세션 생성 후 실제 첫 추론의 conv가
   CUDA인지 확인하고, CPU로 떨어지면 **로그 경고 또는 예외**로 승격.

## 변경 없음

이 작업은 실측 검증만 수행했다. 코드·환경·legacy·백업 불변.

---

# Vision 워커 torch-free 전환 결과 (2026-07-21, 추가)

방침: onnxruntime-gpu 유지, **vision 런타임 프로세스에서 torch 완전 배제** → cuDNN 충돌 제거로 GPU 복구.

## 변경 (model_processors.py, unified_viewer.py)
- **`_yolo_decode`**: `torchvision.ops.nms` → **numpy greedy IoU NMS**(`_nms_numpy`). torchvision 알고리즘
  정확 복제(score desc stable 정렬, IoU>thr 억제, area=(x2-x1)(y2-y1)).
- **`_limit_cpu_threads`**: `torch.set_num_threads(1)` 제거 → onnxruntime `SessionOptions.intra_op_num_threads=1`
  (`_ort_session`) + `OMP/OPENBLAS/MKL_NUM_THREADS` env. torch 미사용.
- **`unified_viewer.py`의 torch 기반 NPU 가용성 체크 제거** — 프로세스에 torch를 올려 onnxruntime-gpu를
  오염시키던 구식 코드(NPU는 이제 워커에서 mblt로 처리).
- **가드 승격**: `_assert_gpu_clean(device)` — GPU 워커 진입 시 `torch in sys.modules`면 예외(조용한 CPU
  폴백 금지). `_ort_session`의 CUDA EP 부재 예외 유지.

## 검증
| 항목 | 결과 |
|---|---|
| 7a NMS parity vs torchvision.ops.nms | **200/200 케이스 + tie 케이스 완전 동일** |
| 7b GPU 실행 (실제 워커) | torch 미로드 확인. detection **GPU 18.75ms vs CPU 166.47ms (≈8.9x)**, classify GPU 2.47ms. 폴백 0.47x **사라짐** |
| 가드 승격 | torch 선로드 시 GPU 워커가 **RuntimeError**(조용한 폴백 아님) |
| NPU 무관 | .mxq detection 정상(3-tuple) — torch(mblt)는 onnxruntime-gpu 없어 무해 |
| 계약·회귀 | 컴파일 OK, 결과 튜플/ready_event 유지, 예측기(xgboost) 정상, legacy/scripts/backup 무변경 |

## 판정
- **vision-only 셋 = GPU 정상 복구(PASS).** onnxruntime-gpu가 sm_120 conv를 실제 GPU 실행.
- **mixed 셋(LLM/VLM) = 이 지시문 밖(후속).** LLM 워커는 torch 필수 → vision(onnxruntime)과 같은
  프로세스 스레드로 공존하면 cuDNN 재충돌. 근본 해법은 **워커 프로세스 분리**(mobilint 방식) — 별도 후속.

## 잔여 (사람 판단)
- OMP env는 워커 진입 시 설정이라 numpy BLAS엔 늦을 수 있음(온전한 적용은 프로세스 시작 전 설정 필요).
  단 주 연산은 onnxruntime(intra_op=1로 고정)이고 torch 과구독은 완전 제거돼 실무상 영향 미미.
- mixed 셋 프로세스 분리(후속 지시문).
