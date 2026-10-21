# Facts for §5.1 / §5.3 (CPU–NPU)

모든 수치에 출처를 적었다.
- `N2A` = `../mms_npu_results/revision_profiling_npu/REPORT_N2A.md`
- `N1` = 같은 폴더 `REPORT_N1.md`
- `ds` = 같은 폴더 `dataset/`
- `out` = `revision_exp/out_npu/`

## Platform and models

| 항목 | 값 | 출처 |
|---|---|---|
| Model sets (families) | 11 | `ds/README.md` |
| family 구성 | rb, rs, yb, ys, rbrs, rsyb, ybys, rbrsyb, rbybys, rsybys, rbrsybys (r = ResNet50, y = YOLOv3, b/s = big/small) | `ds/README.md` |
| feasible placements / rate | 93. N=1 3개(×4 family), N=2 7개(×3), N=3 13개(×3), N=4 21개(×1) | N1 §2-5, `ds/workload_completeness.csv` |
| residency | NPU당 instance 1개. NPU 2개(npu0 Gen4×8, npu1 Gen2×1) | N1 §2-2, §4 |
| devices | cpu, npu0, npu1 | `ds/README.md` |
| 변종 | big/small은 NPU에서 서로 다른 바이너리 4개를 쓴다. sha 확인 결과 sha256이 모두 다르고, campaign 전후 같다. CPU에서는 같은 ONNX다(model당 1개) | N1 §2-4, N2A §2-4 |
| driver | NeublaDriver `/workspace/lib` 판, sha `92ad912c…` (2025-03-31). 다른 두 판은 출력이 홀수 호출마다 깨졌다 | N1 §2-4, §2-6 |
| SDK 판본 | [확인 불가] 보고서·로그에 SDK 버전 문자열이 없다. 컨테이너 `neubla/antara`, Python 3.8.18 | N1 §1 |
| NPU 처리 경로 | NPU instance마다 별도 server process가 driver 호출(SendInput → Launch → ReceiveOutputs)을 한다. driver가 thread-affine이고 한 process에 2개를 둘 수 없기 때문이다. 전처리, front quantize ONNX, ResNet host FC, YOLO dequant+back ONNX, 후처리는 executor worker thread에서 한다. service time에 host 단계와 IPC 왕복이 들어 있다 | N1 요약 3, §2-2 |
| ResNet50 host FC | NPU 바이너리 출력이 avgpool feature 2048개라서, FC(1000×2048, uint8 weight)는 host에서 계산한다 | N1 요약 2, §2-4 |

## Rates and measurement

| 항목 | 값 | 출처 |
|---|---|---|
| β | 0.25 (pilot 규칙에 따른 선택) | N2 지시문 §0, `ds/README.md` |
| rate | λ_i = 0.25 × C^max × k, k ∈ {1,2,3,4}. 배치·변종과 무관하다 | `ds/README.md` |
| Δ | warm-up 5 s + window 30 s | `ds/README.md` |
| runs | 372 (rep0) + 120 (3× N≥3 60 후보 × 추가 2회) = **492** | N2A §0 |
| 실행 시간 | 327.4 min (5.46 h). run당 평균 39.9 s(overhead 포함). rep0 4.10 h, rep1 0.68 h, rep2 0.68 h. 중단 0, 재시도 0 | N2A §0, §1, `logs/campaign_sessions.log` |
| 무결성 | valid 492/492, conservation 0, 44/44 workloads 후보 100% | N2A §0 |
| split | 학습 1·2·4×: 33 workloads, **279 후보**. 평가 3×: 11 workloads, **93 후보**. primary N≥3: **4 workloads, 60 후보** | `out/logs/config_frozen.json`, `out/table3b.csv` |

**Isolated throughput** (closed loop, 1 instance, req/s): `ds/offline_profile.json`, N1 §4

| model | CPU | npu0 big | npu1 big | npu0 small | npu1 small | C^max |
|---|---|---|---|---|---|---|
| ResNet50 | **52.50** | 22.07 | 21.83 | 24.43 | 24.07 | 52.50 (CPU) |
| YOLOv3 | 3.37 | 10.23 | 9.50 | **14.27** | 12.80 | 14.27 (npu0, small) |

원고에는 "ResNet CPU 52.5 vs NPU 24; YOLO NPU 14.3 vs CPU 3.4"로 적는다.

## Evaluation workloads (3×, N≥3, α = 0.3, D = D_macro)

출처: `out/per_workload.csv`, `out/latency_vs_oracle.csv`

| workload | 후보 | Oracle = Ours = Latency-based 배치 | S | Best-of-5 평균 S |
|---|---|---|---|---|
| rbrsyb | 13 | combination_2: rb cpu, rs cpu, yb npu0 | 0.9868 | 0.7780 |
| rbybys | 13 | combination_10: rb cpu, yb npu0, ys npu1 | 0.9922 | 0.9080 |
| rsybys | 13 | combination_5: rs cpu, yb npu0, ys npu1 | 0.9922 | 0.9139 |
| rbrsybys | 21 | combination_5: rb cpu, rs cpu, yb npu0, ys npu1 | 0.9881 | 0.7110 |

- 4개 workload 모두에서 세 방법(Oracle, Ours, Latency-based)이 같은 배치를 골랐다.
- oracle 1·2위 차는 0.0095–0.0187이다. 3회 반복 각각의 oracle이 평균 label의 oracle과 4/4 같다(`out/top1_tie_stats.csv`).

## §5.1.2용 문장 (원고 반영 시 그대로)

"The CPU and NPU use different YOLOv3 implementations: a Keras-based FP32 ONNX model on the CPU and a vendor-provided quantized MMDetection Darknet-53 model on the NPU. We evaluate placement performance using these platform-specific executables."

## §7용 한계 한 문장 (초안)

"Because the CPU and NPU run different implementations of the models, the measured CPU–NPU performance gap should not be read as a pure hardware speed difference for an identical network."
