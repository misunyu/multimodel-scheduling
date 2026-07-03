# EXP-CPULOAD — LM vs VLM 호스트 CPU 사용률 측정 (인과 확정/반증)

목적: "LM이 host CPU(NPU 후처리 실행)와 경쟁해 NPU skip을 76%로 올린다"는 본문 인과를 직접 측정으로
확정/반증. All-NPU N=4, threads=4, co-tenant만 변경(rev20 동일 구성). psutil CPU 샘플러 + pynvml GPU.

## 측정 환경
- 2026-06-18, RTX 5090 + Mobilint MLA100, host 24-core. 단독 측정(시작·종료 시 GPU compute 0).
- 구성: foreground All-NPU N=4 [2,22,3,21], NPU single-mode(legacy mxq), **threads=4**(rev12 24-thread
  오염 반복 안 함). co-tenant = rev20과 동일: L0(없음) / L2_lm(ResNet50+TinyLLaMA, ONNX Runtime) /
  L3_vlm(ResNet50+Qwen2-VL, PyTorch CUDA). 평가창 ~13s(LM은 ~19s, 느려서 길어짐), 3반복.
- 샘플러: psutil 100ms 주기(시스템 평균 CPU%, per-core, 프로세스 CPU%, GPU util). 스크립트:
  `accv_experiments/scripts/phase_cpuload.py`. raw: `cpuload_raw.csv`.

## 결과 (조건 평균, 3반복)
| 조건 | NPU skip | NPU 후처리 latency | CPU sys (mean) | CPU sys p99 | CPU process (mean) | GPU util |
|---|---|---|---|---|---|---|
| baseline (co-tenant 없음) | **0.6%** | 18.3 ms | 16.6% | ~24% | 380% (~3.8 core) | 2.8% |
| **+LM** (TinyLLaMA) | **79.0%** | **57.0 ms** | **92.6%** | 100% | **2213%** (~22 core) | 49.5% |
| **+VLM** (Qwen2-VL) | **0.7%** | 17.7 ms | 20.7% | ~29% | 489% (~4.9 core) | 82.3% |

(per-core p99는 세 조건 모두 100% — 최소 1개 코어는 NPU 후처리로 항상 포화. 비교 지표는 system mean.)

## Gate — skip 재현 PASS
baseline 0.6% / +LM 79.0% / +VLM 0.7% → Table 4의 LM 76% / VLM 1%를 재현(구성 일치 확인). 측정 유효.

## 판정: **인과 확정 (causal confirmed)**
- **+LM**: system CPU가 16.6%→**92.6%**로 거의 포화(24코어), process CPU 2213%(~22코어 점유). 그 결과
  NPU 후처리 latency가 18.3→**57.0 ms**로 33.3 ms budget을 초과 → **skip 79% 재현**. 즉 host CPU 경쟁이
  NPU 후처리를 지연시켜 skip을 올린다는 인과가 직접 수치로 성립.
- **+VLM**: system CPU는 20.7%로 baseline(16.6%)과 거의 같고(낮음), GPU util 82%(GPU-resident). NPU 후처리
  latency 17.7 ms(budget 이내) 유지 → **skip 0.7%**. VLM은 host CPU를 거의 안 써 NPU 후처리가 한가 → 후보
  (a) 확정.
- **Q3**: "LM이 VLM보다 host CPU를 더 쓴다" → **수치로 성립**. system CPU LM 92.6% vs VLM 20.7%(약 4.5×),
  process CPU 2213% vs 489%(약 4.5×). 반증 아님.

## 한 문단 요약
baseline/LM/VLM의 system CPU util = 16.6% / 92.6% / 20.7%, process CPU = 380% / 2213% / 489%, NPU 후처리
latency = 18.3 / 57.0 / 17.7 ms, NPU skip = 0.6 / 79.0 / 0.7 %. **따라서 CPU 경쟁이 NPU skip 79%의 원인이
맞다**: LM은 host CPU를 포화시켜 NPU 후처리 latency를 budget 초과로 끌어올려 skip을 유발하고, VLM은
GPU-resident라 host CPU를 거의 쓰지 않아 NPU 후처리가 영향을 받지 않는다. 본문(361,458)의 인과 서술은
측정으로 뒷받침된다(직접 측정, 추론 아님).

## 산출물 / 준수
`cpuload_raw.csv`(9행), `cpuload_means.json`, `cpuload_stdout.log`, 스크립트 phase_cpuload.py. 구성을
rev20과 동일하게 유지(skip 재현이 게이트), threads=4 고정, 사전등록 샘플링값 고정. 논문 .tex 미수정.
