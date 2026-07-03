# EXP-AUDIT-CPULOAD — LM vs VLM 호스트 CPU 사용량 / NPU skip 차이 (read-only)

질문: All-NPU에서 NPU frame skip이 LM=76% vs VLM=1%인데, 이게 "LM이 호스트 CPU(NPU 후처리 실행)를 더
많이 써서"인지 저장 데이터로 확인. 측정 재실행 없음.

## Q1 — NPU skip 확인 (직접, 확정)
산출원: `rev20_5strat_heavybg.csv` (threads=4 고정). All-NPU(NNNN), N=4, 3 reps:
- **L2_lm (LM)**: npu_skip = 78.8 / 73.8 / 74.9 % → ~**76%**
- **L3_vlm (VLM)**: 0.9 / 1.0 / 0.7 % → ~**1%**
- L1_light: 1.1 / 1.1 / 1.0 % → ~1%
→ 본문 LM 76% / VLM 1%와 일치(확정).

## Q2 — 호스트 CPU 사용률: **직접 측정 기록 없음**
`psutil` / `cpu_percent` / `cpu_times` / `cpu_util` 사용처가 코드·로그 어디에도 없다(grep 0). co-tenant별
**host CPU 사용률을 직접 기록한 데이터가 없다.** (기록된 util은 일부 CSV의 `gpu_util_start`뿐 — GPU용.)
→ "LM이 CPU를 더 쓴다"는 직접 수치로 확정 불가. 아래는 간접 증거.

## Q4(간접) — NPU 후처리 latency: co-tenant별 깨끗한 비교본 부재
- `rev9_partA_npu.csv`: All-NPU NPU latency가 있으나 **L1_light 한 레벨만**(latency_mean ~12ms, skip ~0.6%).
  L2_lm/L3_vlm 행이 없어 LM vs VLM latency 직접 비교 불가.
- `rev12_l2lm_single.csv`: L2_lm NPU 측정이 있으나 **24-thread misconfig**(npu_lat~150ms, skip~100%; rev19
  thread audit가 오염으로 분류) → 깨끗한 threads=4 비교에 부적합.
→ 따라서 co-tenant별 NPU **후처리 latency 직접 비교**도 저장 데이터로는 불가. 단, NPU skip 자체가 eff
  delivery latency>33.3ms의 비율이므로, **skip 76% vs 1%는 "LM이 NPU 전달 지연을 budget 초과로 끌어올렸고,
  VLM은 안 그랬다"는 간접 증거**다(latency를 직접 수치로 보이지는 못함).

## Q3 — VLM skip이 1%로 낮은 이유: 후보 (a), 단 간접
코드 단서(`scripts/step_h2_robustness.py`, `_step_d_common.py`):
- **LM(L2_lm)** = ResNet50 + **TinyLLaMA → ONNX Runtime**(`InferenceSession`, CUDA+CPU EP). ORT LLM 경로는
  host-CPU 관여가 큰 편(로그에 "45 Memcpy nodes are added ... CUDAExecutionProvider" 경고 — host↔device
  복사/CPU fallback op 다수).
- **VLM(L3_vlm)** = ResNet50 + **Qwen2-VL → PyTorch CUDA**(bfloat16, `.to("cuda")`) — GPU-resident, host-CPU
  관여 상대적으로 적음.
NPU 후처리(dequant/decode/NMS)는 host의 torch CPU 스레드(threads=4)에서 돈다. LM이 host CPU를 더 점유하면
이 스레드가 느려져 skip↑ — 이것이 본문의 메커니즘이다. 데이터의 skip 차이(76% vs 1%)와 코드 특성(LM=CPU
관여 큰 ORT-LLM, VLM=GPU-resident PyTorch)은 후보 **(a)**(VLM은 host CPU를 거의 안 써 NPU 후처리가 한가)에
부합하나, **CPU util을 직접 재지 않았으므로 (a)는 간접 추론**이다.

## 요약 (직접/간접 구분)
- **직접 확정**: All-NPU NPU frame skip = LM ~76% / VLM ~1% / L1 ~1% (rev20, threads=4). LM은 NPU 프레임
  전달을 budget 초과로 지연시키고, VLM은 그렇지 않다.
- **간접(추론)**: "LM이 VLM보다 host CPU를 더 쓴다"는 **직접 측정값이 없다**(psutil/CPU util 미기록).
  skip 차이 + co-tenant 실행 방식(LM=TinyLLaMA on ONNX Runtime, host-CPU 관여 큼; VLM=Qwen2-VL on PyTorch
  CUDA, GPU-resident)이라는 코드 근거로 **추정**되는 것이며, "CPU를 더 쓴다"는 확정이 아니다.
- **미상**: co-tenant별 host CPU 사용률, 그리고 깨끗한 threads=4에서의 LM vs VLM NPU 후처리 latency 수치는
  저장 데이터에 없다(rev9=L1만, rev12 L2_lm=thread misconfig).

## 준수
읽기 전용; 저장 CSV/로그/코드만; 재측정·프로파일링 없음; CPU util 직접 기록 부재를 명시하고 "LM이 CPU를 더
씀"은 간접 추론으로 표기; 논문 .tex 미수정. 산출물 = 이 파일.
