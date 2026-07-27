# Qwen2-VL을 background에 추가 — Q3/Q5 확장 (2-생성모델)

날짜: 2026-07-24 · 알고리즘·파라미터 불변 · main.tex 불변 · 기존 1-gen Q3/Q5 그림 보존
관련: [q3_experiment_report.md](q3_experiment_report.md), [q5_experiment_report.md](q5_experiment_report.md), [best_so_far_report.md](best_so_far_report.md)

> **결과 판정: 케이스 B** (§0 표). 생성모델을 둘로 늘리자 **회복에 필요한 배치(둘 다 CPU)가 예측기 랭킹
> 12위로 밀려나** N_cand=5 밖에 놓인다. 예측기가 β·y3(생성 처리량)를 **생성모델마다** 보상해, 둘을
> 가속기에 두는 것을 이중 보상하기 때문이다. → **"workload 규모가 커지면 유효 배치가 후보 예산 밖으로
> 밀려난다"는 알려진 한계의 실증.** Q3/Q5(회복)가 아니라 **Q4 성격(예산 소진 → best-so-far, 유계이나 미회복)**.

## §1. Qwen2-VL background 실행 확인

### 1a. CPU–NPU (DRAM 경험적 판정)
- `mobilint/Qwen2-VL-2B-Instruct` W8 `.mxq`(vision_transformer + text_model 두 모듈)를 `bg_entry --device npu`로
  적재(7.6s)·연속 생성·SIGTERM 종료 확인.
- **트랜스포머 모듈 캐시 권한 문제**: `~/.cache/huggingface/modules/transformers_modules/mobilint/Qwen2-VL...`가
  **root 소유**(5월 26 생성)라 trust_remote_code 커스텀 모듈을 쓸 수 없었다. → `HF_MODULES_CACHE`를
  쓰기 가능한 스크래치 경로로 지정해 해소(환경변수, 코드/파라미터 무관).
- **VLM 이미지 입력**: qwen2_vl은 이미지가 필수(None이면 mobilint 런타임 에러). background는 간섭 생성이
  목적이므로 `bg_entry`에 **더미 프레임(448×448 zeros)** 을 넘기도록 최소 패치. (LLM은 무시.)
- **DRAM 동시 적재**: vision 3개(yolo11s/11m/resnet50) NPU + **llama1b + qwen2_vl** 5개 모델이 aries0에
  **동시 적재 성공**(llama ~6s, qwen ~10s), co-load 중 세 vision 모두 추론(out_qsize>0). → **여유 확인,
  foreground 축소 불필요.** (core-slot은 배치 힌트이고 실제 제약은 DRAM임을 확인 — 5+ 슬롯 요구에도 적재됨.)

### 1b. CPU–GPU (경로·vision 무회귀)
- qwen2_vl-gpu는 conv3d에서 **cuDNN sublibrary version mismatch** 발생. → `bg_entry`에서 VLM+GPU일 때
  **cuDNN 비활성(native conv fallback)**. bg_entry는 torch 격리 프로세스라 vision(onnxruntime, 별 프로세스)에
  무영향. 적재 6.6s, 정상 생성. **경로: native conv (non-cuDNN).**
- **vision 무회귀 확인**: vision 3개 GPU + llama1b + qwen2_vl GPU co-load에서 **vision onnxruntime CUDA
  오류·폴백 0건**, vision 추론 정상(out_qsize>0). **격리 유지.**

### 1c. 거친 전환
- `BackgroundManager`가 두 생성모델(llama1b, qwen2_vl)을 후보 배치의 bit에 따라 독립 전환(종료→기동).
  본 실험(§3)의 후보 순회 로그로 확인.

## §2. 게이트 — 랭킹·오예측 구조 + A/B/C 판정

### 2.1 예측기 랭킹 (워킹셋: llama1b + qwen2_vl + yolo11s/resnet50/mobilenet_v2)
순서 [llama, qwen, yolo, res, mob], a=accel c=cpu. top-8:

| rank | CPU–NPU | CPU–GPU | 비고 |
|---|---|---|---|
| 1 | aaaac | aaaaa | 둘 다 가속기 (top-1) |
| 2 | aaaaa | aaaac | 둘 다 가속기 |
| **3** | **acaaa** | **acaaa** | qwen→CPU (gen 하나만) |
| 4 | aaaca | caaaa | |
| 5 | caaaa | aaaca | llama→CPU (gen 하나만) |
| **12** | **ccaaa** | **ccaaa** | **둘 다 CPU (회복 배치)** |

- **단일 gen-CPU 후보는 rank 3–5**(N_cand=5 이내), **둘 다 CPU는 rank 12**(밖). 양 플랫폼 동일 구조.

### 2.2 사전측정 (top-1 실패 + 어느 배치가 feasible한가)
**NPU (λ=80)**:
| 배치 | 랭킹 | V | fps | 판정 |
|---|---|---|---|---|
| both_accel (top-1) | 1 | 51→59 | 26 | 위반 |
| qwen_cpu (gen 하나) | 3 | 23→26 | 53 | **여전히 위반** |
| llama_cpu (gen 하나) | 5 | 0→40 | 127 | **여전히 위반** |
| **both_cpu (gen 둘)** | **12** | **0→0** | 176 | **feasible** |

**GPU (λ=90)**: 동일 구조(온건): both_accel 3.4, qwen_cpu 1.7, llama_cpu 1.7 (전부 >ε), **both_cpu 0.0(feasible)**.

- **하나만 옮겨선 회복 불가**(NPU V=26/40, GPU V=1.7 모두 >ε). **둘 다 CPU여야 feasible**, 그 배치는 rank 12.
- **P8**: feasible(both_cpu V=0)과 top-5(전부 위반)가 분리 — 경계 아님. NPU가 극명, GPU는 온건.

### 2.3 A/B/C 판정: **케이스 B**
회복 배치(both_cpu)가 **N_cand=5 밖(rank 12)**. 오예측은 자연 발생(top-1 실패)하나 top-5 순회로 도달 불가.
→ **Q4 성격**: BoundGuard가 top-5를 소진하고 best-so-far로 복귀, **유계이나 미회복**. §3에서 실증.

## §3. 본 실험 (네 기법) — 케이스 B 실증

그림: `docs/figures/q5_npu_generalization_2gen.pdf`(NPU), `docs/figures/q3_misprediction_2gen.pdf`(GPU).
**기존 1-gen 그림 미변경.**

### 3.1 NPU (λ=80) — 명확
| 기법 | maxV | lastV | persist | 회복 |
|---|---|---|---|---|
| Static | 69.6 | 59.8 | 104s | ✗ |
| Stop-restart | 76.0 | 59.2 | 104s | ✗ |
| Adaptive | 71.8 | 54.5 | 105s | ✗ |
| **BoundGuard** | 75.9 | **24.9** | 80s | ✗ (유계·미회복) |

BoundGuard 판정 경로: cand_1~5 전부 위반(service_rate 18–59fps) → 소진 → **best-so-far 복귀(cand_3=acaaa,
59.4fps 최고)**. **lastV=24.9로 baseline(54–60) 대비 낮으나 ε 초과** — 회복배치(both_cpu, rank 12) 미도달.

### 3.2 GPU (λ=90) — 온건(구조 동일)
| 기법 | maxV | lastV | persist | 회복 |
|---|---|---|---|---|
| Static | 4.0 | 3.61 | 97s | ✗ |
| Stop-restart | 4.0 | 3.64 | 97s | ✗ |
| Adaptive | 6.0 | 3.47 | 97s | ✗ |
| **BoundGuard** | 34.1 | **3.39** | 68s | ✗ (유계·미회복) |

GPU는 파이프라인이 빨라 2-gen 경합이 약해 위반이 온건(V~3.5). BoundGuard best-so-far 복귀(cand_4, 171.8fps),
lastV=3.39. baseline과 차이가 작으나(3.39 vs 3.5–3.6) **구조는 동일**(전 후보 위반, 복귀, 미회복).

### 3.3 §1c 확인 — 두 생성모델 독립 전환
BoundGuard 로그에서 llama1b·qwen2_vl가 후보 bit에 따라 **각각 독립적으로** npu↔cpu 전환
(`start/stop llama1b`, `start/stop qwen2_vl`). 한 모델만 옮기는 후보, 둘 다 옮기는 후보 모두 정상 동작.

## §4. 대조 (1-gen Q3/Q5 → 2-gen)

| 항목 | 1-gen (llama1b) | 2-gen (llama1b + qwen2_vl) |
|---|---|---|
| 회복배치 랭킹 | **2위**(NPU cnnn) / 5위(GPU cggg) | **12위**(ccaaa, N_cand 밖) |
| 회복에 옮길 gen 수 | 1개 | **2개(둘 다)** |
| BoundGuard 결과 | **회복**(lastV 0.00) | **미회복**(lastV 24.9 NPU / 3.39 GPU) |
| 성격 | Q3/Q5 (회복) | **Q4 (유계·미회복)** |
| best-so-far | 미발동(회복 후보에서 commit) | **발동**(전 후보 실패 → 복귀) |

- **핵심**: 생성모델 하나 추가만으로 **회복배치가 랭킹 2위→12위로 밀려난다.** 예측기가 β·y3를 생성모델마다
  보상하므로, 생성 job이 늘수록 "생성 전부 가속기" 배치들이 상위를 점유하고 feasible(생성 전부 CPU) 배치가
  후보 예산 밖으로 밀린다.

## §5. 논문 반영 제안

**§0 표의 케이스 B로 처리.** workload 서술을 아래처럼 조정:

1. **Table I·workload 서술 유지**: llama1b·qwen2_vl는 예측기 학습·background 후보 집합의 일부. 다만
   **"실험별 활성 background를 명시"** — Q3/Q5(회복 시연)는 **llama1b 단독**, 본 확장은 **2-gen**.
2. **2-gen 결과는 한계의 실증으로 병기**(Q4 근처 또는 한계 논의): *"생성 workload가 늘면 유효 배치가 예측기
   랭킹에서 N_cand 밖으로 밀려나, BoundGuard는 탐색을 유계로 만들고(best-so-far) 영향을 baseline 이하로
   줄이지만 회복은 못 한다. 이는 N_cand·예측기 랭킹 품질에 대한 알려진 의존성의 직접 실증이며, N_cand를
   늘리거나 예측기의 β 항을 활성 생성 job 수로 정규화하면 완화될 수 있다(향후 과제)."*
3. **기존 Q3/Q5(1-gen) 결론 불변** — 회복 시연은 그대로 유효. 2-gen은 **덮지 않고 병기**.
4. **best-so-far의 가치 재확인**: 2-gen에서 BoundGuard가 baseline 이하(NPU 24.9 vs 54–60)로 내려온 것은
   [best_so_far_report.md](best_so_far_report.md)의 복귀 로직 덕. 회복 불가여도 **영향 최소화**는 유지된다.

## 무결성
- 알고리즘·파라미터 불변. 오예측 조작 없음(예측기 top-1이 실제 실패, 회복배치가 랭킹에서 밀려난 것도 예측기
  고유 거동). P8 준수.
- 코드 변경: `runtime/bg_entry.py`(VLM 더미 프레임 + GPU cuDNN 비활성). 스냅샷 `backup/bg_entry.pre_qwen.py`.
  legacy/backup·main.tex 불변. **기존 1-gen Q3/Q5 그림 미변경.**
