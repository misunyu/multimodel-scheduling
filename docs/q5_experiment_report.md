# Q5 — CPU–NPU 플랫폼 일반화 (오예측 회복)

날짜: 2026-07-24 · 논문 Q5: *"Does BoundGuard generalize to constrained heterogeneous platforms (CPU–NPU)?"*
알고리즘·파라미터 불변 · main.tex 불변 · 관련: [q3_experiment_report.md](q3_experiment_report.md), [mixed_exec_report.md](mixed_exec_report.md)

> **Q5의 실질 과제 = Q3(오예측 회복)의 CPU–NPU 재현.** Q1(§4b)·Q1.4(fluid)·Q1.5(B2+스윕)는 이미
> 양 플랫폼 데이터가 있으므로 Q5에서 인용한다. 새로 필요한 것은 NPU background LLM 하의 Q3다.

## §1. NPU background LLM — 구현·DRAM 경험적 판정

- **하드웨어**: `/dev/aries0`(MLA100), `qbruntime v1.2.0`, `mblt_model_zoo` 확인.
- **NPU LLM background**: `runtime/bg_entry.py --device npu`로 `mobilint/Llama-3.2-1B-Instruct` **W8 `.mxq`**
  적재(**7.9s**) → 연속 생성 → SIGTERM 정상 종료. 단일 코어 고정(`NPU_LLM_CORES`, 정적 프로파일과 일관).
- **DRAM 동시 적재 (경험적)**: 공개 조회 API가 없어 실제 적재로 판정. **vision 3개(yolo11s/11m/resnet50)
  NPU 워커 + llama1b NPU** 를 aries0에 **동시 적재 성공**(LLM ready ~8s), co-load 중에도 세 vision이 모두
  추론(output_qsize>0). → **DRAM 여유 확인, foreground 축소 불필요.**
- **거친 전환**: `BackgroundManager`가 NPU 경로에서 종료→기동 + ready 게이팅 동작(사전측정에서
  `stop llama1b/npu → start llama1b/cpu` 로그로 확인).

## §2. 오예측 구조 확인 (게이트) — **충족**

### 2.1 cpu_npu 예측기 랭킹
워킹셋: llama1b(bg) + yolo11s/resnet50/mobilenet_v2(foreground). N_cand=5.

| rank | combo (llama,yolo,res,mob) | llama |
|---|---|---|
| 1 | nnnn | **NPU** (top-1) |
| **2** | **cnnn** | **CPU** ← LLM-CPU 대안 |
| 3 | nncn | NPU |
| 4 | ncnn | NPU |
| 5 | nnnc | NPU |

**LLM-CPU 후보(cnnn)가 2위 ≤ N_cand=5 ✓.** GPU와 동일한 목적함수 불일치: 예측기가 β·y3(생성 처리량)를
보상해 LLM을 NPU에 두려 하나, V(t)는 vision만 측정한다.

### 2.2 top-1 자연 실패 + LLM-CPU feasible (λ=80, buffer=12, 사전측정)
| 배치 | V | backlog(max) | vision fps |
|---|---|---|---|
| **nnnn** (top-1, LLM=NPU) | **33.9–49.3 (위반)** | 36 | ~111 |
| **cnnn** (LLM=CPU) | **0.0 (feasible)** | 2 | ~487 (**4.4×**) |

LLM이 NPU에서 vision과 경합 → vision 위반. LLM을 CPU로 빼면 NPU가 vision 전용이 되어 feasible.
**P8 충족**(V 34–49 ≫ ε=1 vs V=0, backlog≈0 — 경계 아님). **오예측 조작 없음**(예측기 top-1이 실제로 실패).

### 2.3 게이트 판정: **세 조건 모두 충족 → §3 진행.**

## §3. 본 실험 (네 기법) — 그림 `docs/figures/q5_npu_generalization.pdf`

워킹셋: llama1b(NPU background) + yolo11s/resnet50/mobilenet_v2(NPU foreground). λ=80, buffer=12,
시간대응. top5 = `[nnnn, cnnn, nncn, ncnn, nnnc]` (rank.py와 일치).

| 기법 | maxV | lastV | persist(V>ε) | 회복 |
|---|---|---|---|---|
| Static | 77.50 | 33.68 | 158s | ✗ |
| Stop-restart | 36.86 | 32.11 | 114s | ✗ (재시작 없음 — top-1=burst 배치) |
| Adaptive | 37.11 | 31.97 | 114s | ✗ (top-1 오예측 고착) |
| **BoundGuard** | 33.68 | **0.00** | **15s** | **✓** |

**BoundGuard 판정 경로 (P7)**:
```
cand_1 (nnnn, top-1)  placement == violating burst → auto-advance (미측정)
cand_2 (cnnn, LLM→CPU)  [bg-llm] stop llama1b/npu → start llama1b/cpu (거친 전환)
        V_postswap=0.051 ≤ ε=1.0, slope=-1.14/s, tail_drops={none}, service_rate=240fps
        → commit-and-stay 회복
```
- **랭킹 2위(cnnn)에서 회복** — revert-to-best 불필요(예산 소진 전 commit). GPU Q3와 동일하게 목적함수
  불일치(예측기 β·y3가 LLM을 NPU에 두려 함 / V(t)는 vision만)에서 발생하고, LLM을 CPU로 빼면 NPU가
  vision 전용이 되어 회복한다.

**bound 분해·대조**:
- **T_valid**(위반→cand_2 적용) ≈ **9s** (burst 종료 +14s → cand_2 적용 +23s): cand_1 auto(~1s) +
  **δ(LLM NPU→CPU 거친 전환) ≈ 2.9s** + cand_2 T_v=3s + 여유.
- **δ 실측**: LLM CPU 적재 2869ms(= 전환 병목, 새 CPU 워커 기동). NPU `.mxq` 적재는 3516ms이나 **전환
  방향이 NPU→CPU**라 CPU 적재가 δ를 지배 — **GPU의 2.9s와 사실상 동일**(둘 다 CPU 적재가 병목).
- **bound**: `T + N_cand(T_v+δ) + δ` — cand_2에서 commit하므로 실제 탐색은 `T + 2·(T_v+δ)` ≈ 3+2·6 = 15s
  이내. **T_valid(9s) ≤ bound ✓.**
- **T_stable ≈ 0s**: cand_2 첫 샘플부터 V=0.00 — buffer=12로 상속 backlog가 작아 T-창 내 즉시 배출
  (GPU Q3와 동일한 소형-버퍼 T-창 지배; fluid는 대형-버퍼 조건).

**부가 관찰**: stable 구간(λ=25)에서도 V≈7 — LLM-on-NPU가 **저부하에서도 vision과 경합**한다. 예측기
top-1(nnnn)이 부하와 무관하게 나쁘다는 더 강한 진술이며, 오예측이 조작이 아님을 뒷받침한다.

## §4. GPU와의 대조 (Q5 "일반화" 서술 재료)

| 항목 | CPU–GPU (Q3) | CPU–NPU (Q5) | 같음/다름 |
|---|---|---|---|
| 오예측 메커니즘 | β·y3가 LLM을 GPU에 | β·y3가 LLM을 NPU에 | **같음** (목적함수 불일치) |
| baseline 회복 | ✗ (persist 105s) | ✗ (persist 114–158s) | **같음** |
| BoundGuard 회복 | ✓ (lastV 0.00) | ✓ (lastV 0.00) | **같음** |
| 회복 후보 위치 | cand_5 (LLM-CPU가 5위) | **cand_2** (LLM-CPU가 2위) | **다름** (NPU 랭킹이 더 유리) |
| 탐색 유계 | ✓ | ✓ | **같음** |
| δ (LLM→CPU 전환) | ~2.9s | ~2.9s | **같음** (CPU 적재 병목) |
| T_stable | ~4s (T-창) | ~0s (T-창) | 유사 (둘 다 소형 버퍼) |
| μ* (파이프라인) | ~137 fps | ~103 fps | **다름** (NPU 느림) |

- **핵심**: 회복 메커니즘·유계성·δ가 **동일**하다 → **일반화 성립**.
- **NPU가 더 유리했던 점**: LLM-CPU 대안이 랭킹 **2위**(GPU는 5위)라 BoundGuard가 **더 빨리** 도달
  (T_valid 9s vs GPU 18s). 즉 NPU에서 오예측 회복이 오히려 빠르다.
- **NPU 고유 제약의 실제 영향 (관측된 것만)**: DRAM 동시 적재는 vision3+LLM에서 **한계 없음**(§1).
  W8 uint8 양자화·단일 코어는 이번 워크로드에서 회복을 막지 않았다. `.mxq` 적재(3.5s)가 CPU 적재보다
  느리나 전환 방향상 δ에 미반영. (추정 없이 관측 사실만 기록.)

## §5. 논문 반영 메모

- **잘못된 하드웨어 서술 교체 (필수)**: Q5 본문의 *"NPU는 한 번에 한 모델만 실행하며 전환 시 reload가
  필요하다"* 는 **MLA100에 맞지 않는다**. §1에서 vision 3개 + LLM을 **동시 상주**시켜 실증했다. 제약은
  "단일 모델 상주"가 아니라 **DRAM 예산**이며, 본 워크로드에서는 여유가 있었다.
- **전환 스파이크 재서술**: 회복 시의 V 급강하는 "모델 reload"가 아니라 **background LLM의 거친 전환
  (NPU→CPU, δ≈2.9s) + vision 배치 불변**의 결과다(vision 3개는 nnnn·cnnn 양쪽에서 모두 NPU 상주 —
  hot-swap 없음).
- **Q5 결론**: BoundGuard의 오예측 회복이 CPU–NPU로 **일반화**된다. 메커니즘·유계성·δ가 GPU와 동일하고,
  NPU에서는 LLM-CPU 대안이 예측기 랭킹 상위(2위)라 회복이 더 빠르다.
- Q1(§4b)·Q1.4(fluid)·Q1.5(B2+스윕)는 이미 양 플랫폼 데이터 보유 → Q5에서 인용.

## 무결성
- 알고리즘·파라미터 불변(N_cand=5, α=0.3, β=0.5, ε=1.0, T=3, T_v=3, Δ=0.2, θ=0.5, 추세 판정·포화 가드·
  auto-advance·commit-and-stay·best-so-far·backlog 보존).
- 하네스 `q5_scenario.py`(cpu_npu 예측기, npu 디바이스, Table I L_SLO, cand_5 트리거, NPU background).
- legacy/backup·main.tex 불변.
