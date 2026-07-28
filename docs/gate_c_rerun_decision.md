# 게이트 C — 재실행 필요 여부 판정 (작업 85 개정)

작성 2026-07-28. 성격: 복원 + 재산출 + 판정. **재실행 없음. 스케줄 생성·계측 변경 없음(C-1이므로 불필요).**

---

## 판정: **C-1 — 30회 재실행 불필요**

3-vision 워킹셋(16후보) 정본 랭킹에서 회복 배치는 **rank 5**로 legacy와 동일하다.
**β는 이 시나리오의 후보 순서를 바꾸지 않는다.** 논문 Q3/Q6 수치는 그대로 선다.

---

## §1.1 — 32후보의 정체 확정

v23 정본 랭킹의 5모델 = **yolo11s, yolo11m, resnet50, mobilenet_v2, llama1b**
(`_provenance.workload.working_set` 직접 확인). 즉 **첫 번째 읽기**(비전 4 + llama1b, §IV-A 서술과
일치하나 **실행된 적 없는** 워킹셋)다. `qwen2_vl`을 포함한 두 번째 읽기가 아니다 — 그래서 게이트 A의
제약 제거가 이 랭킹들의 순위를 바꾸지 않았다(§0 재확인에서 전 항목 불변).

**따라서 개정 §1의 지적이 옳다**: "5→4"는 β 효과가 아니라 **워킹셋 차이**(16후보 vs 32후보)였다.
v23 리포트 §98-1 표의 열 대조는 무효이며, 아래 §4에서 워킹셋을 명기해 정정한다.

---

## §2 — 미결 2 처리: **분기 2-A (복원 성공)**

### 2.1 SLO — 개정 지시문의 전제와 실제가 다르다 (보고)

개정 §2.1은 `L_SLO = 5×TableI`(yolo11s 31.0 / resnet50 11.2 / mobilenet_v2 6.4 ms)를 쓰라고 했으나,
**논문 Q3 수치를 낸 실행은 균일 `slo_ms=15`를 썼다.** 근거:
`docs/q3_q5_misprediction_report.md:29` — *"`FSRR_FRAME_BUFFER=12` (backlog→latency), **slo_ms=15**,
ε=1.0, T_v=3s"*; `docs/q3_experiment_report.md:6` — *"이전 Q3 성공(BoundGuard 회복 V=0.54)은
**vision-3 + 균일 slo_ms=15** 설정 특유였다."*

게다가 **per-model L_SLO 변형은 이미 시도됐고 폐기됐다**: `q3_experiment_report.md`가 vision-4 +
per-model L_SLO(155/209/56/32 ms) + λ=52로 재실행한 결과를 기록하는데,
`hotswap_buffer_bias_fix.md §6c`는 그 시나리오를 **"시나리오 자체가 무효"**(네 기법 모두 persist=0,
위반 진입조차 안 함)로 판정했다. 반면 `§6b`는 **vision-3 + λ=80 + buffer=12는 "결론 유지"**.

> **즉 5×TableI SLO를 쓰면 Q3 시나리오가 성립하지 않는다.** 개정 §2.1을 그대로 적용하면 안 된다.
> 복원값(균일 15)을 쓴다.

### 2.2 레이트 — **복원 성공** (역산 아님, 기록에서 직접)

**먼저, 로그 기반 역산은 불가능함을 확인했다**: `throughput_fps`는 처리량이 아니라
`1000/avg_inference_time`(= 서비스율 μ)이다(`view_handlers.py:178`; 검증: 1000/9.65 = 103.6 = 기록된
103.63). 따라서 fps로 구간을 나눌 수 없다.

**대신 도착 카운터가 결정적 단서를 준다**: `inference_count + dropped`를 뷰별로 계산하면 세 비전 뷰가
**거의 정확히 같다**(예: 4556 / 4579 / 4581, spread < 0.6 %) → **세 뷰가 공통 λ로 급전**됐다.
그리고 `sweep.yaml`의 조합명이 레이트를 직접 인코딩한다(`gggg_90`, `cggg_90`), 도착 2250 → T = 25.0 s로
정확히 맞아떨어져 이 회계가 옳음을 교차검증한다.

**최종 복원값은 생존한 실험 리포트에 직접 기록돼 있었다**:

| 항목 | 값 | 출처 |
|---|---|---|
| 워킹셋 | llama1b(bg) + yolo11s, resnet50, mobilenet_v2 | 게이트 B 로그 복원 |
| λ (burst) | **80** | `q3_q5_misprediction_report.md:27`, `c2_reactive_baseline_report.md:48`, `hotswap_buffer_bias_fix.md §6b` |
| λ (stable) | **25** | `q3_q5_misprediction_report.md:27` |
| slo_ms | **15 (균일)** | `q3_q5_misprediction_report.md:29` |
| 버퍼 | `FSRR_FRAME_BUFFER=12` | 동일 |
| 레이트 강제 | `FSRR_RATE_REPLICATE=1` (λ=infps) | 동일 |
| ε, T_v, N_cand | 1.0, 3 s, 5 | 동일 |

→ **분기 2-A.** 재실행은 (필요했다면) 재현이었을 것이다. 추정으로 옛 값을 흉내낸 항목 없음.

---

## §3 — 미결 1: **비전 3종** (지시문 결정 채택, 증거로 뒷받침됨)

워킹셋 = `yolo11s, resnet50, mobilenet_v2` + `llama1b`(배경). 게이트 B 복원값 그대로.
**독립 증거**: 비전 4종 변형은 이미 실행됐고(λ=52) 무효 판정됐다(§2.1). 즉 3종 선택은 "실행된 것을
서술한다"는 원칙만이 아니라 **4종이 성립하지 않는다**는 실측으로도 뒷받침된다.

§IV-A "전경 4종" 서술 정정 및 시나리오별 전경 표는 4단계 패치 사항(본 문서는 재료만 제공).

---

## §3.1 — 랭킹 재산출 (승인된 예외, 이 워킹셋 하나)

생성기에 `WS_PAPER_V3`(위 복원값) 추가. 정본 β=1.0/α=0.3으로 재산출:

| 아티팩트 | 플랫폼 | 후보 | 회복 배치 순위 |
|---|---|---|---|
| `rankings/ranking_Q3Q6_paper_v3_cpu-gpu.json` | cpu-gpu | 16 | **5** |
| `rankings/ranking_Q5_paper_v3_cpu-npu.json` | cpu-npu | 16 | **2** |

상위 순서 (cpu-gpu, 표기 `llama,yolo11s,resnet50,mobilenet` / g=GPU c=CPU):
`1:gggg(1.9594) 2:gggc 3:ggcg 4:gcgg 5:cggg(0.9367) 6:gcgc`
→ **top-1 = 전부 GPU(오예측)**, 2–4위는 llama를 GPU에 둔 채 비전 하나를 CPU로 옮기는 배치,
**5위가 회복 배치(llama→CPU)**. 논문 서술 구조와 정확히 일치.

cpu-npu: `1:aaaa 2:caaa(회복) 3:aaca 4:acaa 5:aaac` → **rank 2**, 논문 Q5와 일치.

### β 순수 효과 분리 (워킹셋 고정)
동일 워킹셋 β=0.5 랭킹(비교 전용, `rankings/stale_beta0.5/*_paper_v3_*.json`, `--allow-noncanonical`):
**회복 배치 rank 5 — β=1.0과 동일.** 상위 6개 순서도 동일(점수만 스케일 차이).
→ **β는 이 시나리오의 후보 순서를 전혀 바꾸지 않는다.** vision-only 셋에서 β 무관인 것과 별개로,
LM을 포함한 이 셋에서도 순서가 불변임을 확인했다.

### 강건성
llama1b의 infps를 1/2/10/40/80으로 흔들어도 회복 순위 **5 불변**, top-3 구성도 불변.

### legacy 예측기 재현 확인
legacy 예측기로의 직접 재현은 **불가능**하다 — legacy 번들은 옛 워킹셋(resnext50/vgg19/yolov4)에만
프로파일이 있어 `yolo11s`/`mobilenet_v2`에서 unprofiled-model 검사에 걸린다(`deploy_predictor_logic_legacy`
헤더가 명시). 대신 **실행 로그가 직접 증거**다: 게이트 B에서 복원한 `cand_5` = `llama1b:CPU + 비전3:GPU`가
곧 legacy 랭킹의 5위이며, 정본 랭킹의 5위와 **같은 배치**다.

---

## §3.2 게이트 C 판정 → **C-1**

정본 β 3-vision 랭킹의 회복 순위 = **5** = legacy. 따라서 지시문 규정대로:

- **30회 재실행 불필요.**
- 논문 Q3/Q6 수치 **그대로 유효**.
- `q3q6_canonical_rerun_instructions.md` **§6.1 수치 갱신 항목 철회**:
  - Q3 *"rejects the top-ranked candidate and the next three ... admits the fourth alternative"* — **수정 불필요**(rank 5 유지).
  - Q4 *"rank five is the last position the budget admits, so the margin there is zero"* — **수정 불필요**.
  - `d2_budget_rank` 마커 Q3 $k^\ast$ — **4가 아니라 5 유지**, 재생성 불필요.
- **남는 작업은 재실행과 무관한 것뿐**: Q6 구조 패치(2×2 격자표, bounded↔recovery 문구 정정, A의
  best-so-far 명시, 재방문·전환 상한 항목)와 §IV-A 전경 수 정정(3종 명시) — 모두 4단계 패치.
- 계측 추가(재방문 횟수·전환 상한·후보별 δ)는 **재실행이 없으므로 이번에는 불필요**하다. Q6 본문에 그
  수치를 넣으려면 별도 실행이 필요하므로, 넣을지 여부는 사람 판단(§아래 미결).

---

## 미결 — 사람 판단

1. **Q6 재방문 횟수·전환 상한 수치를 본문에 넣을 것인가.** 넣으려면 계측 + 실행이 필요하다(C-1로
   재실행이 불필요해졌으므로, 이는 "새 측정을 추가할지"의 문제). 넣지 않으면 Q6 패치는 구조·문구
   교정만으로 끝난다.
2. **§IV-A 전경 서술 수정 범위**: 미스프리딕션 계열(Q3/Q6/Q5)과 Q4 2-gen이 3-vision임을 시나리오별
   표로 명시. 반영은 4단계 패치.

## 하지 않은 것
- 재실행, 스케줄 생성, 계측 코드 변경 (**C-1이므로 전부 불필요**).
- 3-vision 외 워킹셋의 랭킹 재생성.
- 레이트 추정 — 복원값이 기록에 있었으므로 추정 불필요.
- 논문 문안 수정.
