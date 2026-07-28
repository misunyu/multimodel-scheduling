# 게이트 D — 복원 설정의 귀속 확인

작성 2026-07-28. 성격: 귀속 검증(측정 없음). 결과: **Q3/§4b/Q4 귀속 확인. Q5는 부분 확인.**

---

## 판정

복원값(`λ=80(burst)/25(stable)`, `slo_ms=15` 균일, `FRAME_BUFFER=12`, `RATE_REPLICATE=1`)은
**07-25 임시 스케줄 실행(BoundGuard/Adaptive/Static/Stop-restart + nodwell/reinvoke/hybrid)에 귀속된다.**

### 근거 — 같은 문서가 설정과 코너 이름을 함께 기록한다

`docs/c2_reactive_baseline_report.md` (**날짜 2026-07-25**, 스냅샷 `backup/schedule_executor_main.pre_c2.py`)는
2×2 ablation을 정의하면서 **코너 이름을 실행 스케줄 파일명과 동일하게** 쓴다 —
`A. no-dwell`=`nodwell.yaml`, `B. re-invoke`=`reinvoke.yaml`, `C. hybrid`=`hybrid.yaml`. 그리고 같은 문서
§2가 시나리오별 설정을 표로 준다:

| 시나리오 | 기록된 설정 (c2 리포트 §2) |
|---|---|
| **Q3** (CPU–GPU 오예측) | llama1b bg + vision3, **λ=80, buf=12** |
| **§4b** (예측 정합, 대조군) | vision3, **λ=45, buf=2** |
| **Q4** (가용 후보 없음) | heavy-4, **λ=90, buf=300** |

각 5회 반복, **런 길이 120 s 고정**, warmup(P4), 이상치 미폐기.

**교차 검증**: `results/`의 07-25 실행 파일이 정확히 이 세 코너 스케줄명을 기록하고(§게이트 B),
기록된 배치가 리포트 서술과 일치한다(아래 §게이트 B 재확인). 즉 설정 기록과 실행 로그가 **같은 세션**을
가리킨다. `slo_ms=15`·`RATE_REPLICATE=1`은 `docs/q3_q5_misprediction_report.md:28-29`(07-22, 같은
시나리오 계보)가 명시하며, c2(07-25)가 "나머지 전부 BoundGuard와 동일"로 상속을 선언한다.

### 실행 로그와의 대조 (배치 수준, 07-25)

| 스케줄 | 최종 배치(기록) | 해석 |
|---|---|---|
| BoundGuard.yaml | `cand_5` = **cggg** (llama1b:CPU, 비전3:GPU) | **회복 배치(rank 5)** — 리포트 5/5 회복과 일치 |
| reinvoke.yaml (B) | `cand_5` = **gggg** | 라벨은 5번째지만 **배치는 top-1** — top-1 고착과 일치 |
| hybrid.yaml (C) | `cand_1/2/5` = 전부 **gggg** | 동일 — top-1 고착 |
| nodwell.yaml (A) | `cand_1`=gggg, `cand_2`=gggc, `cand_4`=gcgg, `cand_5`=cggg | **런마다 다른 배치에 정착** — 고분산과 일치 |
| BoundGuard.yaml (NPU) | `cand_2` = **cnnn** | Q5 회복(rank 2)과 일치 |

→ 리포트의 정성 서술이 로그의 배치 기록과 **독립적으로 일치**한다. 귀속 성립.

---

## 시나리오별 귀속 상태

| 시나리오 | 귀속 | 비고 |
|---|---|---|
| **Q3 (cpu-gpu 오예측)** | **확인** | c2(07-25) §2 + 07-25 실행 로그 + q3_q5(07-22) 노브 명시 |
| **§4b (예측 정합)** | **확인** | c2(07-25) §2: vision3, λ=45, buf=2 |
| **Q4 (후보 없음, heavy-4)** | **확인** | c2(07-25) §2 + `q4_experiment_report.md`(07-24): heavy-4, λ=90, buffer=300, 모델·L_SLO 명시 |
| **Q5 (cpu-npu 오예측)** | **부분 확인** | `q5_experiment_report.md`: 워킹셋(llama1b bg + yolo11s/resnet50/mobilenet_v2), **λ=80, buffer=12** 명시(§2.2 사전측정). 다만 Q5 4기법 전체 실험은 같은 문서가 **"NPU 전용 λ 튜닝 필요 → 게이트"**로 남겨둔 상태 — 최종 Q5 수치의 λ 귀속은 **미상** |
| **Q4 2-gen** | **확인(워킹셋)** | `qwen_background_report.md`: 비전 3 + 생성 2, GPU **λ=90**. SLO 규약은 **미상** |

**미상 항목은 추정하지 않는다.** 아래 구성 표에 "미상"으로 적는다.

---

## SLO 규약 — 두 규약이 실재하며 서로 다른 실험에 쓰였다

| 규약 | 값 | 사용 실험 | 근거 |
|---|---|---|---|
| **균일 15 ms** | 모든 뷰 `slo_ms=15` | **Q3/Q5 오예측 계열** | `q3_q5_misprediction_report.md:29`; `q3_experiment_report.md:6` ("이전 Q3 성공은 vision-3 + **균일 slo_ms=15** 설정 특유") |
| **5×Table I** | yolo11x 70 / yolo11l 52 / yolo11m 44 / resnet50 12 ms | **Q4 heavy-4** | `q4_experiment_report.md:33` |
| **per-frame deadline** | 8.13 ms (yolo11s) | **B2 버퍼 스윕** | `b2_buffer_sweep_report.md:17` |

> **개정 지시문 §2.1의 전제는 오류다.** "SLO는 소실되지 않았다 — 5×latency를 그대로 쓰라"는 Q3에
> 적용되지 않는다. Q3는 균일 15를 썼고, **5×Table I 변형은 이미 시도돼 무효 판정**됐다:
> `q3_experiment_report.md`가 vision-4 + per-model L_SLO(155/209/56/32) + λ=52로 재실행한 결과를
> 기록하고, `hotswap_buffer_bias_fix.md §6c`가 그 시나리오를 **"시나리오 자체가 무효"**(네 기법 모두
> persist=0, 위반 진입조차 안 함)로 판정했다. 반면 §6b는 **vision-3 + λ=80 + buf=12는 "결론 유지"**.

---

## 하지 않은 것
- 미상 항목(Q5 최종 λ, Q4 2-gen SLO 규약)의 추정 기입.
- 새 실행·새 계측.
