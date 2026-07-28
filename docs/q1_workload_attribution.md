# Q1 계열 · §II 동기 그림의 워크로드 구성 — 귀속 확정

작성 2026-07-28. **조회·귀속만. 새 실행·재계측·그림 재생성 없음.**

판정: **§Q1 = 분기 B**(4종 아님) · **§II = 분기 A(개수) + 구성 불일치**(개수는 맞으나 모델이 Table~I 밖).

---

## 1. §Q1 — `bounded_recovery_analysis.pdf`

### 1.1 결정적 사실: **단일 실험이 아니다**

`fig/confirmed_values.json`에서 이 그림에 들어가는 값 8개의 `source`를 뽑으면:

| id | value | quantity | source |
|---|---|---|---|
| q3_bg_search | 22.6 s | search | **b2rep_q3** |
| q3_bg_drain | 2.8 s | drain | **b2rep_q3** |
| q3_bound | 26 s | bound | T+k(T_v+δ), δ=LLM 2.9 s |
| q5_bg_search | 14.2 s | search | **b2rep_q5** |
| q5_bg_drain | 0.0 s | drain | **b2rep_q5** |
| q5_bound | 26 s | bound | 동일 |
| q4_search | 24 s | search | **q4_bsf_out (논문 런)** |
| q4_bound | 24 s | bound | T+N_cand(T_v+δ)+δ, δ=vision 1 s |

→ 이 그림은 **Q3 · Q5 · Q4 세 시나리오의 집계**다. 논문 캡션(*"Breakdown of recovery latency into
detection delay and post-detection phases"*)과 정합한다.

### 1.2 세 시나리오의 전경 구성 (게이트 B·D에서 이미 확정)

| 시나리오 | 전경 모델 | 개수 | 배경 | deadline 규약 |
|---|---|---|---|---|
| **Q3** (cpu-gpu 오예측) | yolo11s, resnet50, mobilenet_v2 | **3** | llama1b | 균일 15 ms |
| **Q5** (cpu-npu 오예측) | yolo11s, resnet50, mobilenet_v2 | **3** | llama1b | 균일 15 ms |
| **Q4** (가용 후보 없음) | yolo11x, yolo11l, yolo11m, resnet50 | **4** | 없음 | 5×Table~I (70/52/44/12 ms) |

### 1.3 판정 — **분기 B**

현행 *"using **four** foreground models"* 는 **세 시나리오 중 둘(Q3·Q5)에서 거짓**이다. 개수를 하나로
단언할 수 없다 — 이 그림은 3종 워크로드 둘과 4종 워크로드 하나를 함께 그린다.

또한 개수만 고쳐도 안 된다: Q4의 4종은 §IV-A가 열거하는 4종(YOLO11s/11m/ResNet50/MobileNet-v2)과
**구성이 다르다**(YOLO11x/l/m + ResNet50 — MobileNet-v2 없음, YOLO11s 없음).

### 1.4 파급 (§2 분기 B 요구)

§Q1이 보고하는 바운드 33 s와 관측 21.0–30.0 s는 **세 시나리오에 걸친 값**이다. `confirmed_values`의
per-scenario 바운드는 **Q3 26 s · Q5 26 s · Q4 24 s**로 서로 다르다(δ가 LLM 2.9 s인지 vision 1 s인지에
따라 갈린다). **수치를 바꾸라는 것이 아니라**, 어느 값이 어느 워크로드의 것인지 밝혀야 한다.

---

## 2. §II — `qos_score_validation.pdf`

### 2.1 고리: 그림 → 스크립트 → 데이터 → 실행

1. **스크립트**: `scripts/qos_recovery_validation.py` — 헤더가 *"produces qos_score_validation.pdf for
   the paper"*라 명시.
2. **스케줄**: `DEFAULT_SCHEDULE = tests/qos_recovery_schedule.yaml`(현재 `tests/stale_phantom/`로 격리).
   3-phase: `combination_initial`(all CPU, 저부하) → `combination_overload`(동일 배치, 고부하) →
   `combination_offload`(GPU로 이전).
3. **데이터**: `backup/results_pre_mla100_20260720_185647/results/qos_recovery_*.csv` — **2026-04-09/10**
   런 다수. 컬럼: `timestamp, combination, total_fps, view1..4_fps, view1..4_infer_ms, drop_rate_fps, v_score`.
4. **그림 갱신**: `docs/figures/qos_score_validation.pdf` mtime **07-24**. 그러나 7월에 qos_recovery
   실행 기록이 없다 → **replot**이다. 근거: 스크립트에 `--no-run ... # replot only`와 `--csv` 옵션이 있고,
   `PENDING_DECISIONS.md:170`이 *"A부(재생성): Fig1 qos_score_validation(**ε=1.0 수정**)"*,
   `figure_regeneration_report.md:24`가 *"qos_score_validation | plot_fig1 | V(t) 정의 검증 | 없음 | 유지"*로
   기록한다. **데이터 불변, ε 선만 정정**.

### 2.2 교차 검증 — 네 뷰 모두 살아 있었다 (**phantom 아님**)

4월 CSV(`qos_recovery_20260410_104615.csv`) 실측:

| phase | 살아 있는 뷰 | 샘플 fps (view1–4) |
|---|---|---|
| combination_initial | **view1–4 전부** | 3.91 / 4.07 / 5.06 / 5.42 |
| combination_overload | **view1–4 전부** | — |
| combination_offload | **view1–4 전부** | — |

추론시간도 185–255 ms로 정상. → **§II 동기 그림은 phantom 오염과 무관**하다. 4월은 MLA100 교체
(2026-07-20) **이전**이라 당시 런타임이 옛 어휘 모델을 실제로 로드했다(`models_onnx/` 폴더 경로).

### 2.3 판정 — **개수는 맞고, 구성이 Table~I 밖**

- **개수 4는 참**: 네 뷰 모두 실측 산출. *"four concurrent DNN applications"*의 **수는 정확**하다.
- **구성은 불일치**: 스케줄의 4종은 **mnasnet, squeezenet1.0-12, resnet50, resnext50**이며,
  이 중 **Table~I에 있는 것은 resnet50 하나뿐**이다. 지시문 §2 분기 A의 단서
  (*"개수만 맞고 구성이 다르면 그것도 정정 대상"*)에 해당한다.
- **deadline 규약**: 스케줄에 `slo_ms`가 **없다**(grep 0건). 뷰별 deadline은 `1000/infps`로 폴백한다
  (`view_handlers.py:46-52`). `infps: 3` → **약 333 ms**. 균일 15 ms도, 5×Table~I도 아닌 **제3의 규약**이다.
- 이 실행의 `v_score`는 0–148.7 범위로, 논문의 창 정규화 $V(t)$(0–1 부근)와 **척도가 다르다**. 그림은
  ε=1.0로 replot됐으므로 표시 척도는 정정된 상태다.

---

## 3. §3.1 — Table~I $L_{SLO}$ 각주 전제 점검

각주: *"This column applies except where Table~III states otherwise."*

- **§Q1(bounded_recovery_analysis)**: 구성 시나리오 Q3·Q5(균일 15 ms)와 Q4(5×Table~I)는 **모두 Table~III에
  이미 있다** → 각주 전제 성립. **추가 행 불필요.**
- **§II(qos_score_validation)**: deadline이 `1000/infps`(≈333 ms) 폴백이라 per-model 규약이 **아니다**.
  각주 전제를 지키려면 **Table~III에 이 실험 행을 추가**해야 한다(그러면 각주는 그대로 두면 된다).

---

## 4. 미상 / 확인하지 않은 것

- §Q1의 관측 21.0–30.0 s가 세 시나리오 중 **어느 조합의 범위인지** — `confirmed_values`에 그 범위값
  자체가 없다(`q3_bg_persist` 25.4±1.4, `q5_bg_search` 14.2 등 개별값만). **미상**.
- §II 그림에 실제로 쓰인 CSV **파일 하나를 특정**하지 못했다(4월 런 다수, replot 시 `--csv` 인자 기록
  없음). 시나리오·구성은 확정되나 **개별 런 특정은 미상**.
- $\lambda$·버퍼는 **이번 범위 밖**(지시문 §1.2).
