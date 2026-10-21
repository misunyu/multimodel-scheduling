# Phase 4 — CPU–GPU 중간 rate(1.5×, 2.5×, 3.5×) 재측정 보고

Phase 2 예측기(Ours)와 Phase 3 per-instance 예측기를 **그대로 평가만** 했다. 재학습, feature·hyperparameter·α 변경은 없다(artifact sha256이 고정본과 같음: T `b6a7ce2d…`, D `ab19e2ad…`).
평가 규칙은 측정 전 PRECHECK로 고정했다. Phase 2 3×에서 Phase 3 값을 그대로 재현했다(`revision_exp/out_intermediate/PRECHECK.md`).

## 1. 측정 요약

| 항목 | 값 |
|---|---|
| 대상 | 12 families(N≥3) × 3 rates = 36 workloads |
| Runs | 1,440 (rep0, 1회 측정), β = 0.1, Phase 2 offline profile |
| 시간 | 2026-10-19 16:01:52 – 23:27:57, 7.43 h(예측 7.43 h) |
| 누락 / invalid / 재시도 / 실패 | 0 / 0 / 0 / 0 |
| 제외 workload | 0 (36개 모두 후보 100% valid) |
| Conservation·trace | 8,784 instance-run 모두 0 / ok |
| 실현율 최소 | 0.9732 (0.95 미만 0개) |
| Generated 차이 위반 | 2 / 36 — N=7 두 family의 3.5×(최대 2.7%) |

- **Generated 차이 위반**: Phase 2와 같이 품질 지표로만 보고하고 제외하지 않았다(Phase 2에서도 N=7 4×에서 같은 양상이 있었다). 이 2개를 뺀 3.5× 10개의 결과는 §2 아래에 있다.
- **비단조 family 2개**: 두 N=7 family의 전부-GPU F가 3.5× < 3×다. 3.5× 단일 run과 3× 3회 평균을 비교한 것이다. 해당 run은 실현율 ≥ 0.9996이고 missed tick도 거의 없다. 측정 오류로 판정하지 않는다.
- **확인 표시 run 5개**: 모두 N≥6의 2.5×/3.5× 전부-GPU이고 F/Λ < 0.95다. GPU 포화로 보인다.

상세는 `results/INTEGRITY.md`에 있다.

## 2. Rate × 방법

| rate | 방법 | 평균 S | regret | Top-1 | Top-5 |
|---|---|---|---|---|---|
| 1.5× | Ours | 1.0000 | 0.0000 | 11/12 | 12/12 |
| 1.5× | Per-instance | 0.9917 | 0.0083 | 5/12 | 10/12 |
| 1.5× | Latency-based (= All-GPU) | 1.0000 | 0.0000 | 11/12 | – |
| 1.5× | Best-of-5 (1,000 trials) | 0.9770 ± 0.0099 | 0.0230 | 0.534 | – |
| 1.5× | Oracle | 1.0000 | – | 12/12 | – |
| 2.5× | Ours | 0.9997 | 0.0003 | 6/12 | 10/12 |
| 2.5× | Per-instance | 0.9998 | 0.0002 | 9/12 | 10/12 |
| 2.5× | Latency-based | 0.9861 | 0.0138 | 5/12 | – |
| 2.5× | Best-of-5 | 0.9529 ± 0.0190 | 0.0470 | 0.409 | – |
| 2.5× | Oracle | 0.9999 | – | 12/12 | – |
| **3× (Phase 3, 3회 평균)** | Ours | 0.9930 | 0.0044 | 7/12 | 11/12 |
| 3× | Per-instance | 0.9868 | 0.0106 | 6/12 | 11/12 |
| 3× | Latency-based | 0.9727 | 0.0247 | 6/12 | – |
| 3× | Best-of-5 | 0.9404 ± 0.0210 | 0.0570 | 0.402 | – |
| 3× | Oracle | 0.9974 | – | 12/12 | – |
| 3.5× | Ours | 0.9868 | 0.0062 | 5/12 | 12/12 |
| 3.5× | Per-instance | 0.9784 | 0.0145 | 6/12 | 10/12 |
| 3.5× | Latency-based | 0.9233 | 0.0697 | 4/12 | – |
| 3.5× | Best-of-5 | 0.9300 ± 0.0222 | 0.0630 | 0.391 | – |
| 3.5× | Oracle | 0.9930 | – | 12/12 | – |

- Best-of-5의 Top-1 칸은 1,000 trials의 평균 적중률이다.
- Oracle 동점(S 차 ≤ 1e-9) workload 수: 1.5× 9/12, 2.5× 2/12, 3× 2/12, 3.5× 1/12. 동점이면 동점 후보 중 어느 것을 골라도 Top-1 적중이다.
- **3.5× generated 위반 2개를 뺀 10개**:

  | 방법 | 평균 S | Top-1 | Top-5 |
  |---|---|---|---|
  | Ours | 0.9925 | 5/10 | 10/10 |
  | Per-instance | 0.9865 | 6/10 | 9/10 |
  | All-GPU | 0.9624 | – | – |
  | Best-of-5 | 0.9420 | – | – |
  | Oracle | 0.9961 | – | – |

**후보 단위 MAE** (480 후보/rate):

| rate | Ours T | Ours D | Per-instance T | Per-instance D |
|---|---|---|---|---|
| 1.5× | 0.0235 | 0.0290 | 0.0243 | 0.0296 |
| 2.5× | 0.0264 | 0.0347 | 0.0167 | 0.0182 |
| 3× (Phase 3) | 0.0273 | 0.0300 | 0.0276 | 0.0236 |
| 3.5× | 0.0283 | 0.0214 | 0.0401 | 0.0256 |

**N≥6 paired (Ours − All-GPU, workload 4개)**:

| rate | 평균 차 | SE | +/0/− |
|---|---|---|---|
| 1.5× | +0.0000 | 0.0000 | 1/3/0 |
| 2.5× | +0.0407 | 0.0265 | 2/0/2 |
| 3× | +0.0608 | 0.0313 | 3/0/1 |
| 3.5× | +0.1866 | 0.0264 | 4/0/0 |

Rate별 선택 비교는 `revision_exp/out_intermediate/selection_vs_3x.csv`에 있다.

## 3. 사실 요약

- **1.5×**: 대부분의 workload에서 GPU 하나로 수요를 다 처리한다. 12개 중 9개가 oracle 동점이고, All-GPU가 11/12에서 최적이다. 그래서 Ours와 All-GPU의 평균 S가 모두 1.0000이다.
- **Rate가 오르면 포화가 커진다.** All-GPU의 regret은 0.014(2.5×) → 0.025(3×) → 0.070(3.5×)으로 커진다. Ours의 regret은 0.0003 → 0.0044 → 0.0062로 작게 유지된다. N≥6 paired 차는 3.5×에서 4개 workload 모두 양수(평균 +0.187)다.
- **2.5×**: Per-instance가 Top-1(9 vs 6)과 평균 S(0.9998 vs 0.9997)에서 Ours보다 약간 높다.
- **Top-1 차이는 안정성 근거로 쓰지 않는다.** 1–3개 workload 차이이고, 중간 rate는 1회 측정이며, 동점 비율도 rate마다 다르다.

## 4. 원고 문장 (템플릿 채움)

> At intermediate rates of 1.5×, 2.5×, and 3.5×, Top-1 hit rates are 11/12, 6/12, and 5/12 and Top-5 hit rates are 12/12, 10/12, and 12/12 (mean score 1.0000, 0.9997, and 0.9868 versus oracle 1.0000, 0.9999, and 0.9930).

**제출본 숫자(Top-1 0.72/0.88/0.88, Top-5 0.94/1.0/0.94, 18 workloads)와 직접 비교할 수 없는 이유:** 제출본과 profile이 다르고(TinyLLaMA 제외, CPU 실행 조건), 대상 workload(12 families, N≥3)도 다르다. 또 이번 값은 1회 측정이며 rate마다 oracle 동점 비율이 다르다.

## 5. 산출물과 확인

- **측정**:
  - `intermediate/RUN_PLAN.md`, `schedules/`, `results/`(1,440 JSON)
  - `results/INTEGRITY.md`, `results/integrity/`
  - `logs/`(exec_order, 10분 progress, GPU 1분, runtime sha256)
- **Dataset**: `intermediate/dataset/`(`README.md` 참조. 원본 스크립트의 고정 이름 `test_x3.csv`에 세 rate가 모두 들어 있다)
- **평가**: `revision_exp/out_intermediate/`
  - `table_intermediate.csv`, `mae_intermediate.csv`
  - `per_workload_intermediate.csv`, `paired_n6_intermediate.csv`, `selection_vs_3x.csv`
  - `predictions_intermediate.csv`, `per_instance_candidates_intermediate.csv`, `summary.json`
- **Leakage 확인**: rate마다 frozen feature 목록 밖의 열 0개. schedule만으로 만든 feature가 동일함을 확인했다(label 정보 미사용).
- **변경 기록**: `intermediate/CHANGES.md`
- **git status**: tracked 파일 변경 없음. 새 산출물은 `revision_profiling/intermediate/`, `revision_exp/out_intermediate/`, `revision_exp_adapter/run_intermediate.py`에만 있다.
