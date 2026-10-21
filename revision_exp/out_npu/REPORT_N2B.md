# Phase N2-B 보고 — CPU–NPU 재학습·평가 (GPU 보드)

- **작성:** 2026-10-16
- **코드:** `revision_exp_adapter/run_npu.py`(새 파일), `revision_exp_adapter/npu_table3b_tex.py`(새 파일)
- **출력:** `revision_exp/out_npu/`
- **표기:** [확인] / [추정] / [확인 불가]
- **규칙:** 사전 등록 값(§0), baseline 규칙(`baselines_npu.md`, 실행 전에 씀)은 결과를 본 뒤 바꾸지 않았다.

## 0. 요약

| 항목 | 결과 |
|---|---|
| 입력 | `origin/revision-profiling-npu` @ `4070ce2e`. worktree `../mms_npu_results`(detached, clean). labels_train / labels_test sha256 앞 16자 `7f5acf8534fd58e4` / `fed522320da95488`, 일치 |
| 학습 | 279 후보(1·2·4×, 33 workloads, 11 families), feature 94개. T·D 각각 grid 16 × 3 folds. 실행 43 s |
| Table 3(b) | 4 workloads, 60 후보. **Ours = Latency-based = Oracle**: Avg 0.990, regret 0.000, Top-1 4/4. Per-instance 0.793 (1/4). Best-of-5 0.828 ± 0.102 (Top-1 0.35). All-CPU 0.367 |
| 핵심 | 3× 평가 4개에서는 latency 규칙이 oracle과 모두 같다. **Ours가 latency-based보다 나은 점은 이 평가 집합에서 드러나지 않는다.** latency 규칙이 틀리는 곳은 학습 split의 4× 4개다(§5) |

## 1. 설정 고정과 3× 격리 [확인]

### 1-1. 입력·설정 고정

- **설정 기록:** `logs/config_frozen.json`(3× label을 열기 전에 기록). 담긴 항목:
  - 입력 commit, label sha16, grid, base params
  - D weight(10, D_macro > 0.01)와 q_i weight(같은 값)
  - family → fold 매핑(11 families → 3 folds, `folds.csv`), feature 94개
  - 선택된 hyper-parameter, artifact sha256
- **fold:** 각 fold에 family 3–4개다. 한 family가 여러 fold로 갈라지지 않는다(assert).

### 1-2. 3× 격리

- **(i) 학습 데이터:** 학습·grid·fold에는 1·2·4× 행만 썼다(rate assert).
- **(ii) feature 열 검사:** 열 94개 모두 schedule feature 패턴(`data.ALLOWED_FEATURE_RE`)에 맞는다. 실측 지표 token이 들어간 열은 0개다.
  - 검사한 token: F_mean, F_max, R_max, R_agg, D_macro, _rep, _se, completed, dropped, generated, throughput, drop, score
- **(iii) 예측을 먼저 쓰고 label을 나중에 읽었다.**
  - 3× feature는 `test_schedules_x3.csv`(schedule)만으로 만들었다.
  - 이 feature로 Ours, Per-instance, Latency-based의 예측과 선택을 계산했다.
  - 결과를 `blind/predictions_x3_blind.csv`, `blind/selections_x3_blind.csv`로 썼다. sha는 `logs/blind_stage.json`에 있다.
  - 그 뒤에야 3× label 파일을 열었다. adapter의 guard가, blind 파일을 쓰기 전에 `labels_test.csv`, `test_x3.csv`, `labels_instances_all.csv`를 열면 assert로 멈춘다.
  - label을 연 뒤 확인한 것:
    - 기존 loader가 다시 만든 3× feature가 blind feature와 bit 단위로 같다.
    - 다시 계산한 예측도 blind 예측과 같다.
- **(iv) NPU feature 열:**
  - feature 94개 중 npu0 열 18개, npu1 열 18개다.
  - 학습 279행 중 npu0 사용 177행, npu1 사용 177행이다. 두 열이 다른 행이 138개라 배치에 따라 구분된다.
  - gpu 열 18개는 builder가 늘 만드는 열이고 값은 0이다.
  - instance 4개의 model hash bucket(38, 101, 32, 64)은 서로 겹치지 않는다.

### 1-3. 사전 점검

- 학습 279행, 평가 93행(primary 60).
- loader의 D와 D_macro_mean의 차는 0이다. T와 throughput_norm의 차도 0이다.
- instance q_i 평균으로 다시 만든 D_macro와 label의 차는 최대 2.2e-16이다.
- evaluator self-test 5개를 통과했다.

## 2. 학습 결과 [확인] — `hparam_search.csv`, `hparam_search_instance.csv`, `train_performance.csv`

| 모델 | 선택 hyper-parameter | CV MAE |
|---|---|---|
| T | lr 0.05, depth 4, n 1000, subsample 0.8 | 0.0888 |
| D (weight 10) | lr 0.05, depth 4, n 1000, subsample 0.8 | 0.0975 |
| Per-instance t_ratio | lr 0.05, depth 6, n 500, subsample 1.0 | 0.0242 (765 instance 행) |
| Per-instance q_i (weight 10) | lr 0.05, depth 6, n 500, subsample 0.8 | 0.0289 |

- **CV MAE 해석:** family 단위로 묶은 CV라, validation fold의 family는 학습에 없다. 그래서 MAE가 CPU–GPU(T 0.029, D 0.032)보다 3배 정도 크다. CPU–GPU는 22 families, 이번은 11 families다 [추정: family 수 차이가 원인].
- **학습 데이터 in-sample:** 1×·2×·4× 모두 Avg가 oracle과 0.0002 안이다. 이 값은 평가 표와 섞지 않는다.

## 3. Table 3(b) [확인] — `table3b.{csv,md}`, `manuscript/table3b.tex`

3× N≥3, 4 workloads, 60 후보, α = 0.3, D = D_macro, 3회 평균 label.

| Method | Avg score | Regret | Top-1 | Top-5 | MAE(T) | MAE(D) |
|---|---|---|---|---|---|---|
| Oracle | 0.9898 | 0.0000 | 4/4 | 4/4 | N/A | N/A |
| **Predictive (Ours)** | **0.9898** | **0.0000** | **4/4** | **4/4** | 0.1372 | 0.1100 |
| Latency-based | 0.9898 | 0.0000 | 4/4 | N/A | N/A | N/A |
| Per-instance predictor | 0.7930 | 0.1968 | 1/4 | 1/4 | 0.2309 | 0.1333 |
| Best-of-5 (1,000 추출) | 0.8277 ± 0.1017 | 0.1621 | 0.350 (평균 적중률) | N/A | N/A | N/A |
| All-CPU (참고) | 0.3666 | 0.6232 | 0/4 | N/A | N/A | N/A |

- **Best-of-5:**
  - Top-1 0.350은 무작위 기대값 0.348((3×5/13 + 5/21)/4)과 맞는다. sampler에 문제가 없다는 뜻이다.
  - oracle 포함률은 0.350이다. ±는 1,000회 추출 간 4-workload 평균의 표준편차다.
- **All-accelerator:** NPU당 instance 1개 제약 때문에 N≥3에서 feasible하지 않아 표에 없다.
- **Ours의 MAE와 선택:** MAE가 커도 순위는 거의 맞다.
  - 예측 Ŝ와 실측 S의 Spearman 상관은 workload별 0.95–0.995다.
  - MAE는 S가 낮은 후보에서 크다. T 오차가 S 하위 3분위에서 0.15–0.17, 최상위 분위에서 0.066이다.
  - 그래서 상위 후보 순서가 유지된다 [확인: 진단 계산, 이 보고서의 수치].
- **paired 비교** (`paired.csv`, `paired_summary.csv`):

| 비교 | 평균 차 | workload 간 SE | +/0/− |
|---|---|---|---|
| Ours − Latency-based | 0.000 | 0.000 | 0/4/0 |
| Ours − Oracle | 0.000 | 0.000 | 0/4/0 |
| Ours − Per-instance | +0.197 | 0.084 | 3/1/0 |

- **Top-1 동률 통계** (`top1_tie_stats.csv`):
  - oracle 1·2위 차 ≤ 0.0002인 workload는 0/4다(차 0.0095–0.0187).
  - 반복별 oracle이 평균 label oracle과 다른 workload는 0/4다. 반복마다 자체 정규화한 경우와 평균 F_max로 정규화한 경우 모두 같다.
- **유의성:** workload가 4개뿐이라 유의성은 주장하지 않는다.

### 3-1. 참고: 3× 전체 11 workloads (`table3b_all11.csv`, 93 후보)

| Method | Avg | Regret | Top-1 | Top-5 |
|---|---|---|---|---|
| Oracle | 0.9934 | 0 | 11/11 | 11/11 |
| Ours | 0.9890 | 0.0044 | 10/11 | 11/11 |
| Latency-based | 0.9934 | 0 | 11/11 | N/A |
| Per-instance | 0.9174 | 0.0759 | 7/11 | 8/11 |
| Best-of-5 | 0.9174 ± 0.0438 | 0.0760 | 0.688 | N/A |
| All-CPU | 0.4989 | 0.4945 | 3/11 | N/A |

- **Ours가 틀린 1곳:** ybys_x3(N=2)이다.
  - Ours는 yb npu1, ys npu0을 골랐다(S 0.9440).
  - oracle은 yb npu0, ys npu1이다(0.9921). regret 0.048이다.
- **Best-of-5와 N=1:** N=1 workload 4개는 후보가 3개라 Best-of-5가 후보 전체를 쓴다. 그래서 Top-1이 1이다.

## 4. Baseline 규칙과 선택 배치 [확인] — `baselines_npu.md`, `blind/selections_x3_blind.csv`, `per_workload.csv`

- **Latency-based** (isolated `service_ms_p50`):
  - ResNet은 CPU(18.8 ms)가 NPU(40–45 ms)보다 짧아 항상 CPU에 놓인다.
  - YOLO big은 npu0(97.7 ms)을 먼저 받는다. small은 npu0(70.0 ms)이 점유돼 있어 npu1(77.9 ms)로 간다.
  - 결과 배치는 4개 모두 oracle과 같다.
  - 이 규칙은 N2-A의 R1 참고 배치와 다르다. R1은 isolated 처리량 순이라 small이 npu0을 먼저 가져간다. 그 결과 R1은 3× YOLO 2개 workload에서 oracle과 달랐다(N2A §3-1).
- **Per-instance:**
  - phase3 코드를 그대로 쓰고 device set과 profile 경로만 바꿨다. 바꾼 함수는 `rows_from_schedules`, `iso_capacity` 두 개다. 범위 안이라 N/A가 아니다.
  - rbrsyb만 맞혔다. 나머지 3개에서는 YOLO 하나를 CPU에 두는 배치를 골랐다.
- **Per-instance가 틀린 이유** (진단, 3× N≥3 instance 201개):
  - instance 처리량 비 t_ratio를 과대 예측했다.
    - ResNet: CPU 0.72 vs 실측 0.66, NPU 0.81 vs 0.54
    - YOLO: NPU 0.98 vs 0.85–0.89
  - q_i는 과소 예측했다. ResNet NPU에서 0.19 vs 0.45다.
  - 오차가 ResNet instance에서 크다(|Δt_ratio| 0.20–0.27). 그래서 합성 F̂의 상대 순서가 바뀐다.
  - 예: rbybys에서 YOLO 하나를 CPU에 둔 배치의 T̂ = 1.00(실측 0.88), oracle 배치의 T̂ = 0.82(실측 1.00)였다.
  - 원인 [추정]: 3× ResNet rate(39.4 req/s)가 NPU 용량(22–24)을 넘는 영역이다. 이 영역의 instance 단위 외삽이 1·2·4× 학습에서 맞지 않았다.

## 5. 핵심 질문

### (1) NPU에서 latency-based는 어디서 틀리는가 — `latency_vs_oracle.csv`

- **3× 평가 4 workloads: 4/4 일치.**
  - workload별 두 배치와 S는 같다(§4, facts 표). 차이는 0이다.
- **학습 split 33 workloads** (label만 쓴 기술 통계, 예측 평가 아님):

| rate | workloads | latency ≠ oracle | 해당 workload (latency 배치 → oracle 배치, regret) |
|---|---|---|---|
| 1× | 11 | 0 | – |
| 2× | 11 | 0 | – |
| 4× | 11 | **4** | ybys (yb npu0, ys npu1 → yb npu1, ys npu0; 0.0388), rbybys (0.0058), rsybys (0.0093), rbrsybys (0.0027). 세 개 모두 같은 방향(yb npu0/ys npu1 → yb npu1/ys npu0) |

- **이 4개는 N2A §3-1의 예상과 같다.** 4×에서 oracle이 big:npu1, small:npu0으로 바뀌는 rbybys, rsybys, rbrsybys, ybys다. latency 규칙은 이들을 모두 놓친다.
- **어디서 틀리는가:**
  - ResNet은 CPU, YOLO는 NPU라는 큰 방향은 모든 rate에서 맞다. 틀린 곳은 33개 중 4개다.
  - 틀리는 곳은 YOLO 2개가 NPU 2개를 나눠 가질 때다. isolated latency는 big에 빠른 npu0을 주지만, 높은 rate(4×)에서는 small이 npu0을 쓰는 편이 낫다.
  - 3×(평가 rate)에서는 latency 규칙의 배정이 맞아, 평가 집합으로는 이 차이가 드러나지 않는다.
- **원고 서술은 사용자가 정한다.** 규칙·split·지표는 바꾸지 않았다.

### (2) Ours가 oracle에 얼마나 가까운가

- **primary 4 workloads:** regret 0, Top-1 4/4, Top-5 4/4다.
- **3× 전체 11:** regret 0.0044, Top-1 10/11이다. 틀린 1곳은 ybys_x3다(N=2, regret 0.048). Ours가 big npu1 / small npu0을 골랐는데, 3×의 oracle은 반대 배치였다.
- **α sensitivity** (`fig4_npu_data.csv`, 재학습 없음): α 0.1–1.0 전부에서 Ours = oracle이다(4/4, regret 0). 그림은 만들지 않았다.

### (3) Best-of-5 대비

- **결과:** Ours 0.990 > Best-of-5 0.828 ± 0.102(regret 0.162)다. 4 workloads 모두 Ours가 Best-of-5 평균보다 높다. workload별 Best-of-5 평균 S는 0.711–0.914다.
- **기존 원고와 비교:** 원고의 "Best-of-5 > Ours"(1월 데이터)는 이 새 데이터에서 **나오지 않았다.**
- 1월 데이터와 직접 비교는 하지 않는다. driver, window, rate 규칙, 모델 경로가 다르기 때문이다(N1 §2-6, N2A §5).

## 6. [추정] / [확인 불가]

- [추정] CV MAE가 CPU–GPU의 3배인 이유는 family 수(11 vs 22)와 family 단위 외삽이다.
- [추정] Per-instance가 틀린 이유는 3× ResNet rate에서 NPU 포화 영역의 instance 단위 외삽이다.
- [확인 불가] SDK 버전 문자열은 N1·N2A 기록에 없다(facts 참고).

## 7. 불변 확인 [확인]

- **현재 작업 트리** (`revision-profiling` @ `38111dd`): tracked 파일 변경 0이다. 새로 생긴 것은 untracked `revision_exp/out_npu/`와 `revision_exp_adapter/run_npu.py`, `npu_table3b_tex.py`다. `revision_exp_adapter/`는 원래 untracked 폴더다.
- **`revision_exp/` 코드:** 수정 0이다(import만).
- **git 변경:**
  - `git fetch`로 remote-tracking ref `origin/revision-profiling-npu`가 생겼다.
  - `git worktree add`로 `../mms_npu_results`(detached @ 4070ce2e, status clean)가 생겼다.
  - 다른 branch는 바뀌지 않았다.

## 8. 파일

| 파일 | 내용 |
|---|---|
| `baselines_npu.md` | baseline 규칙(실행 전 작성) |
| `table3b.{csv,md}`, `table3b_all11.csv` | 평가 표 |
| `per_workload.csv`, `paired.csv`, `paired_summary.csv`, `top1_tie_stats.csv` | workload별 결과, paired 비교, 동률 통계 |
| `latency_vs_oracle.csv` | 3× 11 + 학습 33 workloads, latency 배치 vs oracle |
| `fig4_npu_data.csv` | α sensitivity (참고) |
| `hparam_search.csv`, `hparam_search_instance.csv`, `cv_mae`·`best_params`(`summary.json`), `train_performance.csv` | 학습 결과 |
| `folds.csv`, `folds_summary.csv`, `feature_columns.txt` | fold와 feature 목록 |
| `blind/` | label 읽기 전의 3× 예측·선택 |
| `artifacts/multi/{T,D,D_noweight}.json`, `artifacts/per_instance/{t_ratio,q_i}.json` | 학습된 모델 |
| `logs/` | config_frozen, blind_stage, environment, run log, git status |
| `input_manifest.csv` | 읽은 모든 입력 파일의 sha256 |
| `manuscript/table3b.tex`, `table3b_values.csv`, `facts_npu.md` | 원고 자산 |
