# REPORT_INTERMEDIATE_3REP — Phase 4b: 중간 rate(1.5×, 2.5×, 3.5×) 3회 반복 평균 재평가

- 측정: rep0 = Phase 4(2026-10-19, `results/`), rep1·rep2 = Phase 4b(2026-10-20 14:51 ~ 10-21 05:45, `results_rep1/`, `results_rep2/`).
- 예측기, feature, hyperparameter, α = 0.3, 평가 규칙은 고정했다. 바뀐 것은 label(1회 → 3회 평균)뿐이다.
- 검증 문서:
  - `revision_exp/out_intermediate_3rep/PRECHECK_3REP.md`: 모든 항목 PASS(campaign 전).
  - `RUN_PLAN_REPEATS.md`: 실행 계획.
  - `logs/INTEGRITY_REPEATS.md`: 무결성 점검.
  - `CHANGES_REPEATS.md`: 원본과 다른 점.

## 1. Run·시간·결측·무결성

| | rep0 (Phase 4) | rep1 | rep2 |
|---|---|---|---|
| 계획 / valid | 1440 / 1440 | 1440 / 1440 | 1440 / 1440 |
| 결측, invalid, retry, `failed_runs.csv` | 0, 0, 0, 0 | 0, 0, 0, 0 | 0, 0, 0, 0 |
| run 시간 합 (h) | 7.44 | 7.45 | 7.44 |
| conservation_flow = 0, trace_check ok | 8784/8784 | 8784/8784 | 8784/8784 |
| realization 최소 (< 0.95인 instance) | 0.973 (0) | 0.972 (0) | 0.970 (0) |
| missed_ticks 비율 최대 (> 5%인 instance) | 0.027 (0) | 0.028 (0) | 0.030 (0) |
| generated 차이 규칙 위반 workload | 2 | 3 | 3 |
| snapshot_flag | 0 | 0 | 1 |

- **Campaign 전체**: 14.89 h(예상 14.85 h), 2,880 runs, retry 0회, 실패 0건.
  - 블록 시작마다 runtime 10개 파일의 sha256을 Phase 4와 대조했고, offline profile은 git HEAD와 대조했다. 모두 같았다(`logs/runtime_sha256_rep{1,2}.json`).
- **완전성**: (workload, rep) 108개가 모두 완전하다. 제외한 workload는 0개다(`exclude_workloads.txt` 비어 있음).
- **Generated 차이 위반**: 모두 3.5×의 N = 6·7 workload 세 개에서만 나왔고, 상대 차이는 2.0–3.0%다. Phase 4 rep0와 같은 성격이다. 품질 지표일 뿐 제외 기준은 아니다(Phase 2 규칙).
- **snapshot_flag 1건**: rep2의 (N = 7 yolov4 family, 2.5×, comb 030, squeezenet) instance다.
  - 창 끝의 pending 수가 snapshot과 event trace에서 2 차이 난다.
  - 같은 instance의 conservation과 trace_check는 정상이고, q_i = 0이다.
  - `counters.py`의 진단 필드라서 invalid 판정에는 쓰지 않는다.
- **시간 경과 점검**: 전부-GPU 후보에서 |F_r − F̄|/F̄ > 5%인 경우가 0건이다. 최대는 3.5× N = 7 gpt2 family rep0의 +4.8%다.

**반복 편차.** 후보별 SE는 표본표준편차(ddof = 1)/√3이다. S_r은 공통 F_max를 쓴다.

| rate | S SE median | p90 | max | F 상대 SE median |
|---|---|---|---|---|
| 1.5× | 0.0005 | 0.0019 | 0.0147 | 0.0004 |
| 2.5× | 0.0008 | 0.0028 | 0.0104 | 0.0007 |
| 3.5× | 0.0017 | 0.0058 | 0.0235 | 0.0017 |
| 3× (Phase 2, 비교용) | 0.0010 | 0.0027 | 0.0130 | 0.0010 |

반복 편차는 rate가 높을수록 커진다. 3.5×의 median은 Phase 2 3×의 약 1.7배다.

**반복별 oracle.** 각 반복을 자체 정규화하고, 동점 허용오차는 1e-9로 했다.

| rate | 공통 최적 후보 존재 | 대표 ID 셋 다 같음 | 평균 label oracle이 공통 집합에 포함 |
|---|---|---|---|
| 1.5× | 12/12 | 9/12 | 12/12 |
| 2.5× | 11/12 | 11/12 | 11/12 |
| 3.5× | 12/12 | 12/12 | 12/12 |

## 2. 결과표

Ours/Per-instance/Latency/Oracle은 workload 12개의 평균 S(3회 평균 label)다. Best-of-5는 1,000 trial의 평균 S이고, 괄호 안은 trial 간 std다.

| rate | label | Ours | Per-instance | Latency (= All-GPU) | Best-of-5 | Oracle | Ours regret |
|---|---|---|---|---|---|---|---|
| 1.5× | 3회 평균 | 0.9999 | 0.9920 | 0.9999 | 0.9775 (0.0096) | 1.0000 | 0.000013 |
| 1.5× | rep0 | 0.9999 | 0.9917 | 0.9999 | 0.9770 (0.0099) | 1.0000 | 0.000013 |
| 2.5× | 3회 평균 | 0.9995 | 0.9998 | 0.9862 | 0.9528 (0.0190) | 1.0000 | 0.000409 |
| 2.5× | rep0 | 0.9997 | 0.9998 | 0.9861 | 0.9529 (0.0190) | 0.9999 | 0.000278 |
| 3× (Phase 3) | 3회 평균 | 0.9930 | 0.9868 | 0.9727 | 0.9404 (0.0210) | 0.9974 | 0.004416 |
| 3.5× | 3회 평균 | 0.9852 | 0.9786 | 0.9186 | 0.9293 (0.0221) | 0.9929 | 0.007691 |
| 3.5× | rep0 | 0.9868 | 0.9784 | 0.9233 | 0.9300 (0.0222) | 0.9930 | 0.006196 |

Hit 수(n = 12). Best-of-5 열은 표본에 oracle이 포함된 trial의 비율이다.

| rate | label | Ours Top-1 | Ours Top-5 | Per-inst. Top-1 | Per-inst. Top-5 | Latency Top-1 | Best-of-5 |
|---|---|---|---|---|---|---|---|
| 1.5× | 3회 평균 | **11** | **12** | 5 | 9 | 10 | 0.493 |
| 1.5× | rep0 | 11 | 12 | 5 | 10 | 11 | 0.534 |
| 2.5× | 3회 평균 | **4** | **10** | 9 | 10 | 3 | 0.367 |
| 2.5× | rep0 | 6 | 10 | 9 | 10 | 5 | 0.409 |
| 3× (Phase 3) | 3회 평균 | 7 | 11 | 6 | 11 | 6 | 0.402 |
| 3.5× | 3회 평균 | **4** | **12** | 6 | 10 | 3 | 0.369 |
| 3.5× | rep0 | 5 | 12 | 6 | 10 | 4 | 0.391 |

**Ours − All-GPU, N ≥ 6 (workload 4개, 쌍체 비교).**

| rate | 3회 평균 | rep0 | 3× (Phase 3) |
|---|---|---|---|
| 1.5× | +0.0003 ± 0.0003 (1+/3 = /0−) | +0.0000 (1+/3 = /0−) | |
| 2.5× | +0.0402 ± 0.0256 (3+/0 = /1−) | +0.0407 ± 0.0265 (2+/0 = /2−) | |
| 3× | | | +0.0608 ± 0.0313 (3+/0 = /1−) |
| 3.5× | +0.1926 ± 0.0322 (4+/0 = /0−) | +0.1866 ± 0.0264 (4+/0 = /0−) | |

**T̂/D̂ MAE (3회 평균 / rep0).**
- Ours T: 0.0235/0.0235, 0.0261/0.0264, 0.0282/0.0283.
- Ours D: 0.0292/0.0290, 0.0346/0.0347, 0.0219/0.0214.
- 상세: `mae_intermediate_3rep.csv`.

선택 불변 검증에서 1,440 후보 모두 다음 항목이 rep0와 같았다(`selection_invariance_check.csv`):
- feature, T̂·D̂·Ŝ, 예측 순위, 그리고 Ours·Per-instance·Latency의 선택 ID
- Best-of-5 추출 ID

따라서 아래 변화는 모두 label 변화에서 온다.

## 3. rep0 대비 변화와 원인

Ours의 선택은 그대로이고, 바뀐 것은 label뿐이다.
- **Top-5와 1.5× Top-1**: Top-5는 세 rate 모두 rep0와 같다(12/10/12). 1.5× Top-1도 11/12로 같다.
- **Top-1 감소**: 2.5×는 6 → 4, 3.5×는 5 → 4로 줄었다. 줄어든 세 경우는 모두 **동점 해소**다.
  - **해당 workload**: 2.5×의 gpt2+mnasnet+resnet50+resnext50과 gpt2+mnasnet+resnet50, 3.5×의 gpt2+mnasnet+resnet50이다.
  - **rep0**: Ours 선택(comb 15, 7, 7)은 oracle과 정확히 동점(1e-9 이내)이었다. 이때 oracle 집합 크기는 2였다.
  - **rep1·rep2**: 선택한 후보만 소량의 drop이 생겼다. S_r이 1.0000에서 0.9997–0.9987로 내려갔다.
  - **3회 평균**: 동점이 풀려 선택한 후보가 2위가 됐다. Oracle과의 차이는 0.00018, 0.00021, 0.00045다.
  - **SE와의 비교**: 이 차이는 두 후보 SE를 합성한 값의 약 1–2배다. 같은 원인으로 Latency(= All-GPU, 같은 후보 선택)의 Top-1도 11/5/4 → 10/3/3으로 줄었다.
- **Regret**: 1.5×는 변화가 없다. 2.5×는 0.00028 → 0.00041, 3.5×는 0.0062 → 0.0077로 늘었다.
  - **Oracle 쪽**: 2.5×·3.5× 모든 workload에서 oracle 후보가 rep0와 같다. 따라서 증가는 Ours 선택 후보의 3회 평균 S가 rep0보다 낮아진 데서 온다(**label 변화**).
  - **2.5× (+0.00013)**:
    - N = 7 gpt2 family: +0.00096. comb 77의 S가 −0.0007 바뀌었고, SE는 0.0004다.
    - 동점 해소 두 건: +0.00018, +0.00021.
    - 나머지 네 workload: ±0.0003 이내.
  - **3.5× (+0.0015)**:
    - **N = 7 gpt2 family: +0.0116.** 3.5×에서 rate 평균 regret에 가장 크게 기여한 workload다(rate 평균에 +0.0010). 이 workload의 3회 평균 regret은 0.041로, 전체에서 가장 크다.
      - comb 79의 S가 rep0 대비 −0.0123 바뀌었고, SE는 0.0089다.
      - 같은 family의 전부-GPU 후보도 rep0의 F가 평균보다 +4.8% 높았다(§1 시간 경과 표).
      - 즉 rep0 측정이 상대적으로 높게 나왔던 경우다.
    - N = 7 yolov4 family(comb 103): +0.0045.
    - N = 5 gpt2 family(comb 15): +0.0043.
    - 동점 해소 1건: +0.0005.
    - N = 4 gpt2 family(comb 15): −0.0033으로 줄었다.
  - **SE와의 비교**: 위에 적은 |ΔS|는 해당 후보 S SE의 약 1.4–2.1배다. rep0는 세 반복 중 하나이므로, 평균과 rep0의 차이로 보면 반복 편차 범위 안이다.
- **1.5× oracle 대표 ID 변경**: N = 7 yolov4 family(118 → 127)와 N = 6 gpt2 family(45, 동점 3개)다.
  - rep0의 동점 집합 4개가 3회 평균에서 1개(127)로 줄었다.
  - Ours는 127을 골랐으므로 hit가 유지됐다.
- **Per-instance 1.5× Top-5 변화(10 → 9)**: N = 7 yolov4 family의 comb 102가 rep0에서는 5위 동점이었다. 3회 평균에서는 7위로 내려갔다(S 0.99985 → 0.99966, SE 0.00005). 순위 변화다.
- **반복별 oracle 불일치**:
  - **1.5× 3개**: N = 6 gpt2, N = 7 yolov4, N = 6 mnasnet…vgg19 family다. 대표 ID는 다르지만 세 반복 모두에서 최적인 공통 후보가 있고, 평균 label oracle이 그 안에 있다.
  - **2.5× 1개**: N = 6 mnasnet…vgg19 family다. 반복별 oracle이 54/55/63으로 공통 최적이 없고, 평균 label oracle은 54다.
    - Ours의 선택(39)은 rep0와 3회 평균 모두에서 miss다. 3회 평균 기준 S = 0.9993, oracle과의 차이는 0.0007이다.
  - **3.5×**: 12/12 일치한다.

## 4. 원고 문장

**Ours Top-1/Top-5 (3회 평균, n = 12)**:
- Top-1: 11/12, 4/12, 4/12
- Top-5: 12/12, 10/12, 12/12

**x**: rate별 평균 regret은 0.000013, 0.000409, 0.007691이다. 최대값을 소수 셋째 자리에서 올림하면 **0.008**이다. rep0 기준으로는 0.007이었다.

> For $N \ge 3$, Top-1 hit rates are 11/12, 4/12, and 4/12, and Top-5 hit rates are 12/12, 10/12, and 12/12 at 1.5×, 2.5×, and 3.5×, respectively, with mean placement scores at most 0.008 below the oracle.

## 5. 보호 대상과 출력 위치

- **보호 대상 sha256**: 시작과 종료가 일치한다(`logs/protected_sha256_{start,end}.json`).
  - 10,411 files, changed 0, missing 0, 보호 폴더 내 new 0 → `identical: true`.
- **`git status`**: tracked 파일 변경은 없다(`git status --porcelain --untracked-files=no` 출력이 비어 있음).
- **새 출력**: 아래 경로에만 생성했다.
  - `revision_profiling/intermediate/` 아래: `RUN_PLAN_REPEATS.md`, `CHANGES_REPEATS.md`, `REPORT_INTERMEDIATE_3REP.md`, `code_copy_repeats/`, `results_rep1/`, `results_rep2/`, `dataset_3rep/`, `logs/`(새 파일만)
  - `revision_exp/out_intermediate_3rep/`
  - `revision_exp_adapter/run_intermediate_3rep.py`
