# Phase 2 보고서 — 반복 편차, 전체 campaign, 재학습 (지시문 v1.2)

작성 2026-10-14. 표시: [확인] = 이번에 측정·계산으로 확인, [추정] = 근거는 있으나 직접 확인하지 않음.

## 요약

- Campaign 2,976 runs가 중단 없이 15.30시간에 끝났고 **전부 valid**다(재시도 0, 실패 0, 제외 workload 0). [확인]
- 3× N≥3 12개 workload, α = 0.3, D_macro 기준 **Table 3(a)**: Oracle 0.9974, Ours 0.9930, All-GPU 0.9727, Latency-based 0.9727, Best-of-5 0.9404, All-CPU 0.3966. [확인]
- **Ours − All-GPU는 평균 +0.0203**(workload 간 SE 0.0128)이다. 12개 중 3개에서 이기고, 8개는 같은 배치를 골라 0, 1개에서 진다. 이득은 N≥6의 세 workload에서 나온다. [확인]
- **Latency-based가 All-GPU와 같은 배치가 됐다.** 새 offline profile에서는 8개 모델 모두 GPU latency가 더 짧다. 원고의 "Latency-based 0.63"에 해당하는 차이는 새 데이터에 없다. [확인]
- 전부-GPU가 oracle인 workload는 12개 중 6개(동점 포함)로, 세 drop 정의에서 같다. [확인]
- 설정은 3× 평가 전에 고정했고 결과를 본 뒤 바꾼 것이 없다. [확인]

## 0. 설정 고정 확인

- β 0.1, α 0.3, Δ(warm-up 5초 + 측정 10초), thread 4, spinning off, nice 없음, 전처리, feature 정의, counter 정의, split(1·2·4× 학습 / 3× 평가)은 §1 시작 시점 그대로다. Runtime 코드는 Phase 1b 최종에서 바꾸지 않았다(`CHANGES.md`).
- 재학습 설정(grid, fold, D 가중치 10, feature 204개, 선택된 hyperparameter)은 3× 파일을 읽기 **전에** `revision_exp/out_reprofiled/logs/config_frozen.json`에 기록됐다(2026-10-14 10:51:51). 3× 파일은 최종 평가 함수 한 곳에서만 읽는다.
- Campaign 뒤에 새로 만든 스크립트 3개(`integrity.py`, `recompute_scores.py`의 최종 실행, `run_reprofiled.py`)는 첫 실행에서 오류 없이 끝났고, 결과를 본 뒤 코드를 고치지 않았다.
- `revision_exp/`의 기존 파일, tracked 파일은 변경이 없다(`git status`에 수정된 tracked 파일 0, 새 `__pycache__` 0).

## 1. Campaign

### 1.1 Run 수 (`schedules/RUN_PLAN.md`)

| 항목 | 계획 | 실제 |
|---|---|---|
| families | 22 | 22 |
| 고유 배치(2^N의 합) | 504 | 504 |
| rep0 (1·2·3·4×) | 2,016 | 2,016 valid |
| rep1 + rep2 (3×, N≥3) | 960 | 960 valid |
| 합계 | 2,976 | 2,976 valid |
| 시간 | 15.29 h | 15.30 h |

### 1.2 순서와 시간

- rep0 전체(10.34 h, 10/13 19:31 – 10/14 05:51) → rep1(2.48 h) → rep2(2.48 h, 10:48 종료). 반복은 서로 다른 시간 블록이다.
- family 순서와 family 안 (rate, 배치) 순서는 고정 seed로 섞었다. 실제 실행 기록은 `logs/exec_order.csv`.
- 중단·재개 없음(`logs/campaign_sessions.log`에 START 1회, END 1회). Campaign 중 다른 작업 없음.

### 1.3 무결성 (`results/INTEGRITY.md`)

- **Workload 완전성**: 88 workloads 모두 후보 100% valid. 평가 대상 12개는 3회 반복까지 모두 있다. Primary 집계에서 제외한 workload 0개.
- **Conservation·trace**: instance 17,696개 모두 conservation 차 0, trace 검사 통과.
- **도착 실현율**(generated / scheduled): 최소 0.959, 1% 분위 0.989, 중앙값 1.000. 0.95 미만 0개.
- **Generated 차이**(같은 workload의 배치 간 generated 수 차): 위반 workload 2 / 88, 둘 다 N = 7의 4×(학습 데이터 쪽)이고 최대 4.1%다. 가장 적게 생성된 배치는 CPU에 5–6개 instance를 둔 배치다. **3× N≥3의 12개에는 위반이 없다**(최대 1.95%).

### 1.4 시간 drift

- 같은 후보 480개의 F 비율: rep1/rep0 중앙값 0.9999, rep2/rep0 0.9999, rep2/rep1 1.0000. 5–95% 분위 0.992–1.007. 시간에 따른 한 방향 변화는 보이지 않는다.
- 1× 단일 instance run(16개, 4.1–10.1시간 지점)에서 GPU는 모두 요청을 전부 처리했다.
- GPU 온도는 시간대별 평균 46–53°C, 최고 58°C.

## 2. §5.4 반복 편차 (`pilot/repeat_5_4/REPEAT.md`)

- 2×, N = 6, 후보 8개 × 3회 = 24 runs 모두 valid. F의 3회 상대 범위 중앙값 0.09%, 최대 0.17%.
- 순위 불확실 쌍(|ΔS| < √(SE_A² + SE_B²)): 28쌍 중 1쌍. 기준(7쌍) 미만이라 **시나리오 1**로 진행했다.
- 반복별 Top-1은 같지 않았다(rep0 GGGCCG, rep1·2 CGGCGG). 상위 세 배치가 drop 없이 요청 총량을 다 처리한 사실상 동률이다. 이 항목은 판정에 쓰지 않았다.
- 1 SE, 25%는 운영 규칙이고 유의성 검정이 아니다.

Campaign 본 측정에서의 반복 편차(3× N≥3 후보 480개): S의 SE 중앙값 0.0010, 90% 분위 0.0027, 최대 0.0130. 반복별 oracle이 세 번 모두 같은 workload는 8 / 12이고, 다른 4개는 1·2위 차가 0.0002 이하인 동률권이다.

## 3. Label 통계 (`dataset/`, `results/integrity/label_stats_by_workload.csv`)

- 후보 2,016개(train 1,512 / test 504), feature 204개. 정의는 `dataset/README.md`, split은 `dataset/SPLIT_CHECK.md`.
- 전부-GPU가 oracle(동점 포함)인 workload 수, N≥3 12개 중:

| rate | D_macro | D_agg |
|---|---|---|
| 1× | 4 | 4 |
| 2× | 9 | 9 |
| 3× | 6 | 6 |
| 4× | 6 | 6 |

- 3× N≥3에서 전부-GPU 배치의 상태:

| workload (N) | T(all-GPU) | D_macro(all-GPU) | oracle 배치 | S(oracle) | S(all-GPU) |
|---|---|---|---|---|---|
| gpt2+mnasnet+resnet50 (3) | 0.9998 | 0 | GCG | 1.0000 | 0.9998 |
| gpt2+mnasnet+resnet50+resnext50 (4) | 1 | 0 | GCGG = GGGG | 1 | 1 |
| gpt2+…+shufflenet (5) | 0.9982 | 0.0012 | GGGGC | 1.0000 | 0.9978 |
| gpt2+…+squeezenet (6) | 0.9302 | 0.0421 | GCGGCG | 0.9925 | 0.9175 |
| gpt2+…+vgg19 (7) | 0.8676 | 0.0898 | GGGGCCG | 0.9769 | 0.8407 |
| gpt2+resnext50+vgg19 (3) | 1 | 0 | GGG | 1 | 1 |
| gpt2+resnext50+yolov4 (3) | 1 | 0 | GGG | 1 | 1 |
| mnasnet+resnet50+resnext50 (3) | 1 | 0 | GGG | 1 | 1 |
| mnasnet+…+vgg19 (6) | 0.9971 | 0.0020 | CGGGGG | 0.9999 | 0.9965 |
| mnasnet+…+yolov4 (7) | 0.9309 | 0.0352 | CGGCGGG | 0.9996 | 0.9203 |
| mnasnet+resnext50+yolov4 (3) | 1 | 0 | CGG = GGG | 1 | 1 |
| resnet50+vgg19+yolov4 (3) | 1 | 0 | GGG | 1 | 1 |

12개 중 9개는 전부-GPU가 oracle이거나 0.004 이내다. GPU가 포화되는 것은 N≥6의 3개다.

## 4. 재학습 결과 (`revision_exp/out_reprofiled/`)

### 4.1 Table 3(a) — 3× hold-out, N≥3, 12 workloads, α = 0.3, D = D_macro

| 전략 | Avg score | Top-1 | Top-5 | Regret | (참고) 기존 데이터 재학습, 18 workloads | (참고) 원고 |
|---|---|---|---|---|---|---|
| Latency-based | 0.9727 | 6/12 | – | 0.0247 | 0.6763, 3/18 | 0.63, 0.17 |
| **Ours** | **0.9930** | 7/12 | 11/12 | 0.0044 | 0.9505, 13/18, 17/18 | 0.95, 0.72, 0.94 |
| Best-of-5 | 0.9404 ± 0.021 | 0.402 | – | 0.0570 | 0.7299, 0.307 | 0.78 ± 0.040, 0.30 |
| All-GPU | 0.9727 | 6/12 | – | 0.0247 | 0.9426, 13/17 | – |
| All-CPU | 0.3966 | 0/12 | – | 0.6008 | −0.1945 | – |
| Oracle | 0.9974 | 12/12 | 12/12 | 0 | 0.9867 | 0.99 |

- 오른쪽 두 열은 **기존 데이터·기존 drop 정의**의 숫자라 행끼리 직접 비교할 수 없다.
- Best-of-5는 **후보 추출**(5개 배치를 비복원 추출해 그중 측정값이 가장 좋은 것)을 1,000번 반복한 것이고, 측정을 반복한 것이 아니다. ±는 그 1,000번의 표준편차, Top-1은 평균 적중률이다.
- Top-1은 동점 oracle을 모두 정답으로 센다. Ours가 고른 12개 중 8개가 전부-GPU 배치다.
- 예측 MAE(3×, N≥3 후보 480개): T 0.027, D 0.030 (기존 데이터 재학습: 0.060, 0.090).

### 4.2 Ours vs All-GPU, paired (`paired_per_workload.csv`)

| 비교 | 평균 차 | workload 간 SE | + / 0 / − | 최소 | 최대 | label SE(반복 잡음만) |
|---|---|---|---|---|---|---|
| Ours − All-GPU | +0.0203 | 0.0128 | 3 / 8 / 1 | −0.0161 | +0.1362 | 0.0005 |
| Oracle − All-GPU | +0.0247 | 0.0132 | 6 / 6 / 0 | 0 | +0.1362 | 0.0006 |
| Ours − Oracle | −0.0044 | 0.0026 | 0 / 7 / 5 | −0.0263 | 0 | 0.0003 |

- 이긴 3개: gpt2 포함 N=7 (+0.136), gpt2 포함 N=6 (+0.070), yolov4 포함 N=7 (+0.053). 진 1개: mnasnet…vgg19 N=6 (−0.016; 전부-GPU가 oracle에 0.003 차인데 Ours가 다른 배치를 골랐다).
- 0인 8개는 Ours가 전부-GPU와 같은 배치를 골랐다.
- 두 SE는 뜻이 다르다. "workload 간 SE"는 12개 workload 사이의 흩어짐이고, "label SE"는 3회 반복의 측정 잡음만이다. 차이가 난 4개 workload는 세 반복 모두에서 부호가 같다.
- 12개 workload에서 평균 차는 SE의 1.6배다. 유의성 검정은 하지 않았다.

### 4.3 Fig. 3 — 모델 수별 (`fig3_data.csv`)

| N ≥ | workloads | Oracle | Ours | All-GPU | Latency-based | Best-of-5 | All-CPU |
|---|---|---|---|---|---|---|---|
| 3 | 12 | 0.9974 | 0.9930 | 0.9727 | 0.9727 | 0.9404 | 0.3966 |
| 4 | 6 | 0.9948 | 0.9860 | 0.9455 | 0.9455 | 0.9226 | 0.5370 |
| 5 | 5 | 0.9938 | 0.9832 | 0.9346 | 0.9346 | 0.9177 | 0.5517 |
| 6 | 4 | 0.9922 | 0.9796 | 0.9188 | 0.9188 | 0.9085 | 0.5350 |
| 7 | 2 | 0.9882 | 0.9751 | 0.8805 | 0.8805 | 0.8876 | 0.4737 |

기존 그림의 숫자(원고: N≥3에서 Oracle 0.99, Ours 0.95, Best-of-5 0.77, Latency 0.63 … N≥7에서 0.98, 0.86, 0.42, 0.45)는 `fig3_data.csv`의 별도 열에 있다.

### 4.4 Fig. 4 — α sensitivity (`fig4_data.csv`)

α는 0.3으로 고정이고, 아래는 α를 바꿨을 때의 sensitivity다. 서로 다른 α의 절대 점수를 비교해 α를 고르지 않는다.

| α | Oracle | oracle이 α=0.3과 다른 workload | Multi Avg | Multi Top-1 | Multi regret | Single Avg | Single Top-1 | Single regret |
|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.9986 | 1 | 0.9952 | 6 | 0.0034 | 0.9953 | 6 | 0.0033 |
| 0.2 | 0.9977 | 1 | 0.9934 | 6 | 0.0043 | 0.9931 | 6 | 0.0046 |
| 0.3 | 0.9974 | 0 | 0.9930 | 7 | 0.0044 | 0.9919 | 8 | 0.0055 |
| 0.4 | 0.9972 | 0 | 0.9926 | 7 | 0.0046 | 0.9913 | 6 | 0.0059 |
| 0.5 | 0.9971 | 0 | 0.9923 | 7 | 0.0048 | 0.9897 | 6 | 0.0074 |
| 0.6 | 0.9969 | 0 | 0.9919 | 7 | 0.0050 | 0.9877 | 5 | 0.0092 |
| 0.7 | 0.9967 | 0 | 0.9915 | 7 | 0.0052 | 0.9900 | 7 | 0.0067 |
| 0.8 | 0.9965 | 0 | 0.9912 | 7 | 0.0054 | 0.9902 | 6 | 0.0063 |
| 0.9 | 0.9963 | 0 | 0.9908 | 7 | 0.0056 | 0.9881 | 6 | 0.0082 |
| 1.0 | 0.9962 | 0 | 0.9904 | 7 | 0.0057 | 0.9889 | 7 | 0.0072 |

- Oracle 배치는 α에 거의 반응하지 않는다(α 0.1–0.2에서 1개 workload만 바뀜). Multi의 선택도 α 0.1에서 1개만 바뀐다.
- Multi는 재학습 없이 10개 α 중 9개에서 Single(α마다 재학습)보다 regret이 작거나 같다. 차이는 최대 0.004다.

### 4.5 그 밖의 결과

- **D 가중치 ablation** (`ablation_d_weight.csv`): 가중치 10(기본) Avg 0.9930, Top-5 11/12, regret 0.0044. 가중치 없음 Avg 0.9946, Top-5 12/12, regret 0.0028. **이 데이터에서는 가중치가 없는 쪽이 조금 낫다.** 차이는 한 workload(mnasnet…vgg19 N=6)의 선택에서 나온다. 기본 설정은 바꾸지 않았다.
- **Drop 정의** (`oracle_definitions.csv`): (b) D_macro와 (a) D_agg의 oracle이 다른 workload 3 / 12, (b)와 (c) D_max는 1 / 12. gpt2 포함 N=6·7에서 (a)의 oracle은 GPT-2를 CPU에 두는 배치다(GPT-2의 q_i 0.68–0.71, 초당 6.4건이라 aggregate에는 거의 안 보인다). (b)의 oracle은 GPT-2를 GPU에 두고 squeezenet(q_i 0.03–0.07)을 일부 떨어뜨린다. Instance별 수치는 `oracle_definitions_instances.csv`.
- **(a) D_agg 참고 실행** (`revision_exp/out_reprofiled_agg/`): Oracle 0.9990, Ours 0.9930, All-GPU 0.9543, Ours − All-GPU +0.0388 (SE 0.0235, + 4 / 0 6 / − 2).
- **Latency-based** (`latency_targets.csv`): closed-loop latency와 1× open-loop latency 어느 쪽으로도 8개 모델의 선택이 모두 GPU다. 따라서 CPU의 resnet50·resnext50 latency 차(1.37–1.48배)는 결과에 영향이 없다.
- **Clean subset** (`table_clean_subset.csv`): 3× N≥3에 generated 차이 위반 workload가 없어 clean subset = 전체 12개이고 숫자가 같다.
- **학습 데이터 성능** (`train_performance.csv`, 평가가 아니라 in-sample): 1× 0.9998, 2× 0.9999, 4× 0.9907 (각 12 workloads).
- **Validation C**: 같은 절차를 기존 데이터에 적용한 값(0.9505, 13/18)과 새 데이터의 값(0.9930, 7/12)을 `validation_C.csv`에 나란히 뒀다.

## 5. 핵심 질문

**Ours가 All-GPU보다 나은가.** 평균으로는 낫다(+0.020, SE 0.013). 다만 12개 중 8개에서 같은 배치를 고르므로 차이는 GPU가 포화되는 N≥6의 3개 workload에서만 생기고, 1개에서는 0.016 진다. Oracle과 All-GPU의 차 자체가 0.025라서 얻을 수 있는 여지의 82%를 얻은 셈이다. "대부분의 workload에서 All-GPU가 이미 최적에 가깝고, 모델 수가 많아 GPU가 포화될 때 예측 배치가 이득을 준다"가 데이터에 맞는 서술이다.

**전부-GPU가 oracle인 비율은.** 12개 중 6개(동점 2개 포함)이고 세 drop 정의에서 같다. 이전 데이터(13/17)보다 낮아졌지만, oracle이 아닌 6개 중 3개도 All-GPU와 0.004 이내다.

**Latency-based baseline은 유효한가.** 새 profile에서는 모든 모델의 GPU latency가 짧아 Latency-based가 All-GPU와 같다. 원고의 "Latency-based가 크게 뒤진다"는 결과는 재현되지 않는다. TinyLLaMA를 제외했고 CPU 실행 조건이 달라진 것이 원인으로 보인다. [추정]

**결과가 측정 잡음에 흔들리는가.** 아니다. 후보별 S의 SE는 중앙값 0.001이고, Ours와 All-GPU가 다른 4개 workload의 차(0.016–0.136)는 그 label SE(0.002–0.004)의 9배 이상이며 세 반복에서 부호가 같다. 불확실성은 측정이 아니라 workload가 12개뿐이라는 데서 온다.

## 6. 품질 지표와 한계

- 평가 workload는 12개이고 N≥6은 4개, N=7은 2개다. Fig. 3의 오른쪽 점들은 2–4개의 평균이다.
- 평가는 held-out rate다. 22 families 모두 학습(1·2·4×)에도 있다.
- 3× N≥3만 3회 평균 label이고 학습 데이터는 1회 측정이다.
- GPT-2는 3×에서 10초에 64건이라 q_i의 한 건이 1.6%다.
- β = 0.1에서 1×는 대부분 drop이 없다(N≥3에서 전부-GPU oracle 4/12이지만 후보 간 차가 매우 작다).
- 4×의 N=7 workload 2개는 generated 차이 위반(학습 데이터).
- Generated 차이 판정은 200 tick 미만 instance에 절대 기준(2건)을 쓴다. 지시문의 문구(2%)와 다른 점이고 `CHANGES.md`에 적었다. 순수 상대 2%로 보아도 3× N≥3에는 위반이 없다.
- NPU, TinyLLaMA는 이 데이터에 없다.

## 7. 원고에 쓸 수 있는 사실과 다음 단계의 입력

원고에 쓸 수 있는 사실(모두 [확인]):

- 22 model sets, 504 placements, 4 rates, 2,976 runs(3× N≥3은 3회), 전부 valid.
- Table 3(a), Fig. 3, Fig. 4의 숫자는 §4의 표.
- 예측 MAE: T 0.027, D 0.030.
- Drop 정의는 instance별 drop 비율의 평균(D_macro)이고 α = 0.3.

다음 단계에서 결정이 필요한 것:

1. **Latency-based baseline의 서술.** All-GPU와 같아졌으므로 표에서 한 행으로 합칠지, 둘 다 두고 같다고 밝힐지.
2. **주장의 수위.** "All-GPU 대비 평균 +0.020, 12개 중 3개에서 이득"에 맞춰 abstract와 §5의 문장을 고쳐야 한다.
3. **D 가중치.** 새 데이터에서는 가중치 없는 쪽이 조금 낫다. 원고 설정(10)을 유지하고 ablation으로 보고할지.
4. **시나리오 2**(전체 3회, +약 16 h)는 실행하지 않았다. 반복 잡음이 결론에 영향을 주지 않아 필요해 보이지 않는다.

## 파일

| 내용 | 위치 |
|---|---|
| §5.4 | `pilot/repeat_5_4/REPEAT.md` |
| 호환성 사전 검증 | `dataset_pilot_check/COMPAT_CHECK.md` |
| Run 계획 | `schedules/RUN_PLAN.md`, `schedules/README.md` |
| 무결성 | `results/INTEGRITY.md`, `results/integrity/` |
| Dataset | `dataset/README.md`, `dataset/SPLIT_CHECK.md` |
| 재학습(D_macro) | `../revision_exp/out_reprofiled/` (`table3a.md`, `paired_summary.csv`, `fig3_data.csv`, `fig4_data.csv`, `figs/`) |
| 재학습(D_agg 참고) | `../revision_exp/out_reprofiled_agg/` |
| 코드 변경 | `CHANGES.md` |
