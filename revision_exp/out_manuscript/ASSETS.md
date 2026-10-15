# 수정 원고 자산 (CPU–GPU)

작성 2026-10-14. 모든 산출물은 `revision_exp/out_manuscript/`, 생성 코드는 `out_manuscript/code/`에 있다.
새 실험은 runtime overhead 측정 하나이고, 나머지는 확정된 결과를 옮긴 것이다.

- **이전 측정**(`experimental_results/`, 구 pipeline)은 쓰지 않았다.
- **불변 확인**: `out_reprofiled/`, `out_phase3/`, `revision_profiling/`에 이 작업 중 바뀐 파일이 없다(수정 시각 대조). Tracked 파일 변경 0, 새 `__pycache__` 0.

## 1. Fig. 1(a) — All-GPU vs Oracle, workload별

- **파일**: `fig1a.pdf`, `fig1a.png`, `fig1a_data.csv`.
- **출처**: `out_reprofiled/per_workload_x3.csv`(strategy All-GPU, Oracle의 `S_selected`), `out_reprofiled/oracle_definitions.csv`(oracle 배치, `n_oracle`).
- **배치**: 12 workloads를 N 오름차순(3,3,3,3,3,3,4,5,6,6,7,7)으로 놓고, 같은 N 안에서는 라벨 순이다.
  - 라벨은 g=GPT-2, m=MnasNet, r=ResNet50, x=ResNeXt50, s=ShuffleNet, q=SqueezeNet, v=VGG19, y=YOLOv4다.
- **전부-GPU가 oracle인 workload는 6개**(g+x+v, g+x+y, m+r+x, m+x+y, r+v+y, g+m+r+x)다. 그중 m+x+y와 g+m+r+x는 oracle 동점 2개 중 하나가 전부-GPU다.
- **차이가 큰 workload**: g+m+r+x+s+q+v (N=7) 0.136, m+r+x+s+q+v+y (N=7) 0.079, g+m+r+x+s+q (N=6) 0.075.

## 2. Fig. 1(b) — GPT-2 starvation

- **파일**: `fig1b.pdf`, `fig1b.png`, `fig1b_data.csv`, 값 `figure_facts.json`.
- **출처**: `revision_profiling/dataset/labels_test.csv`(3회 평균 label), `labels_instances_all.csv`(instance별 3회 평균), `out_reprofiled/oracle_definitions.csv`(배치 일치 확인).
- **Workload**: gpt2+mnasnet+resnet50+resnext50+shufflenet+squeezenet+vgg19, N=7, 3×.

| 배치 | combination | 구성 (정렬된 모델 순 g,m,r,x,s,q,v) | F (req/s) | T | D_agg | D_macro | S_agg | S_macro |
|---|---|---|---|---|---|---|---|---|
| (i) S_agg 최대 | 108 | CCGGCGG | 1319.77 | 1.000 | 0.013 | 0.102 | **0.996** | 0.969 |
| (ii) S_macro 최대 | 079 | GGGGCCG | 1294.80 | 0.981 | 0.070 | 0.014 | 0.960 | **0.977** |
| (iii) F 최대 | 108 | (i)과 같음 | | | | | | |

원고 §2 문장용 값:

- 두 배치의 T 차(i − ii): **0.019**.
- (ii)에서 모든 instance의 최대 q_i: **0.072**(SqueezeNet, CPU). 나머지 instance는 0.017 이하다.
- (i)에서 GPT-2의 q_i: **0.708**(GPT-2가 CPU에 있음).
- 두 배치의 D_agg 차(ii − i): **+0.057**(0.070 vs 0.013).
- throughput 최대 배치는 (i)과 같다(combination 108).

## 3. Fig. 3 — 모델 수별 평균 score

- **파일**: `fig3.pdf`, `fig3.png`, `fig3_data.csv`.
- **출처**: `out_reprofiled/fig3_data.csv`.
- **시리즈**: Oracle, Ours, All-GPU(= Latency-based, 한 시리즈), Best-of-5(±표준편차 띠). All-CPU는 넣지 않았다.
  - 모든 N 구간에서 All-GPU와 Latency-based 값이 같음을 assert로 확인했다.
- **x축**: `≥3 (12)`, `≥4 (6)`, `≥5 (5)`, `≥6 (4)`, `≥7 (2)`.
- **그리기 코드**: `plots.py`의 스타일 함수(`_style`)와 선·marker 설정을 그대로 썼다. `plots.fig3`는 Latency-based를 따로 그리고 x축 라벨을 바꿀 수 없어서, 같은 설정의 함수를 `code/figures.py`에 새로 썼다(`plots.py`는 수정하지 않음).

## 4. Fig. 4 — α sensitivity

- **파일**: `fig4.pdf`, `fig4.png`, `fig4_data.csv`.
- **출처**: `out_phase3/fig4_noclip/fig4_data.csv`(Multi = Phase 2 값, Single = clip 없는 값).
- **구성**: 2패널이다. 왼쪽은 regret 선 그래프(y축 0부터), 오른쪽은 Top-5 hit(n/12) 막대다. Score 패널은 없앴고, Top-1은 CSV에만 있다.

| 항목 | 예상 | 결과 |
|---|---|---|
| Multi regret ≤ Single인 α | 9/10 | **9/10**(예외 α=0.1: 0.0034 vs 0.0033) |
| 최대 regret 차 | – | **0.0042**(α=0.6) |
| Top-1 범위 | 둘 다 6–7/12 | Multi 6–7/12, Single 6–7/12 |
| Top-5 범위 | – | Multi 10–12/12, Single 9–12/12 |

## 5. Runtime overhead (새 측정) — `runtime_overhead.md`

- **파일**: `runtime_overhead.md`, `runtime_overhead.csv`, `runtime_overhead.json`.
- **결과**: N=7(128 후보)의 decision time(열거 → feasibility → feature → 예측 → 정렬)은 warm 100회 중앙값 **75.3 ms**(p10–p90 72.5–81.0)다.
  - feature 생성 53.0 ms, 예측 21.6 ms다.
  - cold는 모델 load 64.9 ms를 더해 146.2 ms다.
  - N=3(8 후보)은 5.0 ms다.
  - 후보 1개 단건 처리는 1.1–1.4 ms다.
- **조건**: CPU만, XGBoost nthread 0(전부 사용), 예측은 모델마다 batch 1회 호출이다.
- **검증**: 이 경로의 T̂·D̂가 Phase 2 예측과 같다(차 1e-16).

## 6. Table 2 데이터 — `table2_models.csv`

- **출처**: `revision_profiling/code/model_registry.yaml`과 ONNX 파일 메타데이터(`onnx` 1.19.0으로 읽기만 함). sha256은 8개 모두 registry와 일치한다.

| 모델 | 파일 크기 (MB, 10^6 B) | opset | 입력 | precision | 작업 |
|---|---|---|---|---|---|
| mnasnet | 17.5 | 10 | 1×3×224×224 NCHW | fp32 | 분류 |
| resnet50 | 102.4 | 7 | 1×3×224×224 NCHW | fp32 | 분류 |
| resnext50 | 100.0 | 10 | 1×3×224×224 NCHW | fp32 | 분류 |
| shufflenet-v2-12 | 9.2 | 12 | 1×3×224×224 NCHW | fp32 | 분류 |
| squeezenet1.0-12 | 5.0 | 12 | 1×3×224×224 NCHW | fp32 | 분류 |
| vgg19 | 574.7 | 8 | 1×3×224×224 NCHW | fp32 | 분류 |
| yolov4 | 257.5 | 11 | ?×416×416×3 NHWC | fp32 | 검출 |
| gpt2 | 653.7 | 13 | input_ids, attention_mask (batch, seq) | fp32 | 텍스트 생성 |

- **GPT-2**: prompt 32 token, greedy 16 token, KV cache 없음(graph 입력이 input_ids·attention_mask뿐이고, `lm_runner.py`가 매 step 전체 forward).
- **기존 원고 값과 대조**: GPT-2 654 MB ↔ 653.7 MB, VGG19 575 MB ↔ 574.7 MB로 일치한다(10^6 B 기준. MiB로는 623.4, 548.1).
- **YOLOv4**: 416×416 NHWC로 CHANGES.md의 기록과 일치한다.

## 7. LaTeX 표 — `table3a.tex`, `table4.tex`

- 값은 workload별 저장 결과(`out_phase3/per_workload_all.csv`)와 저장된 예측(`out_phase3/predictions_x3.csv`, `per_instance_candidates_x3.csv`)에서 전체 정밀도로 다시 계산한 뒤, 소수 셋째 자리로 한 번만 반올림했다.
- 원천 요약표(`out_phase3/table_baseline.csv`, `table_all.csv`, 4자리)와 34개 값이 모두 일치한다(`tables_crosscheck.csv`).
- 계산된 값은 `table3a_values.csv`, `table4_values.csv`에 있다(`table4_values.csv`의 `table_all_row` 열은 원천 행 전체).
- **Table 3(a)**:

| 행 | 원천 행 | Avg | Regret | Top-1 | Top-5 |
|---|---|---|---|---|---|
| Latency-based (= All-GPU) | `table_baseline.csv` "Latency-based (= All-GPU)" (0.9727, 0.0247, 6/12) | 0.973 | 0.025 | 6/12 | N/A |
| Per-instance predictor | "Per-instance predictor ([14]-inspired)" (0.9868, 0.0106, 6/12, 11/12) | 0.987 | 0.011 | 6/12 | 11/12 |
| Predictive (Ours) | "XGBoost Full" (0.9930, 0.0044, 7/12, 11/12) = `out_reprofiled/table3a.csv` "Ours (retrained)" | 0.993 | 0.004 | 7/12 | 11/12 |
| Best-of-5 | "Best-of-5" (0.9404, 0.0570, 0.402) + `out_reprofiled/table3a.csv` std 0.021 | 0.940 ± 0.021 | 0.057 | 0.40 | N/A |
| Oracle | "Oracle" (0.9974, 0, 12/12) | 0.997 | 0.000 | 12/12 | N/A |

- **Table 4**: 원천은 `out_phase3/table_all.csv`(clip 없는 최종)다.
  - XGBoost (Ours) 0.027/0.030/0.993/0.004/7/12/11/12
  - Ridge 0.030/0.021/0.994/0.003/7/12/11/12
  - Random forest 0.031/0.041/0.995/0.003/8/12/11/12
  - MLP 0.060/0.047/0.973/0.024/6/12/9/12
  - −Pairwise 0.026/0.045/0.992/0.006/6/12/10/12
  - −Identity×device 0.031/0.035/0.996/0.001/8/12/12/12
- **반올림 주의**:
  - MLP regret은 4자리 표에서 0.0245, 정확한 값은 0.024474라 0.024로 적었다.
  - RF regret은 0.0026이라 0.003이다.
  - Top-k 규칙은 Phase 3와 같다(Oracle·Latency의 Top-5는 N/A).

## 8. Profiling cost — `profiling_cost.md`

- **출처**: `revision_profiling/logs/exec_order.csv`(run별 시작·종료), `campaign_sessions.log`.
- **결과**:
  - 학습용 1·2·4× 1,512 runs **7.754 h**, 평가용 3× rep0 504 runs **2.588 h**, rep1+rep2 960 runs **4.955 h**, 합계 **15.297 h** = campaign 경과 시간 15.297 h(일치).
  - run 중앙값은 18.56 s(N=1 16.2 s – N=7 19.0 s)다.
  - N=7 set 1개는 512 runs 2.764 h, 반복 포함 768 runs 4.146 h다.
- **15.3 h에 들어가지 않는 것**: isolated offline profile(약 4분 [추정]), Phase 1b pilot(약 43분), §5.4 반복 pilot(7.1분).

## 9. §4.2 config — `config_check.md`

- grid, 3-fold family GroupKFold(shuffle 없음), squared error, feature 204, D weight 10(D_macro > 0.01)은 모두 일치한다.
- 초안에 빠진 것: 선택 기준이 D에서도 **가중치 없는** validation MAE의 fold 평균이라는 점, 동점은 grid 첫 후보라는 점.
- 선택된 hyperparameter:
  - T: lr 0.05, depth 6, n 500, subsample 1.0
  - D: lr 0.05, depth 4, n 1000, subsample 0.8

## 10. Implementation details — `impl_details.md`

12행 모두 파일·행 번호로 확인했다. 원고 초안과 다른 점은 아래에 모았다.

## 원고 초안과 다르거나 주의할 값

| 항목 | 초안 / 예상 | 실제 |
|---|---|---|
| 분류 전처리 | "표준 ImageNet 전처리" | 6개 중 4개(mnasnet, resnet50, resnext50, shufflenet)만 ImageNet mean/std이다. **squeezenet1.0, vgg19는 Caffe 방식(BGR, 평균만 뺌)**(`model_registry.yaml`) |
| Execution model 서술 | – | `executor.py:1-2` docstring은 "one feeder thread"지만 실제 기본값은 instance별 feeder process다(`executor.py:221`, 결과 JSON `env.process_layout`). 원고는 process로 써야 한다 |
| GPT-2 CPU EP 비율 | 27% | 26.9%(일치) |
| 모델 크기 | GPT-2 654 MB, VGG19 575 MB | 653.7 MB, 574.7 MB(10^6 B 기준 일치) |
| YOLOv4 입력 | 416 NHWC | 416×416 NHWC(일치) |
| Fig. 4 Top-1 범위 | 6–7/12 | 6–7/12(일치) |
| Fig. 4 regret | 9/10 | 9/10(일치), 최대 차 0.0042 |
| Table 4 MLP regret | 0.0245(4자리 표) | 셋째 자리로 0.024 |
| Fig. 1(b) (iii) | – | throughput 최대 배치 = S_agg 최대 배치 |

## [확인 불가]·[추정] 목록

- **CUDA 12.8 / cuDNN 9.10**: 설치된 pip wheel 버전(`nvidia_cuda_runtime_cu12-12.8.90`, `nvidia_cudnn_cu12-9.10.2.21`)이다. ORT가 run마다 load한 library 버전은 기록되지 않았다 [추정].
- **NVIDIA driver 580.159.03, Ubuntu 22.04.5**: 2026-10-14에 조회한 값이다. campaign 당시 값은 기록이 없다 [추정].
- **GPU sharing**: 같은 process의 session들이 CUDA primary context를 공유한다는 것은 ORT 동작에 근거한 추정이다(코드에는 stream·MPS 설정이 없다는 것만 확인) [추정].
- **Isolated offline profile 시간(약 4분)**: 결과 파일의 수정 시각 기준이다. 첫 run 시작 시각은 기록이 없다 [추정].
