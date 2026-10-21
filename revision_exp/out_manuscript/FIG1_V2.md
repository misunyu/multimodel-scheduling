# Fig. 1 v2 (original format restored)

- **코드**: `code/fig1_v2.py`.
- **출력**: `fig1a_v2.{pdf,png,csv}`, `fig1b_v2.{pdf,png,csv}`. 기존 `fig1a.*`와 `fig1b.*`는 그대로 두었다. 코드는 같은 이름의 파일이 이미 있으면 멈춘다.
- **데이터**: `out_reprofiled/per_workload_x3.csv`(workload 목록, All-GPU 행), `revision_profiling/dataset/labels_test.csv`(F, T, D_macro), `out_manuscript/fig1b_data.csv`(교차 확인). 이 세 파일만 썼다.
- **스타일**: `revision_exp/plots.py`의 `_style`을 썼다. 폭은 8.5 cm(단일 컬럼), 글자 크기는 5.5–6 pt다. 이 PC에는 Times New Roman이 없어 matplotlib serif 대체 글꼴로 그려졌다. 기존 그림과 같은 조건이다.

## Fig. 1(a) v2: All-GPU vs throughput-optimal

T(All-GPU) = F_allgpu / F_max이고, throughput-optimal은 F = F_max인 배치라서 T = 1이다. score와 oracle은 그림에 쓰지 않았고, CSV에 참고용으로만 두었다.

| N | workload | F_allgpu | F_max | T_allgpu | throughput-optimal (배치) |
|---|---|---|---|---|---|
| 3 | gpt2+mnasnet+resnet50 | 375.30 | 375.37 | 0.9998 | combination_005 (GCG) |
| 3 | gpt2+resnext50+vgg19 | 302.57 | 302.57 | 1.0000 | combination_007 (GGG = All-GPU) |
| 3 | gpt2+resnext50+yolov4 | 173.50 | 173.50 | 1.0000 | combination_007 (All-GPU) |
| 3 | mnasnet+resnet50+resnext50 | 496.10 | 496.10 | 1.0000 | combination_006 (CGG), All-GPU와 동률 |
| 3 | mnasnet+resnext50+yolov4 | 404.30 | 404.30 | 1.0000 | combination_006 (CGG), All-GPU와 동률 |
| 3 | resnet50+vgg19+yolov4 | 341.10 | 341.10 | 1.0000 | combination_007 (All-GPU) |
| 4 | gpt2+mnasnet+resnet50+resnext50 | 502.47 | 502.47 | 1.0000 | combination_013 (GCGG), All-GPU와 동률 |
| 5 | …+shufflenet | 755.50 | 756.87 | 0.9982 | combination_015 (GGGGC) |
| 6 | gpt2+…+squeezenet | 1072.10 | 1152.60 | **0.9302** | combination_060 (CCGGGG) |
| 6 | mnasnet+…+vgg19 | 1318.30 | 1322.13 | 0.9971 | combination_062 (CGGGGG) |
| 7 | gpt2+…+vgg19 | 1145.07 | 1319.77 | **0.8676** | combination_108 (CCGGCGG) |
| 7 | mnasnet+…+yolov4 | 1265.00 | 1358.90 | **0.9309** | combination_118 (CGGCGGG) |

- **x 배치**: 같은 N 안에서 ±0.3 범위에 T_allgpu 오름차순으로 균등하게 놓았다(CSV의 `x` 열).
- **F_max 동률**: 3개 workload에서 F_max가 정확히 같은 배치가 2개였다. 평균 count가 같은 경우이며, 세 경우 모두 둘 중 하나가 All-GPU다. 이때 throughput-optimal 배치는 evaluator와 같은 규칙(combination 번호가 낮은 쪽)으로 정했다. CSV의 `n_F_max_tied`와 `allgpu_attains_F_max` 열에 표시되어 있다. T_allgpu에는 영향이 없다(1.0).
- **그림에서 겹치는 점**: N=3–5인 9개 workload는 All-GPU가 F_max와 같거나 0.2% 안이다. 그래서 두 점이 그림에서 겹쳐 보인다. 데이터가 그렇기 때문이다.

### §5.2 본문 수치와의 관계

- **throughput 기준**: All-GPU T 최솟값은 **0.868**이다(gpt2+mnasnet+resnet50+resnext50+shufflenet+squeezenet+vgg19, N=7). 이 workload에서 throughput 손실은 1 − 0.868 = 0.132다.
- **N=6·7의 All-GPU T**: 0.930, 0.997, 0.868, 0.931이다. N=6·7 workload 4개 중 3개에서 All-GPU가 7% 이상 낮다.
- **score 기준 0.136과의 관계**: 본문의 최대 차 0.136은 같은 workload의 score 기준 값이다. S_oracle(0.9769) − S_All-GPU(0.8407) = 0.1362이고, `per_workload_x3.csv`에서 Latency-based의 regret도 같은 0.1362다(Latency-based가 All-GPU를 고름).
  - throughput 차(0.132)와 score 차(0.136)는 비교 대상이 다르다. 앞의 것은 throughput-optimal 배치와 비교한 값이고, 뒤의 것은 score oracle(combination_079)과 비교한 값이다.
  - 두 값이 가장 큰 workload는 같다. 그래서 둘은 모순되지 않는다.
  - 다만 본문에서 "0.136"을 throughput 차로 쓰면 틀린다. throughput 기준으로는 0.132(13.2%)다.

## Fig. 1(b) v2: T and D of two placements

- **Workload**: gpt2+mnasnet+resnet50+resnext50+shufflenet+squeezenet+vgg19, N=7, 3×, 후보 128개.
- **재검증 결과**: `labels_test.csv`로 다시 확인했고, assert가 모두 통과했다.
  - combination_108만 F_max 최대다(동률 없음).
  - combination_079만 S = T − 0.3·D_macro 최대다(동률 없음).
  - F, T, D_macro는 fig1b_data.csv와 1e-6 안에서 같다. 차이는 CSV 반올림 때문이다.
  - 배치 문자열은 fig1b_data의 instance별 device와 같다.

| 배치 | combination | 구성 (g m r x s q v) | F (req/s) | T | D_macro | S = T − 0.3·D |
|---|---|---|---|---|---|---|
| (i) Throughput-maximizing | combination_108 | CCGGCGG | 1319.77 | 1.000 | 0.102 | 0.969 (0.96942) |
| (ii) Drop-aware (max T − 0.3D) | combination_079 | GGGGCCG | 1294.80 | 0.981 | 0.014 | 0.977 (0.97687) |

**패널 분리**: T와 D를 한 패널(y 0–1.12)에 그리면, 인쇄 크기에서 D = 0.014 막대의 높이가 약 0.45 mm다. 1 mm보다 낮아 식별이 어렵다고 보고, 지시대로 T와 D를 좌우 작은 패널 2개로 나눴다. 두 패널 모두 y축은 0에서 시작한다. 막대 위에 값(1.000/0.981, 0.102/0.014)을 표시했고, 범례는 위쪽에 2열로 두었다.

## 렌더 확인 (8.5 cm)

- **fig1a_v2**: 축 라벨·눈금·범례가 겹치지 않는다. 범례는 왼쪽 아래 빈 영역에 있다. y 범위는 0.84–1.01이며 데이터에 맞춰 자동으로 정해졌다.
- **fig1b_v2**: 막대 값 라벨이 서로, 그리고 축 위쪽과 겹치지 않는다. 범례는 패널 위에 있다.
