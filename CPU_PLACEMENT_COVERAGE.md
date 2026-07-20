# CPU 배치 조합 커버리지 검증 — deploy_cpu_npu / deploy_cpu_gpu

- 날짜: 2026-07-20
- 대상: 본수집 540 조합 × 2 플랫폼 (CPU-NPU / CPU-GPU), train / out-of-fold test 양쪽
- **판정: 두 예측기 모두 CPU 배치 커버리지 충분 (✅ 정상)**

## 0. 검증에 쓴 데이터와 재현 방법

| 입력 | 경로 |
|---|---|
| 본수집 원시 창(540개) | 세션 scratchpad `collect/windows_npu/performance_npu_all.json`, `windows_gpu/performance_gpu_all.json` (`collect_{npu,gpu}_results.jsonl` 540행과 동일 순서) |
| 배치 정의(스케줄) | `xgboost_model/schedules/collection/collect_cpu_{npu,gpu}.yaml` (+ `.meta.json`의 set/n_accel) |
| out-of-fold 예측 | scratchpad `collect/oof_{npu,gpu}.json` (train_npu.py / train_gpu.py가 저장) |

fold 분할은 `out_of_fold_predictions(folds=3, seed=42)`의
`RandomState(42).randint(0,3,540)`을 **행 순서 그대로** 재현했다. 행 순서 = 창 파일
순서이며, oof 파일의 `models`/`rate` 배열과 540행 전부 일치함을 확인(불일치 0)했으므로
fold 재현은 정확하다. 두 플랫폼의 창 파일은 조합 순서까지 동일하고 배치도 정렬되어
있어(CPU 배치 집합 불일치 0), 아래 수치는 **두 예측기에 동일하게** 적용된다.

참고: 최종 아티팩트(`deploy_cpu_{npu,gpu}_*`)는 540행 **전체**로 학습되므로 "train
포함 여부"는 자명하게 성립한다. 아래 train/test 구분은 검증에 쓴 3-fold out-of-fold
기준이다(각 fold가 한 번씩 test, 나머지 두 fold가 train).

사전 무결성 체크: qwen2_vl CPU 배치 위반 0건, meta의 `n_accel`과 실제 배치 불일치
0건, 재계산한 `y3_valid`와 oof 기록 불일치 0건.

## 1. 전체 커버리지 (두 예측기 공통, 540 조합)

| 항목 | 값 |
|---|---|
| CPU 배치 조합 (자유배치 모델 ≥1개가 CPU) | **495 / 540 (91.7%)** |
| "전부 CPU" 조합 (가속기 배치 0, 층 0 극단) | **45 / 540 (8.3%)** — 15세트 × 3 rate, 세트마다 포함 |
| "전부 가속기" 조합 (CPU 배치 0) | 45 / 540 (반대편 극단도 세트마다 3개씩) |

층화 축("가속기 배치 모델 수")이 양 극단을 모든 세트에서 강제했기 때문에, CPU 배치가
전혀 없는 조합은 정확히 "전부 가속기" 층뿐이다.

## 2. 모델별 CPU 배치 빈도 (train / test 분리)

전체 빈도와, 3-fold 각각에서 test(그 fold) / train(나머지 두 fold)에 들어간 횟수:

| 모델 | 전체 | f0 te/tr | f1 te/tr | f2 te/tr |
|---|---|---|---|---|
| llama1b | 195 | 62/133 | 64/131 | 69/126 |
| mobilenet_v2 | 195 | 63/132 | 75/120 | 57/138 |
| resnet50 | 261 | 89/172 | 83/178 | 89/172 |
| yolo11l | 120 | 40/80 | 36/84 | 44/76 |
| yolo11m | 174 | 55/119 | 62/112 | 57/117 |
| yolo11n | 72 | 28/44 | 23/49 | 21/51 |
| yolo11s | 240 | 84/156 | 79/161 | 77/163 |
| yolo11x | 84 | 31/53 | 23/61 | 30/54 |
| qwen2_vl | **0** (제약상 정상) | — | — | — |

**train·test 어느 쪽에도 0인 셀이 없다.** 가장 적은 yolo11n(작은 세트 3개에만 등장)도
fold별 test에 21~28개, train에 44~51개 있다.

## 3. fold별 CPU 배치 조합 분포

| fold | 크기 | test: any-CPU (전부-CPU) | train: any-CPU (전부-CPU) |
|---|---|---|---|
| 0 | 183 | 166/183 (17) | 329/357 (28) |
| 1 | 180 | 167/180 (16) | 328/360 (29) |
| 2 | 177 | 162/177 (12) | 333/363 (33) |

세 fold 모두 test의 91~93%가 CPU 배치 조합이고, "전부 CPU" 극단도 fold마다 12개
이상 test에 등장한다. 특정 fold 몰림 없음.

## 4. llama1b CPU 배치 — y1/y2 학습 포함 + y3 마스킹

- llama1b를 CPU에 배치한 조합: **195개** (fold별 test 62 / 64 / 69 — 고른 분포).
- **y3 마스킹은 12개뿐**: llama1b가 있는 10개 세트 중 9개는 qwen2_vl을 함께 포함하고
  qwen2_vl은 항상 가속기에 있으므로, llama1b가 CPU여도 그 조합의 y3(합산 토큰)은
  유효 측정으로 남는다. 마스킹되는 것은 qwen2_vl이 없는 유일한 LLM 세트
  S3(llama1b,resnet50,yolo11s)에서 llama1b가 CPU인 12개 조합(fold별 6/3/3)이다.
  전체 y3 마스킹 168행 = 이 12행 + vision-only 156행 (보고서의 168과 일치).
- **y1/y2는 195개 전부 학습·검증에 포함**: 마스킹은 y3에만 적용되며(`train_targets`,
  `out_of_fold_predictions` 코드 확인), oof에서 llama-CPU 195행의 pred_y1/pred_y2에
  NaN이 0개 — 모두 out-of-fold로 예측(=test)되었고, 다른 fold에서는 train으로 쓰였다.
- 마스킹된 12행의 oof pred_y3는 12/12 NaN — 설계대로 y3 학습·검증에서 제외됨.

즉 "llama1b를 CPU에 두면 y1/y2가 어떻게 되는가"는 195개 조합으로 충분히 학습·검증
되었고, 노이즈인 CPU-LLM 토큰 처리량만 정확히 걸러졌다.

## 5. 세트별 CPU 배치 조합 수 (조합 = 배치 × 3 rate)

| 세트 | 모델수 | any-CPU / 전체 | 전부-CPU | 비고 |
|---|---|---|---|---|
| S1 (vision) | 2 | 9/12 | 3 | 자유배치 2 → 4배치 중 3이 CPU 포함 |
| S2 (vision) | 3 | 21/24 | 3 | |
| S3 (LLM) | 3 | 21/24 | 3 | 유일한 llama-without-qwen 세트, y3 마스킹 12 발생 |
| S4 (VLM) | 3 | 9/12 | 3 | 자유배치 2 (qwen2_vl 고정) |
| base1~5 (LLM+VLM) | 4 | 각 21/24 | 각 3 | |
| S5 (LLM+VLM) | 5 | 45/48 | 3 | |
| S6 (LLM+VLM) | 6 | 45/48 | 3 | 층화 |
| S7 (LLM+VLM) | 7 | 57/60 | 3 | 층화 |
| S8 (LLM+VLM) | 8 | 69/72 | 3 | 층화 |
| S9 (vision) | 5 | 45/48 | 3 | 층화 |
| S10 (vision) | 7 | 69/72 | 3 | 층화 |

15세트 전부에서 CPU 배치 조합이 75% 이상이고, "전부 CPU" 극단이 각 rate마다 1개씩
(세트당 3개) 들어 있다.

- **vision-only 세트(S1·S2·S9·S10)**: 156행 중 any-CPU 144 (92.3%) → 충분.
- **LLM/VLM 포함 세트(나머지 11개)**: 384행 중 any-CPU 351 (91.4%) → 충분.

## 6. 판정

**두 예측기(deploy_cpu_npu / deploy_cpu_gpu) 모두 CPU 배치 커버리지 충분 — 결함
없음.**

- CPU 배치 조합이 전체의 91.7%, 15세트 전부에 분포하고 "전부 CPU" 극단 포함.
- 3-fold train/test 양쪽 모두에서, **모든 모델의 CPU 배치가 0인 셀이 하나도 없다**
  (최소 셀: yolo11n test 21개).
- fold 간 몰림 없음(any-CPU 비율 91~93%로 균일).
- llama1b CPU 배치는 y1/y2로 195조합 학습·검증 완료, y3는 설계대로 노이즈 12행만
  마스킹.
- vision-only / LLM 포함 세트를 나눠 봐도 양쪽 다 91% 이상.

단, 두 가지는 유의사항으로 남긴다 (결함 아님):
1. **yolo11n의 CPU 배치 표본이 상대적으로 적다**(72조합; base1·S9·S10에만 등장).
   fold별 test 21~28개로 검증은 성립하지만, yolo11n 중심 세트를 새로 추가한다면
   표본이 가장 얇은 축이다.
2. **S3에서 llama1b를 CPU에 둔 12조합은 y3 라벨이 없다**(의도된 마스킹). 이 배치의
   y3 예측은 학습 근거가 없으므로, 점수식에서 y3 항을 쓰는 경우 S3-류(qwen 없는 LLM
   세트)의 CPU-LLM 배치 y3는 신뢰 대상이 아니다 — 현재 파이프라인은 해당 행의 y3
   예측을 NaN으로 남겨 이미 이를 반영한다.

### 데이터 관리 주의
본수집 원시 창 파일(각 540창)은 저장소가 아니라 이전 세션 scratchpad
(`/tmp/claude-1001/.../6f0af394-.../scratchpad/collect/`)에만 존재한다. 저장소의
`xgboost_model/performance_data/cpu_{npu,gpu}/performance.json`은 **160창짜리 파일럿
데이터**다. `/tmp`가 비워지면 본수집 원시 데이터와 oof 기록이 사라지므로, 저장소로
복사해 두는 것을 권장한다.
