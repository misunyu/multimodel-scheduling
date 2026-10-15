# Runtime overhead of the placement decision

측정일 2026-10-14. 코드 `code/runtime_overhead.py`, 원자료 `runtime_overhead.json`, 표 `runtime_overhead.csv`.

## 조건

- **장치**: CPU만 사용(`CUDA_VISIBLE_DEVICES=""`, XGBoost `device=cpu`). Intel Core Ultra 9 285K, 논리 CPU 24개.
  - XGBoost `nthread` = 0(설정하지 않음, 가용 thread 전부 사용). load 후 `n_jobs` 파라미터는 None.
  - 측정 전 load average 0.15, GPU에 다른 compute process 없음.
- **버전**: Python 3.10.12, xgboost 3.0.5, numpy 2.2.6, pandas 2.3.3.
- **코드 경로는 `revision_exp`와 같다.**
  - feature는 기존 `featurize_window`로 만들고 학습 때의 204열 순서로 맞춘다.
  - 예측은 Phase 2 artifact(`out_reprofiled/artifacts/multi/{T,D}.json`)에 대해 `predict.predict_multi`를 쓴다.
  - 점수는 Ŝ = T̂ − 0.3·D̂(clip 없음)이고, 정렬 규칙은 `evaluate.select`와 같다.
- **입력**: decision에는 schedule(모델과 요청 rate)만 들어가고 측정값은 쓰지 않는다.
- **검증**: 이 경로의 T̂·D̂가 Phase 2 평가 예측(`out_reprofiled/predictions_x3.csv`)과 128개, 8개 후보 모두에서 같다(최대 차 1e-16). 선택 배치도 Phase 2의 Ours와 같다(N=7 GGGGCCG, N=3 GGG).
- **예측 호출**: 모델마다 후보 전체를 한 번의 batch 호출로 예측한다(`XGBRegressor.predict(X)`, 후보별 루프 아님). Feature는 후보별 루프로 만든 뒤 DataFrame 하나로 합친다.
- **반복**: warm 100회의 중앙값(p10–p90)과, 모델 load를 포함한 첫 1회(cold)를 따로 잰다.

## 결과 (ms)

| 단계 | N=7 (128 후보) warm 중앙값 [p10–p90] | N=7 cold | N=3 (8 후보) warm 중앙값 [p10–p90] | N=3 cold |
|---|---|---|---|---|
| (a) 후보 열거 | 0.071 [0.063–0.075] | 0.065 | 0.005 [0.005–0.006] | 0.009 |
| (b) feasibility 검사 | 0.041 [0.035–0.043] | 0.040 | 0.003 [0.003–0.003] | 0.004 |
| (c) feature 생성 | 53.023 [51.269–55.167] | 52.524 | 3.086 [3.037–3.192] | 3.432 |
| (d) 예측 T̂·D̂ | 21.632 [20.578–27.692] | 28.637 | 1.892 [0.844–17.062] | 20.423 |
| (e) score·정렬 | 0.014 [0.012–0.018] | 0.014 | 0.008 [0.008–0.009] | 0.015 |
| **합계 (a–e)** | **75.250 [72.512–81.048]** | 81.280 | **4.996 [3.955–20.203]** | 23.884 |
| 모델 load (T, D) | – | 64.944 | – | 62.968 |
| 합계 + load | – | 146.224 | – | 86.851 |
| 후보 1개: feature + 예측 | 1.399 [1.386–1.472] | – | 1.072 [1.057–1.129] | – |

- **Decision time = (a)–(e) 합계**: N=7 128개 후보에서 약 75 ms, N=3에서 약 5 ms다. 예측 단계만 떼어 decision time이라 부르지 않는다. 예측은 N=7에서 전체의 29%이고, feature 생성이 70%다.
- **Cold**: 모델 load(약 63–65 ms)가 한 번 더해진다. Python 모듈 import(기존 suite 포함 약 358 ms)는 process 시작 비용이라 따로 기록했다.
- **N=3의 예측 p90(17 ms)**: 8행 예측에 24 thread를 쓰는 XGBoost의 thread pool 지연으로 보인다 [추정]. 중앙값은 1.9 ms다.
- 후보 1개를 따로 처리하면 1.1–1.4 ms라서, 128개를 하나씩 처리하면 약 180 ms가 된다. Batch 처리가 그보다 빠르다 [추정: 128 × 1.399 ms와의 비교].
