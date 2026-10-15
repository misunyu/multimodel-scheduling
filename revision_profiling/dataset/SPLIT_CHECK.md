# Split 점검 (Phase 2 v1.2 §3.2)

출처: `dataset/compat_check.json` (`code/check_dataset_compat.py --data dataset --final`).

| | train (`train_x3.csv`) | test (`test_x3.csv`) |
|---|---|---|
| rate | 1×, 2×, 4× | 3× |
| workloads | 66 | 22 |
| 그중 N≥3 | 36 | 12 (평가 대상) |
| 후보(행) | 1,512 | 504 |
| families | 22 | 22 |
| 중복 행 (workload, combination) | 0 | 0 |

- train에 3× 행: 0. test에 3× 이외 행: 0.
- 두 집합의 workload(family × rate) 겹침: 0. 행 id 겹침: 0.
- 같은 family가 양쪽에 있는 수: 22 / 22. 이것은 설계상 허용이다. 평가는 **학습에 없던 rate(3×)** 에 대한 held-out-rate 평가이고, 학습에 없던 모델 조합에 대한 평가가 아니다.
- 학습 내부의 hyperparameter 선택은 GroupKFold(3, family 기준)라서, 한 family의 1×·2×·4× 행은 항상 같은 fold에 들어간다(`revision_exp/out_reprofiled/folds_summary.csv`).
- 3× N≥3 후보 480개의 label은 3회 평균이고, 나머지는 1회 측정이다.
