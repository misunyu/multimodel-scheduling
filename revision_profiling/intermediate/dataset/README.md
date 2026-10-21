# Phase 4 dataset (중간 rate 1.5×, 2.5×, 3.5×)

원본 `code/recompute_scores.py`를 그대로 실행해 만들었다.

```
cd revision_profiling
../.venv/bin/python3 -B code/recompute_scores.py --results intermediate/results --out intermediate/dataset --train-mults 0 --test-mults 1.5,2.5,3.5
```

- 결과: valid runs 1440, invalid 0, workloads 36, incomplete 0, excluded 0.
- 스키마는 Phase 2 `dataset/`와 같다(S = T − 0.3·D_macro, T = F/F_max, clip 없음).
- **파일 이름 주의**: 원본 스크립트가 출력 이름을 `test_x3.csv`, `test_schedules_x3.csv`로 고정해 쓴다. 이 폴더의 `test_x3.csv`에는 **세 rate(1.5×, 2.5×, 3.5×) 1,440행 전부**가 들어 있다. rate는 `labels_test.csv`의 `rate`·`multiplier` 열로 구분한다.
- `train_*.csv`, `labels_train.csv`는 헤더만 있다(이 Phase는 평가 전용이라 학습 데이터가 없다; `--train-mults 0`).
