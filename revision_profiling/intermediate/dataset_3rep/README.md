# dataset_3rep — Phase 4b 중간 rate 3회 평균 label

원본 `revision_profiling/code/recompute_scores.py`(수정 없음)로 생성:

```
cd revision_profiling
../.venv/bin/python3 -B code/recompute_scores.py --results intermediate/results intermediate/results_rep1 intermediate/results_rep2 \
    --out intermediate/dataset_3rep --train-mults 0 --test-mults 1.5,2.5,3.5
```

- 입력: rep0 = Phase 4 `intermediate/results/`(2026-10-19), rep1·rep2 = Phase 4b `results_rep1/`, `results_rep2/`(2026-10-20~21).
- valid runs 4,320, invalid 0, workloads 36, incomplete 0, excluded 0 (`logs/integrity_repeats/exclude_workloads.txt` 비어 있음 → `--exclude-workloads` 미사용).
- Label 정의는 Phase 2 §3.1과 같다: F̄ = mean_r F_r, D̄_macro = mean_r D_macro,r, F_max = 후보 중 max F̄, T = F̄/F_max, S = T − 0.3·D̄_macro.
  `*_rep<r>` 열은 반복별 값(S_r는 공통 F_max 사용), `*_se`는 표본표준편차(ddof=1)/√3.
- 파일 이름 `test_x3.csv`, `test_schedules_x3.csv`는 원본 코드가 고정한 이름이다("x3" = 3회 반복, rate 3×가 아님). `train_*`는 비어 있다(`--train-mults 0`).
- 평가: `revision_exp_adapter/run_intermediate_3rep.py --final` → `revision_exp/out_intermediate_3rep/`.
