# 본수집(full collection) 540조합 원시 데이터 백업

`deploy_cpu_npu` / `deploy_cpu_gpu` 예측기를 학습한 **본수집 원본**이다.
수집: 2026-07-15 ~ 07-18, 각 플랫폼 540조합, 실패 0
(보고서: `FULL_COLLECTION_NPU_REPORT.md`, `FULL_COLLECTION_GPU_REPORT.md`,
커버리지 감사: `CPU_PLACEMENT_COVERAGE.md`).

원본은 세션 scratchpad(`/tmp/claude-1001/.../6f0af394-.../scratchpad/collect/`)에만
있었고, /tmp 소실 대비로 2026-07-20 저장소에 복사했다 (sha256 검증 완료).

## ⚠️ `performance_data/`와 혼동 주의

`xgboost_model/performance_data/cpu_{npu,gpu}/performance.json`은 **160창짜리
파일럿 데이터**다 (2026-07-09~10). 현행 예측기 아티팩트는 그 데이터가 아니라
**이 디렉토리의 540창 데이터**로 학습됐다. 이 디렉토리를 `performance_data/` 밖에
둔 이유: `build_dataset`은 perf_dir 아래 `performance*.json`을 재귀로 전부 읽으므로,
파일럿과 같은 트리에 두면 재학습 시 160창 파일럿과 540창 본수집이 섞인다.

## 구성

```
cpu_npu/ , cpu_gpu/            # 플랫폼별 동일 구성
  collect_*_results.jsonl      # 조합당 1행 × 540 — 수집 요약(y1/y2/y3, 모델별 배치·fps)
  performance_*_full540.json   # 540창 windows 파일 — build_dataset이 그대로 읽는 학습 입력
  oof_*.json                   # out-of-fold 예측 (3-fold, seed=42) — 보고서 Spearman의 근거
  collect_*_anchor.jsonl       # anchor(전부-NPU/GPU S3) 반복 측정 — 드리프트 감시 기록
scripts/
  collect_run.py               # 수집 러너
  jsonl_to_windows.py          # results.jsonl → windows JSON 변환 (performance_*_full540.json 생성)
  train_npu.py , train_gpu.py  # 최종 학습 + oof 저장 (경로는 당시 scratchpad 기준 — 재사용 시 수정 필요)
  compare_platforms.py         # NPU vs GPU 최적 배치 비교 (보고서 7절)
```

## 재학습 방법

```bash
python xgboost_model/deploy_selector_xgb_suite.py train \
  --perf_dir xgboost_model/full_collection_540/cpu_npu \
  --schedule_dir xgboost_model/schedules/collection \
  --static_json xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json \
  --model_out xgboost_model/artifacts/deploy_cpu_npu \
  --rate_factors 1.0 1.5 2.0 3.0 4.0
```

(cpu_gpu도 동일, 경로만 교체. `--rate_factors`를 생략하면 기본값 1/2/4×로 걸러져
1.5×·3× 행이 빠지므로 반드시 전 rate를 명시할 것. 원래 학습은 `scripts/train_npu.py`
처럼 `build_dataset(..., normalize=True)`를 rate 필터 없이 직접 호출했다.)

fold 재현: `out_of_fold_predictions(folds=3, seed=42)` —
`RandomState(42).randint(0,3,540)`을 windows 파일의 행 순서에 그대로 적용.
