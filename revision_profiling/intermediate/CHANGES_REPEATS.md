# CHANGES_REPEATS — Phase 4b (중간 rate rep1, rep2 반복 측정과 3회 평균 재평가)

Phase 4의 파일은 하나도 고치지 않았다. 대상은 `intermediate/code_copy/*`, `revision_exp_adapter/run_intermediate.py`, `CHANGES.md`, `RUN_PLAN.md`, `REPORT_INTERMEDIATE.md`, `schedules/`, `results/`(rep0), `dataset/`, `smoke/`, 기존 `logs/` 파일, `revision_exp/out_intermediate/`이다.

보호 대상의 sha256을 campaign 전후에 기록해 비교한다. `git status`는 untracked 파일의 변경을 잡지 못하므로 이 비교가 기준이다.
- 대상: 위 폴더 전부, Phase 2 `dataset/`·`results/`, predictor artifact 4개, offline profile, runtime 10개, `recompute_scores.py`·`run_campaign.py`·`gen_schedules.py`
- 시작: `logs/protected_sha256_start.json`(campaign 전, 10,411 files)
- 종료: `logs/protected_sha256_end.json`
- 코드: `code_copy_repeats/protect.py`

## 1. 그대로 쓴 것 (import만, 수정 없음)

| 용도 | 파일 |
|---|---|
| 측정 runtime | 원본 `revision_profiling/code/executor.py`와 모듈 10개(Phase 4와 같은 파일; 블록 시작마다 sha256을 Phase 4 `logs/runtime_sha256.json`과 직접 비교) |
| Schedule | Phase 4 `intermediate/schedules/*.yaml` 36개(재생성 없음, `logs/schedule_sha256.json`에 기록, runner 시작 시 대조) |
| Family 목록·경로·시간식 | Phase 4 `code_copy/intermediate_plan.py`(`families`, `perm`, `tag`, `est_sec`, `SCHED`) |
| Resume 판정 | 원본 `code/run_campaign.py: status()` |
| Label | 원본 `code/recompute_scores.py`(`load_runs`, `build_tables`, CLI) — Phase 2 3× label을 만든 코드 |
| 평가 | Phase 4 adapter `run_intermediate.py`의 `load_dataset`, `evaluate_set`, `evaluate_set_rate`, `summary_row`, `mae`, `frozen_columns`, `load_models`(import). 그 아래 `revision_exp/{data,evaluate,baselines,predict,run_all}.py`, phase3 `instance_rows`·`compose_select`도 그대로 쓴다 |

## 2. 새 파일 (`intermediate/code_copy_repeats/`)

| 파일 | 내용 | 원본과의 관계 |
|---|---|---|
| `repeats_plan.py` | rep1·rep2 run list와 `RUN_PLAN_REPEATS.md` | Phase 4 `intermediate_plan.jobs()`의 rep0 블록을 rep r 블록 두 개로 바꿨다. family 순서는 `SeedSequence([43, r])`, family 안은 `SeedSequence([43, r, family_index])`(지시문 §1; Phase 2 원본은 `[42, 0, r]`, `[42, family_index, r]` 형태였다). 결과 경로는 `results_rep<r>/`. 검사 항목: rep0 후보 집합 동일, schedule 36개 동일, family_index가 RUN_PLAN.md 표의 번호 |
| `run_repeats_campaign.py` | runner | Phase 4 `run_intermediate_campaign.py`의 사본이다. 바뀐 점: (1) `import repeats_plan`; (2) run 수 2,880 assert; (3) 시작 시 schedule sha256 대조; (4) **블록이 바뀔 때마다** runtime sha256을 Phase 4 `logs/runtime_sha256.json`과 직접 비교하고(원본은 git HEAD 비교만), offline profile을 git HEAD blob과 비교해 `logs/runtime_sha256_rep<r>.json`에 기록; (5) 로그 파일 이름을 새로 정함(아래 §3); (6) 출력 문구 `[phase4b]` |
| `labels_3rep.py` | 3회 평균 label 검증, 반복 편차 통계, 반복별 oracle | label은 원본 `recompute_scores.build_tables`를 그대로 부른다. 반복별 oracle은 Phase 2 `code/integrity.py` 151–158행의 규칙(반복 자체 정규화)에 **oracle 집합**(동점 1e-9)과 세 반복 공통 최적 여부를 더했다(지시문 §2) |
| `integrity_repeats.py` | 무결성 점검 | Phase 4 `integrity_intermediate.py`와 Phase 2 `integrity.py`의 규칙을 rep별로 적용한다. 완전성은 (workload, rep) 단위로 보고, 하나라도 빠지면 workload를 제외한다. 반복 편차는 rate별로 Phase 2 3×와 나란히 둔다. 반복별 oracle 일치를 본다. 시간 경과 표시 규칙: 전부-GPU 후보의 |F_r − F̄|/F̄ > 5%를 "급변"으로 표시하며, campaign 종료 전에 정했다 |
| `protect.py` | 보호 대상 sha256 시작·종료 비교 | 새 파일 |

## 3. 경로·로그 이름 (지시문 문구와 다른 곳)

- **로그 파일 이름**: Phase 4의 `logs/campaign_progress.log`, `gpu_1min.csv`, `campaign_sessions.log`, `campaign_runs/`는 보호 대상이라 덧붙여 쓰지 않았다. 대신 다음 새 이름을 쓴다.
  - `logs/campaign_progress_repeats.log`, `logs/gpu_1min_repeats.csv`
  - `logs/campaign_sessions_repeats.log`, `logs/campaign_runs_repeats/`
  - `logs/campaign_repeats_stdout.log`, `logs/campaign_repeats.pid`
  - `logs/exec_order_rep{1,2}.csv`, `logs/failed_runs.csv`, `logs/run_plan_repeats.csv`, `logs/schedule_sha256.json`
- **무결성 보고서 위치**: 지시문 §2는 `results/INTEGRITY_REPEATS.md`라고 했다. 그러나 `results/`는 Phase 4 rep0 결과 폴더(보호 대상)이고, §4-5의 허용 경로에도 없다. 그래서 `logs/INTEGRITY_REPEATS.md`와 `logs/integrity_repeats/*.csv`에 둔다.
- **`tmux` 없음**(Phase 2·4와 같음): `setsid nohup`으로 띄웠다. `$!`가 setsid 앞단의 pid(1512899)를 가리켜, 세션 로그의 pid(1512901)로 `campaign_repeats.pid`를 다시 썼다. 실행에는 영향이 없다.

## 4. 평가 adapter (`revision_exp_adapter/run_intermediate_3rep.py`, 새 파일)

- `--precheck`: §0 → `revision_exp/out_intermediate_3rep/PRECHECK_3REP.md`, `precheck_3rep.json`. campaign 전 첫 실행에서 모든 검사가 PASS했다.
- `--final`: §3.
- Best-of-5:
  - 라이브러리 `baselines.best_of_k`는 추출과 선택을 한 함수에서 한다.
  - 이 adapter는 Phase 4의 추출(`SeedSequence([42, t, w])`, w = Phase 4 rate 집합의 정렬된 workload 순번)을 후보 ID로 저장한다. 그다음 다섯 중 최적 선택만 3회 평균 label로 한다(`bo5_samples`, `bo5_from_samples`).
  - rep0에서 이 함수가 `best_of_k`와 bit 단위로 같음을 PRECHECK에서 확인했다. 결측 workload가 생겨도 w를 다시 매기지 않는다.
- 선택 불변 검증: rep0 dataset과 3회 평균 dataset에서 다음이 모두 같아야 한다. 하나라도 다르면 평가를 중단한다(`selection_invariance_check.csv`).
  - 후보별 feature, Ours의 T̂·D̂·Ŝ, 예측 순위, 선택 ID
  - Per-instance의 F̂·D̂·T̂_select·Ŝ와 선택 ID
  - Latency 선택 ID
  - Best-of-5 추출 ID
- Dataset: 원본 `recompute_scores.py`를 `--results intermediate/results intermediate/results_rep1 intermediate/results_rep2 --out intermediate/dataset_3rep`로 실행한다. 원본은 출력 이름을 `test_x3.csv`로 고정한다.
