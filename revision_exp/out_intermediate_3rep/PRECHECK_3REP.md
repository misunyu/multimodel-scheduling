# PRECHECK_3REP — Phase 4b §0 (측정 전)

실행 2026-10-20 14:51:21. 코드 `revision_exp_adapter/run_intermediate_3rep.py --precheck`, label 코드 `revision_profiling/intermediate/code_copy_repeats/labels_3rep.py`.

**결과: PASS — rep1 측정 시작 가능**

## 검사

| 검사 | 결과 |
|---|---|
| 0-1 Phase 2 3x: 22 workloads, 504 candidates, all rows matched | PASS |
| 0-1 F_bar, D_bar_macro, T, S, F_max, SE and per-repeat columns equal Phase 2 labels_test.csv (max abs diff <= 1e-9) | PASS |
| 0-1 D_macro_r (run value) = mean_i q_{i,r}, and D_bar_macro = mean_r mean_i q_{i,r} (<= 1e-12) | PASS |
| 0-1 S SE median (3x N>=3, 480 candidates) = 0.0010 (Phase 2 INTEGRITY.md / REPORT_PHASE2.md); p90 0.0027, max 0.0130 | PASS |
| 0-1 per-repeat oracle ids (own normalisation) = Phase 2 results/integrity/oracle_x3.csv, 'all three equal' 8/12 | PASS |
| 0-2a Phase 4 PRECHECK table (precheck_table.csv) reproduced exactly | PASS |
| 0-2a Phase 4 PRECHECK candidate predictions (precheck_predictions_x3.csv) reproduced exactly | PASS |
| 0-2b Phase 4 rep0 table_intermediate.csv (15 rate rows) reproduced exactly | PASS |
| 0-2b Phase 4 rep0 per_workload_intermediate.csv (36 rate rows) reproduced exactly | PASS |
| 0-2b Phase 4 rep0 mae_intermediate.csv reproduced exactly | PASS |
| 0-2b Phase 4 rep0 predictions_intermediate.csv reproduced exactly | PASS |
| 0-2b Best-of-5 with the stored draws = baselines.best_of_k on rep0 (bit-identical, 3 rates) | PASS |
| 0-3 runtime files sha256 = Phase 4 logs/runtime_sha256.json (10 files, direct comparison) | PASS |
| 0-3 predictor artifacts (T, D, t_ratio, q_i) sha256 = Phase 4 precheck.json | PASS |
| 0-3 offline profile sha256 = git HEAD blob (Phase 4 recorded no profile hash; Phase 4 RUN_PLAN checked 'unchanged vs git HEAD') | PASS |

## 0-1 3회 평균 label 코드

- Label은 원본 `code/recompute_scores.py`의 `load_runs()`·`build_tables()`(수정 없음)가 만든다. Phase 2 3× label을 만든 함수와 같다.
- 정의: 후보별 `F̄ = mean_r F_r`, `D̄_macro = mean_r D_macro,r`(`D_macro,r` = run의 `system.drop_macro` = instance별 `q_{i,r}`의 평균; 아래 표에서 차 ≤ 1.2e-16 확인), workload별 `F_max = max_cand F̄`, `T = F̄/F_max`, `S = T − 0.3·D̄_macro`.
- **반복별 `S_r`의 정규화 분모 (Phase 2 구현 확인)**: SE·반복 편차에 쓰는 `S_r`(`S_macro_a0.3_rep<r>`)는 `F_r / F_max − 0.3·D_macro,r`이고, `F_max`는 **3회 평균 `F̄`의 공통 최대값**이다(`recompute_scores.build_tables` 91행 `S_rep = F / fmax - ALPHA * Dm`, `fmax = max(F.mean())`). SE = 표본표준편차(ddof=1)/√3(`recompute_scores.se`). 반면 **반복별 oracle**은 Phase 2 `code/integrity.py` 151–158행대로 **각 반복 자체의 정규화** `F_r / max_cand F_r − 0.3·D_macro,r`로 구한다(단일 측정 campaign이 냈을 oracle). 이 Phase도 두 정의를 그대로 쓴다.
- Phase 2 3× 원시 반복 1464 runs → 후보 504개, Phase 2 `dataset/labels_test.csv` 504행과 1:1 대조. 최대 절대 차:

| 열 | 최대 차 |
|---|---|
| F_mean | 2.27e-13 |
| D_macro_mean | 9.71e-17 |
| throughput_norm | 1.11e-16 |
| S_macro_a0.3 | 1.11e-16 |
| F_max | 2.27e-13 |
| F_se | 1.78e-15 |
| D_macro_se | 9.97e-17 |
| S_macro_a0.3_se | 9.95e-17 |
| F_rep0 | 0 |
| F_rep1 | 0 |
| F_rep2 | 0 |
| D_macro_rep0 | 9.71e-17 |
| D_macro_rep1 | 9.71e-17 |
| D_macro_rep2 | 9.71e-17 |
| S_macro_a0.3_rep0 | 1.11e-16 |
| S_macro_a0.3_rep1 | 1.11e-16 |
| S_macro_a0.3_rep2 | 2.22e-16 |

- `D_macro,r` vs `mean_i q_{i,r}`: {'rep0': 1.1102230246251565e-16, 'rep1': 1.1102230246251565e-16, 'rep2': 1.1102230246251565e-16, 'mean': 1.1102230246251565e-16}

- 반복 편차 재현(3× N≥3, 후보 480개; Phase 2 `results/INTEGRITY.md` §6: S의 SE 중앙값 0.0010, 90% 0.0027, 최대 0.0130):

| 지표 | 후보 | 중앙값 | 90% 분위 | 최대 |
|---|---|---|---|---|
| F (fps) | 480 | 0.9046 | 2.8587 | 9.1510 |
| F relative (SE / F_bar) | 480 | 0.0010 | 0.0030 | 0.0135 |
| D_macro | 480 | 0.0006 | 0.0016 | 0.0089 |
| S | 480 | 0.0010 | 0.0027 | 0.0130 |

- 반복별 oracle(자체 정규화, 동점 1e-9): 대표 ID가 Phase 2 `results/integrity/oracle_x3.csv`와 36개(12 × 3) 모두 같다. 세 반복 대표 ID가 같은 workload 8/12(Phase 2 8/12), 세 반복에 공통으로 최적인 후보가 있는 workload 8/12(`precheck_phase2_rep_oracles.csv`).

## 0-2 평가 경로

- (a) Phase 4 adapter의 `evaluate_set()`(import, 수정 없음)로 Phase 2 3× 데이터를 평가해 Phase 4 `precheck_table.csv`·`precheck_predictions_x3.csv`를 같은 CSV 저장·읽기 후 비교: 표 최대 차 0, 예측 최대 차 0(불일치 칸 0, 0). Phase 4 PRECHECK가 Phase 3 값과 같음을 이미 확인했으므로 Phase 3 값도 재현된다.
- (b) 이 adapter의 `evaluate_rates()`로 Phase 4 rep0 dataset을 평가해 Phase 4 `table_intermediate.csv`(rate 15행), `per_workload_intermediate.csv`(rate 36행), `mae_intermediate.csv`, `predictions_intermediate.csv`와 비교: 최대 차 표 0, workload별 0, MAE 0.
- Best-of-5는 Phase 4 추출(`SeedSequence([42, t, w])`)을 후보 ID로 저장해 두고, 다섯 중 최적 선택에만 label을 쓰는 함수로 바꿨다. rep0에서 이 함수가 `baselines.best_of_k`와 bit 단위로 같다.

## 0-3 동일성

| 파일 | sha256 | 대조 |
|---|---|---|
| `revision_profiling/code/executor.py` | `81967524be10ad97ef27388991cbedb232ce1966562fce737399b618aa14ca23` | Phase 4 `logs/runtime_sha256.json` |
| `revision_profiling/code/feeder.py` | `5bb076af58a1d2d419541d7529073ed481e55944e86d13a314cb6699c3cff514` | Phase 4 `logs/runtime_sha256.json` |
| `revision_profiling/code/counters.py` | `09f7792cab56caf2f167b9bf693ce800b9b444dc406de6617b3567f1d32c9e4b` | Phase 4 `logs/runtime_sha256.json` |
| `revision_profiling/code/shmqueue.py` | `4dda31c9b73e566c408601995cdad154db3cb4502a0b98801f464c97679c842e` | Phase 4 `logs/runtime_sha256.json` |
| `revision_profiling/code/worker_base.py` | `d0e6fbd2de7ca07c68ad5ce03c65961a9a792ea6120d3d54aa1769b8c21d8273` | Phase 4 `logs/runtime_sha256.json` |
| `revision_profiling/code/worker_cls.py` | `6dafff59495fbf0142e28007790a1d645f14ebf07144a5754594ea6bf6dfa48d` | Phase 4 `logs/runtime_sha256.json` |
| `revision_profiling/code/worker_yolo.py` | `019125644351675ffb23d3a6dfa74721436ec279cb9f7d539720cbf199e813fb` | Phase 4 `logs/runtime_sha256.json` |
| `revision_profiling/code/lm_runner.py` | `4b71b9e43632d5f841d22e69375e35fd943f8fb08653fa415e984a4a52369410` | Phase 4 `logs/runtime_sha256.json` |
| `revision_profiling/code/image_processing.py` | `61274606b9d1cddbef45318a39b6caacdbbfbcd997da35e22cf65a5fe86bf7b6` | Phase 4 `logs/runtime_sha256.json` |
| `revision_profiling/code/model_registry.yaml` | `32119c2bb49ade1231a02dc6cfcb0fc1ca9ff66e8b4c5a3e44438eadfad1ff1c` | Phase 4 `logs/runtime_sha256.json` |
| `T.json` | `b6a7ce2d159f9e9874a93bc1ac1a4124a9b1aa642db2600468d2e4b45c3b20fa` | Phase 4 `precheck.json` |
| `D.json` | `ab19e2ad2d0614b919799ff7d5dc0c29d3793c4b1859732ac56bfa3ec8bf9334` | Phase 4 `precheck.json` |
| `per_instance/t_ratio.json` | `2a19ee15dc58a8ee1d92013c552310baef18e838541758ef3d081778874388de` | Phase 4 `precheck.json` |
| `per_instance/q_i.json` | `b28007d87cfffd8530e914f4518d19f625fa7eac1258425c21ab24c31880bbcc` | Phase 4 `precheck.json` |
| `revision_profiling/offline_profile/offline_profile.json` | `9c24f9a154212ee2010c96166742baf036fdc038677be5fab0d878093a345f0d` | git HEAD blob `9c24f9a15421…`. Phase 4 PRECHECK에는 profile hash 기록이 없다. Phase 4 RUN_PLAN이 'git HEAD와 같음'을 확인했으므로 같은 기준으로 비교했다 |
