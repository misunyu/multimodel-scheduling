# CHANGES — Phase 4 (중간 rate 1.5×, 2.5×, 3.5×)

원본 파일(`revision_profiling/code/`, `revision_exp/`, `revision_exp/out_reprofiled/`, `revision_exp/out_phase3/`, `revision_exp_adapter/run_reprofiled.py`)은 하나도 수정하지 않았다. 아래는 모두 새 파일이다.

## 1. 원본을 그대로 쓴 것

| 용도 | 파일 | 비고 |
|---|---|---|
| 측정 runtime | `revision_profiling/code/executor.py`와 그 모듈(`feeder, counters, shmqueue, worker_base, worker_cls, worker_yolo, lm_runner, image_processing`, `model_registry.yaml`) | Runner가 subprocess로 원본 `executor.py`를 호출한다. 시작 시 10개 파일이 git에 tracked이고 HEAD와 같음을 assert(`logs/runtime_sha256.json`). Executor 자체가 저장소 모듈이 `revision_profiling/code/` 밖에서 load되면 중단한다 |
| Schedule 생성 | `code/gen_schedules.py: build()`, `reconstructed_families()`, `gpu_throughput()` | `build(..., mult)`가 이미 실수 배수를 받는다(`rate = beta * base * mult`). **정수 배수 제한이 없어 사본을 고치지 않았다** |
| Resume 판정 | `code/run_campaign.py: status()` | |
| Label·dataset | `code/recompute_scores.py` | rate 이름이 `x{m:g}`라 `x1.5` 등을 그대로 처리한다 |
| 평가 | `revision_exp/{data,evaluate,baselines,predict,run_all}.py`, `revision_exp_adapter/run_reprofiled.py`(보조 함수), `revision_exp/phase3/baselines/{instance_rows,compose_select}.py` | 경로만 run time에 지정(`D.DS`, `D.SETS`) |

## 2. 사본과 바뀐 점 (`intermediate/code_copy/`)

| 사본 | 원본 | 바뀐 곳 | 이유 |
|---|---|---|---|
| `intermediate_plan.py` | `code/phase2_plan.py` | `MULTS = (1.5, 2.5, 3.5)`(원본 33행 `(1, 2, 3, 4)`); N≥3 family만(원본은 22개 전부); seed 42 → **43**(원본 56·60행의 `SeedSequence([42, 0])`, `[42, family_index]`); rep1·rep2 블록 삭제(원본 63–69행); 출력 경로 `intermediate/{schedules,results}`; 파일 이름 `x{m:g}`(원본 `x{m}`); schedule 생성 직후 rate assert와 `schedules/rate_check.csv` 추가; 검증 항목을 1,440 runs 기준으로 교체 | 지시문 §1–2 |
| `run_intermediate_campaign.py` | `code/run_phase2.py` | import를 `intermediate_plan`으로(원본 25행); `LOGS = intermediate/logs`(원본 30행); `CODE_DIR`을 원본 `revision_profiling/code`로 고정해 원본 `executor.py`를 호출(원본 28행은 자기 폴더); 시작 시 run 수 1,440 assert와 runtime 파일의 git HEAD 동일성 assert, `logs/runtime_sha256.json` 기록(원본 96행 뒤); 출력 문구 `[phase2]` → `[phase4]` | 원본 runtime 유지, 경로 분리 |
| `integrity_intermediate.py` | `code/integrity.py` | 반복 블록 관련 항목(원본 129–185행: 3회 반복 편차, 반복별 oracle, rep 간 drift)과 1× 단일 instance 항목(186–195행) 삭제; generated 차이를 rep0 하나로(원본 97–121행의 rep loop); 추가: 모듈 경로 검사, scheduled tick − rate×10 분포, 전부-GPU 후보의 시간순 표와 확인 표시(F/Λ < 0.95 또는 실현율 < 0.95), Phase 2 3×와 나란히 보는 표와 비단조 표시 | 지시문 §2.1 |

Generated 차이·완전성·conservation·실현율의 판정 규칙은 원본과 같다.

## 3. 평가 adapter (`revision_exp_adapter/run_intermediate.py`, 새 파일)

- `--precheck`: §3.0. Phase 2 3× 데이터에 이 Phase의 평가 함수 `evaluate_set()`을 적용해 Phase 3 값과 대조 → `revision_exp/out_intermediate/PRECHECK.md`.
  - 첫 실행에서 내가 정한 검사 하나("Ours 후보별 예측의 메모리 값과 Phase 3 CSV 값의 차 = 0")가 최대 2.2e-16 차로 FAIL했다. 두 값을 같은 `to_csv`/`read_csv`로 거치면 504개 모두 차 0이라, pandas CSV 실수 표기의 1 ulp 반올림이다. 검사를 같은 경로의 값끼리 비교하도록 고쳤고(예측값·규칙 변경 없음), 두 번째 실행에서 모든 검사가 통과했다. 두 차이 값을 `precheck.json`에 모두 남겼다.
- `--final`: §3.1–3.3.
- `data.family_key()` / `rate_key()`(원본 `revision_exp/data.py:85-93`)는 옛 이름 `_x1-5`만 인식해 `..._x1.5.yaml`의 rate를 `x1`, family를 `<family>_x1.5`로 돌려준다. 이 두 열은 선택·feature에 쓰이지 않는다(feature는 `featurize_window`가 schedule의 infps와 model·device로만 만든다). Adapter는 loader 결과의 `family`·`rate`를 dataset의 `labels_test.csv` 값으로 바꿔 쓴다. 원본 `data.py`는 고치지 않았다.
- Per-instance 후보 행은 `instance_rows.rows_from_schedules()`가 schedule CSV 전체(3 rates)에서 만들므로, rate별 평가에서는 그 결과를 해당 rate의 workload로만 거른다(함수 자체는 바꾸지 않고 adapter 안에서 감쌈).

## 4. 실행 환경

- **`tmux`가 이 PC에 없다**(Phase 2와 같음). `setsid nohup`으로 띄웠다. PID `logs/campaign.pid`, 출력 `logs/campaign_stdout.log`. 터미널이 끊겨도 계속 돈다.
- 시작 직후 `campaign.pid` 기록이 셸 경로 문제로 실패해, 세션 로그(`logs/campaign_sessions.log`)의 pid로 다시 기록했다(실행에는 영향 없음).
- Campaign: 2026-10-19 16:01:52 – 23:27:57, 1,440 runs, 재시도 0, 실패 0.
- `integrity_intermediate.py`는 campaign 결과에 처음 실행한 그대로 끝났다(스크립트 수정 없음).
- Dataset은 원본 `recompute_scores.py`가 출력 이름을 `test_x3.csv`로 고정하므로, 그 파일에 세 rate가 모두 들어 있다(`dataset/README.md`).
- 측정 전 smoke run 1회(`intermediate/smoke/`, gpt2+mnasnet+resnet50 1.5× 전부-GPU)로 실수 배수 schedule이 원본 executor에서 valid임을 확인했다. 이 결과는 dataset에 넣지 않는다.
