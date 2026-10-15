# Campaign 전 `revision_exp` 호환성 사전 검증 (Phase 2 v1.2 §2.1b)

2026-10-13 19:29. **통과.** 학습·평가는 실행하지 않았다. 한 일은 parsing과 feature 생성뿐이다.

- 입력: Phase 1b pilot의 β=0.1 runs 60개(`pilot/arrival/runs` 36개 + `pilot/repeat_5_4/runs` 24개). 5 workloads, 후보 20개.
- 생성: `code/recompute_scores.py --allow-partial` → 이 폴더의 `train_x3.csv`(2× 행 14개), `test_x3.csv`(4× 행 6개), schedule CSV 2개. **파일명만 기존 규칙을 따랐을 뿐 실제 split이 아니다.** Pilot에는 1×·3×가 없어 2×를 train 자리, 4×를 test 자리에 넣어 loader를 통과시켰다.
- 검사: `code/check_dataset_compat.py` → `compat_check.json`. 수정하지 않은 `revision_exp/data.py`의 `load_set`·`build_features`와 기존 `featurize_window`를 그대로 불렀다(`D.DS`, `D.SETS` 두 변수만 실행 중에 바꿔 경로를 지정).
- 이 폴더의 데이터는 최종 학습·평가에 쓰지 않는다. `dataset/`과 섞지 않는다.

## 결과

| 항목 | 결과 |
|---|---|
| loader가 에러 없이 읽음 (train 자리 14행, test 자리 6행, 중복 제거 0행) | 통과 |
| CSV 열 = `schedule_file, combination, json_content` (주석 없음) | 통과 |
| Model 이름: registry 이름이 기존 dataset의 이름과 같아 `_get_model_features`의 hash bucket이 그대로 | 통과 |
| Device 이름: `GPU`/`CPU`(window), `gpu`/`cpu`(schedule) → `_norm_exec`가 둘 다 처리 | 통과 |
| Instance 순서 = model 이름 정렬순, combination 번호 = bit mask (20행 전부 대조) | 통과, 불일치 0 |
| Builder가 읽은 `infps` = 실제 요청 rate (114개 값 대조) | 통과, 불일치 0 |
| Leakage assert (`build_features`의 허용 이름 정규식 + 금지 token) | 통과 |
| 실측 counter·label 이름이 feature에 없음 | 통과, 0개 |
| `T` = `throughput_norm`, `D` = **D_macro** (최대 차이 1e-16) | 통과 |
| `derived`에 `drop_agg_norm`, `drop_max` 있음 | 통과 |
| D_macro ≠ D_agg인 행 | 20행 중 17행 (두 정의가 실제로 다른 값임을 확인) |
| group(schedule 파일명) ↔ `get_group_id` 1:1 | 통과 (`load_set` 내부 assert) |
| group = `<family>_<rate>.yaml` | 통과 |
| train 자리와 test 자리의 row·group 겹침 | 없음 |

## Feature column

| | 수 |
|---|---|
| 기존 (`revision_exp/out/feature_columns.txt`) | 235 |
| Pilot 데이터 | 175 |
| 공통 | 175 (공통 열의 순서 동일) |
| 새로 생긴 열 | 0 |
| 없어진 열 | 60 |

없어진 60개의 내역:

| 종류 | 수 | 이유 |
|---|---|---|
| `model_hash_17*` (sum·mean·max × 5) | 15 | TinyLLaMA(bucket 17) 제외 |
| `model_hash_24*` | 15 | YOLOv4(bucket 24)가 pilot family에 없음. **campaign 데이터에서는 다시 생긴다** |
| `pairhash_*` | 30 | 위 두 모델이 낀 쌍, 그리고 pilot에 없는 모델 조합 |

Campaign 데이터의 feature 수는 235보다 적고 175보다 많을 것이다. 정확한 수는 `dataset/README.md`에 적는다. Feature 정의는 바꾸지 않았다. 열 집합은 기존 방식대로 학습 데이터에서 나온 열로 정해진다.

모델 8종의 hash bucket은 서로 겹치지 않는다: mnasnet 22, resnet50 108, resnext50 78, shufflenet 69, squeezenet 45, vgg19 104, yolov4 24, gpt2 125.

## 발견해서 처리한 것 (수정 위치: recompute)

**Schedule layout.** Executor용 schedule(`schedules/*.yaml`)은 `meta` + `combinations.<id>.instances.<name>` 구조다. 기존 parser(`_rows_from_combo_struct`)는 이 구조에서 instance를 찾지 못해 `infps`가 전부 0이 된다. 에러 없이 조용히 0이 되는 경우라 따로 검사했다.
처리: `recompute_scores.py`가 dataset용 schedule CSV를 기존 layout(`combination_xxx → instance → model, execution, infps`)으로 써 준다. 숫자는 같다. Executor용 파일과 `revision_exp`는 건드리지 않았다. 검증 항목 "Builder가 읽은 infps = 실제 요청 rate"가 이것을 확인한다.

Adapter에서 고칠 것은 없다. `run_all.py`에 `--data_root` 인자가 없으므로 지시문대로 `revision_exp_adapter/`를 만들어 경로만 바꾼다.
