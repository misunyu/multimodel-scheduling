# CPU–NPU profiling 비용 (run별 실측 합)

- 읽기 전용 집계다. 새 측정과 코드 수정은 하지 않았다.
- 입력: worktree `../mms_npu_results` (detached @ `4070ce2e`)의 `revision_profiling_npu/`.
- 표: `profiling_cost_npu.csv`

**요약:** 학습 279 runs 3.08 h, 3× 평가 93 runs 1.03 h, 반복 120 runs 1.35 h, 전체 492 runs 5.46 h.

| 분류 | 정의 (run_plan.csv) | runs | 합계 (h) | 평균 (s) | 중앙값 (s) |
|---|---|---|---|---|---|
| 학습 split | block rep0, rate x1·x2·x4 | 279 | 3.08 | 39.71 | 39.3 |
| 3× 평가 1회차 | block rep0, rate x3 | 93 | 1.03 | 39.74 | 39.3 |
| 3× 반복 | block rep1 + rep2 (3× N≥3 60 후보 × 2) | 120 | 1.35 | 40.55 | 40.1 |
| **전체** | | **492** | **5.46** | 39.92 | 39.6 |

정밀값(h): 3.0778 / 1.0265 / 1.3515 / 5.4558.

## 원고 문장용 값

- 학습 split profiling 시간: **3.1 h** (279 runs)
- 전체 campaign 시간: **5.5 h** (492 runs)

## 참고: run 사이 overhead

- rep0의 실제 경과 시간과 run 시간 합의 차는 **약 0.9 s**다. run 372개 사이 overhead가 사실상 0이다.
  - rep0 경과 시간 14,776.4 s, run 합 14,775.5 s.
  - 이 차이는 `sec`의 0.1 s 반올림 오차 범위 안이다.
- **경과 시간을 잰 방법:** run JSON에는 wall-clock 시각이 없다. 그래서 같은 부팅의 monotonic 시각 `timing.feeder_start_ns`를 썼다.
  - 첫 rep0 run(#1)의 값부터 첫 rep1 run(#373)의 값까지를 쟀다.
  - 492개 값이 실행 순서대로 증가하는 것을 assert로 확인했다.
- **세션 전체:** `logs/campaign_sessions.log`의 SESSION START 08:53:48 UTC와 END 14:21:09 UTC로 327.4 min이다. run 합은 327.35 min이다.

## 사용한 필드와 파일

- **run 시간:** `results/runs/dma32_log.csv`의 `sec` 열이다.
  - INTEGRITY §1의 `hours`, `sec_p50`, `sec_mean`을 만든 것과 같은 열이다(`revision_profiling/code/integrity_npu.py`의 `dl`, 190–191행).
  - 정의(`revision_profiling/code/run_campaign_npu.py` 183–187행): run 하나의 subprocess 실행(`run_one`) 앞뒤 `time.time()` 차이다. 프로세스 시작부터 종료까지이고, 모델 로드·warm-up 5 s·window 30 s·종료가 들어 있다. 0.1 s 단위다.
- **campaign log 대조:** 같은 값이 `logs/campaign.log`의 run별 줄(`… status=ok 42.1s`)에도 있다. 492개 모두 run 이름과 초가 같다(assert).
- **분류:** `run_plan.csv`의 `block`과 `schedule`(family, rate)로 나눴다. run 이름이 `schedule__combination__rep{repeat}`와 같은지 assert했다.
- **label 대조:**
  - 학습 분류의 (family, rate, combination) 집합은 `dataset/labels_train.csv` 279행과 같다.
  - 3× 1회차 분류는 `dataset/labels_test.csv` 93행과 같다.
  - 반복 분류는 `labels_test.csv`의 `n_reps == 3` 후보 60개와 같다.

## Assert 결과 (모두 통과)

- `run_plan.csv` 492행, `dma32_log.csv` 492행이다. 모두 attempt 1이고 status ok다.
- block별 hours가 INTEGRITY §1 값과 ±1% 안에서 맞는다.
  - rep0: 4.1043 (INTEGRITY 4.104)
  - rep1: 0.6759 (0.676)
  - rep2: 0.6756 (0.676)
  - 합: 5.4558 (5.46)
- 분류별 run 수가 기대와 같다: 279 / 93 / 120, 합 492.
- label 파일 대조와 monotonic 시각 증가는 위에 적은 대로 통과했다.

## 불변 확인

- 집계 스크립트는 저장소 밖 세션 임시 폴더에 두었다(`scratchpad/prof_cost.py`).
- 저장소에 새로 생긴 파일은 `revision_exp/out_npu/profiling_cost_npu.csv`와 `profiling_cost_npu.md` 2개뿐이다.
- `git diff HEAD`: tracked 파일 변경 0이다. `out_npu/`는 원래 untracked이며, 이번에 새로 생긴 파일은 위 2개뿐이다.
- `../mms_npu_results`: `git status` 변경 0이고 HEAD는 `4070ce2e` 그대로다.
