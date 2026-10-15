# Profiling cost (campaign logs)

출처: `revision_profiling/logs/exec_order.csv`(run별 `start_epoch`, `end_epoch`, `seconds`), `logs/campaign_sessions.log`. 코드 `code/profiling_cost.py`, 원자료 `profiling_cost.json`, `profiling_cost_parts.csv`, `profiling_cost_byN.csv`.
Run wall time은 executor subprocess 하나의 시작–종료 시간이다. process 생성, 모델 load, warm-up 5 s, 측정 10 s, 결과 저장이 모두 들어간다 [확인: `run_phase2.py:123-132`].

## 결과

| 구분 | runs | 합계 (h) | run 중앙값 (s) |
|---|---|---|---|
| 학습용 1·2·4× (rep0) | 1,512 | 7.754 | 18.53 |
| 평가용 3× rep0 | 504 | 2.588 | 18.55 |
| 평가용 3× rep1 + rep2 (반복) | 960 | 4.955 | 18.62 |
| **합계** | **2,976** | **15.297** | 18.56 |

- 네 값의 합 15.297 h는 campaign 전체 경과 시간(첫 run 시작 → 마지막 run 종료) 15.297 h와 같다. run 사이 간격은 합쳐 0.3초다. `campaign_sessions.log`의 END 줄도 15.30 h이고, 중단·재시도는 없었다.

| N | runs | 중앙값 (s) | p90 (s) |
|---|---|---|---|
| 1 | 64 | 16.15 | 16.83 |
| 2 | 32 | 17.52 | 17.95 |
| 3 | 288 | 17.64 | 18.47 |
| 4 | 96 | 17.73 | 18.39 |
| 5 | 192 | 18.00 | 18.61 |
| 6 | 768 | 18.02 | 18.67 |
| 7 | 1,536 | 18.97 | 19.87 |

N=7 model set 1개(`gpt2+mnasnet+resnet50+resnext50+shufflenet+squeezenet+vgg19`):

- 4 rates × 128 배치 = 512 runs에 **2.764 h**가 걸렸다(반복 제외).
- 3× 반복 2회를 더하면 768 runs, **4.146 h**다.
- 다른 N=7 set(`…+vgg19+yolov4`)은 각각 2.662 h, 3.994 h다.

## 15.3 h에 들어가지 않는 것

이 시간들은 campaign 밖에서 따로 잰 것이다.

- **Isolated offline profile**(8 모델 × CPU/GPU, nice 0으로 다시 잰 최종본): `offline_profile/runs/`의 파일 시각으로 2026-10-13 16:51–16:55, 약 4분 [추정: 파일 수정 시각 기준. 첫 run의 시작 시각은 기록 없음].
- **Phase 1b pilot과 5.1 smoke**: `logs/phase1b_pilot.log` 16:55:21–17:38:17, 약 43분.
- **§5.4 반복 pilot**: `logs/repeat_5_4.log` "finished in 7.1 min".
- Phase 1 pilot과 nice 10으로 잰 첫 offline profile은 설정을 바꾸기 전의 측정이라 제외했다.

## 원고용 한 줄

"Profiling for training (1×, 2×, 4×; 1,512 runs) took 7.8 h and profiling for evaluation (3×; 504 runs plus two repeats of the 480 N≥3 placements) took 7.5 h, 15.3 h in total on one machine (median 18.6 s per run including model loading, 5 s warm-up and the 10 s window); profiling one seven-model set at four rates (512 runs) took 2.8 h."
