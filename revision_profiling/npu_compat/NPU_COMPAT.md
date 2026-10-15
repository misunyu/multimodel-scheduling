# CPU–NPU 저장 데이터의 새 정의 호환성 (사양 v2.2 §6)

조사일 2026-10-13. 읽기 전용. 읽은 것: `results_antara/*.json`(44 파일), `git show new_main:{unified_viewer.py, view_handlers.py, schedule_executor_main.py}`(commit `6d9b13f`, 2026-01-05). 실행한 것 없음. 재학습 없음.

표기: [확인] 코드·데이터로 직접 확인, [추정] 근거는 있으나 직접 확인 못 함, [확인 불가].

## 1. 판정

**재계산 불가.** Counter 신뢰성 검증 네 항목 중 (i)과 (ii)가 통과하지 못했다. 사양서 규칙에 따라 근사 D_macro와 best-measured는 계산하지 않았다.

| 항목 | 결과 | 근거 |
|---|---|---|
| (i) `inference_count` 집계 구간 = `window_sec` 30 s | **불일치** [확인] | 아래 2.1 |
| (ii) `dropped_frames_due_to_full_queue`가 모든 instance에서 집계 | **아님** [확인] | 아래 2.2 |
| (iii) headless instance 없음 | 통과 [확인] | 아래 2.3 |
| (iv) throughput 필드 = `1000/latency` | 맞음. 예외 5개는 완료 0건 [확인] | 아래 2.4 |

## 2. 근거

### 2.1 집계 구간 (i)
- `window_sec`은 실행 지속 시간 인자를 그대로 저장한 값이다(`new_main:unified_viewer.py:653`, 저장 `:942`). 종료는 `start_execution` 시점부터 `QTimer.singleShot(duration * 1000, …)`로 잡는다(`:673`). 즉 30 s는 **process 생성·모델 load를 포함한 전체 실행 시간**이다.
- Warm-up 뒤 통계를 지우는 함수는 정의만 있고 호출이 없다: `view_handlers.py:74` `reset_stats`, `:238` `reset_counters`. 두 파일 전체에서 호출 지점이 0개다 [확인: grep].
- 따라서 `inference_count`는 실행 시작부터 종료까지의 누적이고, 그 안에서 실제로 추론이 돈 시간은 30 s보다 짧다(load 시간만큼). `Σ inference_count / 30`은 계측 구간 throughput이 아니라 "load 포함 30 s 동안의 평균"이다. Load 시간은 JSON에 없다 [확인 불가: instance별 load 시간].
- 참고: 저장된 `throughput_fps` 대비 `inference_count/30`의 비는 중앙값 0.456, 범위 0.073–0.989다(완료 > 0인 1,015 instance).

### 2.2 Drop 집계 (ii)
- 저장되는 drop은 `video_feeder.drop_counts`에서만 읽는다(`unified_viewer.py:924-925`). 이 dict는 YOLO용 video feeder만 갖고 있다(`view_handlers.py:215`, 증가 `:280`).
- ResNet용 image feeder는 queue가 가득 차면 **세지 않고 넘어간다**: `view_handlers.py:382-383` `except queue.Full: pass`.
- 데이터도 같다 [확인]:

| 모델 종류 | 장치 | instance 수 | drop > 0 |
|---|---|---|---|
| ResNet | CPU / NPU0 / NPU1 | 260 / 112 / 112 | **0 / 0 / 0** |
| YOLO | CPU / NPU0 / NPU1 | 288 / 124 / 124 | 288 / 124 / 124 |

  전체 1,020 instance 중 ResNet 484개(47%)의 drop이 구조적으로 0이다. CPU–GPU의 headless 고정 0(`unified_viewer.py:1708`)과 같은 종류의 결함이 다른 위치에 있다.
- YOLO의 drop 값도 "요청 1건당 1회"가 아니다. Feeder는 enqueue에 성공했을 때만 `last_enqueue_ts`를 갱신하므로(`:277`), queue가 가득 찬 동안에는 polling loop가 돌 때마다(약 `1/(2·source_fps)` 간격, `:261`, `:285`) drop을 1씩 더한다(`:276-280`). 즉 값은 **요청 수가 아니라 재시도 횟수**다. `generated ≈ completed + dropped` 근사의 전제가 성립하지 않는다 [확인: 코드].

### 2.3 Headless (iii)
- `new_main`의 저장 경로는 view1–view4만 다룬다(`unified_viewer.py:898-931`). JSON의 1,020 instance가 모두 view1–view4에 있다(372 / 324 / 240 / 84). Headless instance는 없다 [확인].

### 2.4 Throughput 필드 (iv)
- `avg_fps = 1000 / avg_infer_time`(`view_handlers.py:96`), 저장 `unified_viewer.py:958`. 1,015 instance에서 오차 1% 안으로 일치한다.
- 나머지 5개는 `inference_count = 0`, `throughput_fps = 0`, `avg_inference_time_ms = 0`인 instance다. 전부 NPU 배치(NPU0 1개, NPU1 4개; YOLO 4개, ResNet 1개)이고 30 s 동안 결과를 하나도 내지 못했다. 원인(load 지연인지 장치 오류인지)은 JSON으로 알 수 없다 [확인 불가].

## 3. 데이터 수 관계 (원고 수정용)

| 원고 표현 | 실제 | 집계 단위 |
|---|---|---|
| "279 placements" | 93 × 3 = 279 | **학습 표본 수**(1·2·4× rate의 측정 행). Placement는 rate당 93개(11 family 합) |
| "1,116 samples" | 데이터에 없는 수(279 × 4) | 실제 측정 행은 **372 = 93 placements × 4 rates**, 학습 279 + 평가(3×) 93 |

근거: `revision_audit/REPORT.md` §B-2, `revision_audit/tables/npu_family_inventory.csv`. 이번에 다시 센 값: 44 파일, 372 rows, `window_sec` 372개 전부 30.0 [확인].

원고에 쓸 정확한 문장: "11 workload families, 93 feasible placements per rate, 4 rates → 372 measured samples (279 for training at 1×/2×/4×, 93 for evaluation at 3×)".

## 4. 선택지 (결정은 사용자)

| # | 선택지 | 비용 | 한계 |
|---|---|---|---|
| 1 | NPU 보드에서 재측정 | 372 runs × 약 31 s ≈ 3.2 h + 새 runtime(`revision_profiling/code/`)의 NPU 이식. NPU worker(`npu.py` driver 호출)와 "NPU당 instance 1개" 제약을 새 executor에 넣어야 한다 | 보드 접근 필요. 이식 공수는 이번에 산정하지 않았다 |
| 2 | 근사 재계산 | — | **이번 검증으로 근거가 없어졌다.** ResNet drop이 없고, YOLO drop은 요청 수가 아니며, 집계 구간에 load가 섞여 있다. 한계를 명시해도 D_macro를 만들 수 없다 |
| 3 | NPU 평가 범위 축소 | 원고 수정만 | 기존 NPU 수치(α 혼용, 4 workloads, artifact 불일치)는 기존 정의로만 유효하다는 점을 본문에 적어야 한다 |

조사한 쪽의 의견: 2는 선택지에서 빼는 것이 맞다. 1과 3 중의 선택은 NPU 보드 접근 가능 여부에 달려 있다.
