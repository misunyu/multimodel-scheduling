# CHANGES — revision_profiling

Branch `revision-profiling` (from `ubuntu_gpu` @ `1f7e6bc`). 원본 tracked 파일은 하나도 수정하지 않았다. 아래는 전부 `revision_profiling/` 안의 새 파일이다. Commit은 하지 않았다(작업 트리의 untracked 폴더).

## 1. 원본 runtime 대비 바뀐 점 (파일·위치·이유)

원본 runtime(`model_processors.py`, `unified_viewer.py`, `view_handlers.py`)은 PyQt GUI와 묶여 있고 모델 경로가 함수 안에 고정되어 있어(`model_processors.py:245, 818`), 사본을 고치는 대신 headless executor를 새로 썼다. 원본 모듈은 import하지 않는다. `executor.py`가 시작·load 후·종료 시 세 번 `sys.modules`를 검사해, 저장소 안에서 온 모듈이 `revision_profiling/code/` 밖에 있으면 중단한다.

| 원본 동작 | 원본 위치 | 새 동작 | 새 위치 | 이유 |
|---|---|---|---|---|
| 모델 이름과 무관하게 `resnet50_big.onnx` / `yolov3_*.onnx` load | `model_processors.py:92, 245, 674, 818` | 이름 → ONNX 1개. 없는 이름은 중단. Load마다 sha256 검사 | `code/model_registry.yaml`, `code/executor.py: resolve_model` | 사양 §2.1 |
| Feeder가 `time.sleep(0.005)` loop에서 `now − last ≥ interval`이면 enqueue. 성공 때만 `last` 갱신 | `view_handlers.py`(feeder), `unified_viewer.py`(headless feeder) | 절대 시각 격자 `start + k/r`. 늦은 tick은 skip(`missed_ticks`), catch-up 없음 | `code/feeder.py: feeder_loop` | 사양 §2.3 |
| JPEG decode·file I/O가 feeder 안 | `view_handlers.py` | 실행 전 메모리에 decode(이미지 128장, 동영상 64 frame). Feeder는 index만 넣음 | `code/executor.py: load_inputs` | 사양 §2.2 |
| View instance throughput = `1000 / 평균 latency` | `view_handlers.py:96` | `completed / window`. completed는 출력 queue put 성공 직후 한 번 | `code/worker_base.py: Worker.run`, `code/counters.py` | 사양 §2.4 |
| Headless drop이 0으로 고정 | `unified_viewer.py:1708` | 모든 instance에서 `dropped_full` 집계 | `code/feeder.py`, `code/shmqueue.py: put_nowait` | 사양 §2.4 |
| `window_sec`이 None으로 저장, warm-up 뒤 counter reset | `unified_viewer.py:1107, 1132, 509` | Reset 없음. 이벤트마다 monotonic timestamp, 실행 뒤 `t0 ≤ ts < t1`로 집계. `t1 − t0`은 정확히 10 s | `code/counters.py: window_stats`, `code/executor.py` | 사양 §2.4, §2.6 |
| 분류 전처리: letterbox 224 + `/255`(mean/std 없음) | `image_processing.py:37-58, 83-93` | 짧은 변 256 resize → 가운데 224 crop → RGB → `/255` → ImageNet mean/std. Caffe2 계열 2개(squeezenet1.0-12, vgg19)는 BGR 0–255 − channel mean | `code/worker_cls.py: cls_preprocess` | 원본 전처리로 시험한 한 건에서 mnasnet이 class 0 이미지를 class 103으로 냈다(원본 전처리의 top-1은 따로 재지 않았다). 교체 후 128장 top-1: 83–119 / 128 (`logs/models_sanity.json`). **사양에 없던 변경** |
| YOLO 후처리: yolov3 계열 decode | `image_processing.py:170-393` | YOLOv4(tf2onnx, NHWC 416) anchor decode + class별 NMS | `code/worker_yolo.py` | 실제 `yolov4.onnx`의 출력 형식에 맞춤 |
| LM: 이미지 분류로 실행됨 | `model_processors.py:245, 818` | 고정 prompt 32 tokens + greedy 16 steps, KV cache 없음 | `code/lm_runner.py` | 사양 §3 |
| GUI·display thread가 결과를 소비 | `view_handlers.py` | Drainer thread 1개가 모든 출력 queue를 비움. 실패는 `consumer_failed` | `code/executor.py: Drainer` | 사양 §2.4 |

유지한 것: 한 process 안에서 instance마다 worker thread 1개, ONNX Runtime CPU EP / CUDA EP(CPU fallback 금지), CPU session의 `intra_op_num_threads = os.cpu_count()`(`model_processors.py:240`), 입력 queue 용량 2.

## 2. Feeder 구현 이력 (사양 §2.3 "실현율 미달 시 process 분리")

정식 pilot 전에 가장 무거운 설정(N=8, β=1.0, 4×, 배치 3개)으로 개발 점검을 했다. 각 시도의 결과 JSON은 `pilot/arrival/attempt*_dev/`에 있다. 이 run들은 pilot 집계에 넣지 않는다.

| 시도 | 구현 | 최소 실현율 (전부-GPU / 전부-CPU / 반반) | 판단 |
|---|---|---|---|
| 1 | executor process 안의 feeder thread, `queue.Queue` | 0.71 / 0.35 / 0.49 | 기준(0.95) 미달. CPU instance 하나가 24개 spinning thread를 쓰고 GIL도 공유한다 |
| 2 | instance별 feeder **process** + lock 있는 shared-memory queue | 0.95 / 0.19 / 0.94 | 개선. 전부-CPU 미달 |
| 3 | 2 + 추론 thread를 nice 10으로 | 0.95 / 0.86 / 0.99 | 개선. 경계 읽기가 12 ms 늦음(lock을 쥔 worker thread가 preempt됨) |
| 4 | 3 + **lock 없는** SPSC queue | 0.99 / 0.90 / 0.99 | 채택. 경계 읽기는 여전히 최대 14 ms 지연(main thread가 GIL을 기다림) |
| 5 | 4 + 경계 읽기를 별도 **reader process**로 | 0.98 / 0.86 / — | 채택. 읽기 지연 0.2 ms. 실현율은 실행마다 0.86–0.99로 흔들림 |
| 6 | 5 + 추론 thread를 `SCHED_IDLE`로 | 0.98 / 0.95 / 0.97 | **기각.** 전부-GPU throughput이 약 410 → 153 req/s로 떨어진다(worker 자신이 굶는다) |

최종: 시도 5의 구성(`INFER_NICE = 10`, `INFER_SCHED_IDLE = False`).

- `code/shmqueue.py` — 단일 producer·단일 consumer, 64-bit word 세 개(TAIL, HEAD, FIN). queue 길이 = TAIL − HEAD, in-flight = HEAD − FIN, pending = TAIL − FIN. Lock이 없다.
- `code/feeder.py: feeder_process_main` — spawn한 process. 모델 load 전에 띄우고 `start_ns`가 공유 메모리에 써질 때까지 대기. 종료 시 이벤트 buffer, 자기 `sys.modules` 경로, clock 종류를 pipe로 보고한다.
- `code/feeder.py: boundary_reader_main` — `t0`, `t1`에 모든 queue의 상태를 읽는 process. 읽기만 하고 아무도 기다리지 않는다.
- `code/worker_base.py: lower_inference_priority` — worker thread와 session을 만드는 loader thread가 자기 nice를 10으로 올린다. ORT thread pool은 이를 상속한다. Feeder·reader·main은 nice 0이다. **사양에 없던 변경.** 추론 thread끼리의 상대 우선순위는 그대로다.

## 3. 새 파일 목록

| 파일 | 역할 |
|---|---|
| `SPEC_v2.2.md` | 사양서 사본(대화 기록에서 그대로 추출) |
| `code/model_registry.yaml` | 9개 모델의 경로·sha256·전처리·후처리·요청 단위 |
| `code/executor.py` | 한 배치를 실행하고 JSON + `*.events.npz`(이벤트 timestamp 원본)를 쓴다 |
| `code/counters.py` | 이벤트 buffer, 계측 구간 집계, conservation, invalid 판정, system 지표(F, R, D_macro, D_max) |
| `code/feeder.py`, `code/shmqueue.py` | 위 2절 |
| `code/worker_base.py`, `worker_cls.py`, `worker_yolo.py`, `lm_runner.py` | Worker |
| `code/image_processing.py` | 원본의 **수정 없는 사본**. `yolo_preprocess_local`만 쓴다 |
| `code/imagenet_classes.txt` | 원본 사본(위 모듈이 읽음) |
| `code/lm_assets/prompts.json` | 고정 prompt의 token id(GPT-2 32개, TinyLLaMA BOS 포함 32개) |
| `code/lm_assets/gpt2/{tokenizer.json, config.json}` | huggingface.co `openai-community/gpt2`에서 내려받음(2026-10-13) |
| `code/offline_profile.py` | 모델 × device의 isolated capacity(closed loop) |
| `code/gen_schedules.py` | `rate = β × GPU isolated throughput × 배수`. Family는 기존 `gen_schedules/*.yaml`에서 모델 목록만 읽는다 |
| `code/run_campaign.py` | Run마다 subprocess. `run_key`로 재개, 불완전·invalid 결과는 `*.rejected.*`로 남기고 재실행 |
| `code/run_phase1_pilot.sh` | Pilot 1–4단계 순서 |
| `code/analyze_smoke.py`, `code/analyze_pilot.py` | §5.1–5.5 판정표 |
| `code/family_reconstruct.py` | §3.4 family 재구성과 run 수 |
| `code/check_models_sanity.py` | 분류 top-1, YOLO 검출 수(통과 기준 아님) |
| `vendor/` | `pip install --target`으로 받은 `tokenizers`(prompt 만들 때와 출력 decode에만 사용. runtime은 import하지 않는다). `.venv`는 건드리지 않았다 |

## 4. 아직 없는 것

- `code/recompute_scores.py`: Phase 2에서 dataset CSV를 만들 때 필요하다. Phase 1의 세 정의 계산은 `code/analyze_pilot.py: dropmetric`에 있다.
- `schedules/`: Phase 2용 YAML은 LM·β 승인 뒤에 만든다. Pilot용 YAML은 각 `pilot/*/schedules/`에 있다.

---

# Phase 1b (SPEC_PHASE1B_v1.1.md, 2026-10-13)

Phase 1 결과는 `pilot_phase1/`, `offline_profile_phase1/`, `phase1_reports/`로 옮겨 보존했다. 새 결과는 `pilot/`, `offline_profile/`.

| 파일 | 위치 | 변경 | 이유 |
|---|---|---|---|
| `code/model_registry.yaml` | 끝 | `tiny-llama-chat-onnx` 항목 삭제(모델 파일은 그대로) | 사용자 결정 2 |
| `code/worker_base.py` | `make_session` | 모든 session(CPU·GPU)에 `intra_op_num_threads = 4`, `session.intra_op.allow_spinning = "0"` | 사용자 결정 1 |
| `code/worker_base.py` | `INFER_NICE`, `lower_inference_priority` | 환경변수 `REVPROF_NICE`로 설정, 기본 0. 0이면 `os.nice`를 부르지 않는다 | §2 비교 결과 nice 제거 |
| `code/feeder.py` | `feeder_loop` | 요청 ID(= tick index)를 queue에 넣고 `gen_id` buffer에 기록 | §3-3 요청 추적 |
| `code/shmqueue.py` | `put_nowait`, `leftover_ids` | slot에 요청 ID 저장, 종료 시 남은 ID 조회 | 같음 |
| `code/worker_base.py` | `Worker.run` | `deq_id`, `done_id`, `fail_id` 기록 | 같음 |
| `code/counters.py` | `window_stats`, `invalid_reasons` | pending을 요청별 timestamp로 재구성. Conservation은 정확히 0만 통과. `trace_check`, `snapshot_diff_*`, `snapshot_flag` 추가. "±1 허용" 삭제 | §3 |
| `code/executor.py` | 집계부 | 위 인자 전달, schema `v2.2+1b`, `env`에 thread·spinning·nice 기록 | |
| `code/gen_schedules.py` | `reconstructed_families` | TinyLLaMA 제외 후 재구성한 22 family. Family key는 모델 이름을 `+`로 이은 것 | §1.4 |
| `code/check_cuda_cpu_nodes.py` | 새 파일 | CUDA session의 CPU EP node 수·시간(ORT profiling 1회) | 사용자 결정 1 |
| `code/analyze_pilot1b.py`, `code/run_phase1b_pilot.sh` | 새 파일 | Phase 1b 판정표와 실행 순서 | |
| `code/analyze_smoke.py` | conservation 항목 | `trace_check == ok` 조건 추가 | §3 |

실행 조건(최종): thread 4, spinning off, nice 없음, feeder process + reader process, lock 없는 queue.

Offline profile은 두 번 쟀다. 처음 것은 nice 10 상태였으므로 `offline_profile/nice10_initial/`에 두고, nice 결정 뒤 nice 0으로 다시 잰 값을 쓴다.

## Phase 2 (v1.2) — 2026-10-13

실행 조건과 runtime(`executor.py`, `feeder.py`, `shmqueue.py`, `worker_*.py`, `counters.py`, `model_registry.yaml`)은 Phase 1b 최종에서 바꾸지 않았다.

| 파일 | 변경 | 이유 |
|---|---|---|
| `code/gen_schedules.py` | `--masks`에 `randK` 추가(혼합 배치 `1 … 2^N−2`에서 `SeedSequence(42)`로 K개 비복원 추출) | §1 후보 8개 |
| `code/analyze_repeat54.py` | 신규 | §1의 8개 항목, 판정 규칙 |
| `code/recompute_scores.py` | 신규 | §3.1 label·dataset CSV. 3회 평균 뒤 정규화. Dataset용 schedule CSV는 기존 parser layout으로 출력 |
| `code/check_dataset_compat.py` | 신규 | §2.1b·§3.1 검증. `revision_exp/data.py`를 수정 없이 import하고 `D.DS`, `D.SETS`만 실행 중 지정 |
| `code/phase2_plan.py` | 신규 | §2.1 schedule 88개, §2.2 run 수 검증, §2.3 실행 순서 |
| `code/run_phase2.py` | 신규 | §2.3 campaign runner. 재개, 재시도 최대 2회, `exec_order.csv`, 10분 progress, 1분 GPU 요약 |

지시문과 다르게 한 것:

- **`tmux`가 이 PC에 없다.** `setsid nohup`으로 띄웠다(PID는 `logs/campaign.pid`, 출력 `logs/campaign_stdout.log`). 터미널·세션이 끊겨도 계속 돈다.
- **family_index는 1부터 센다.** Family 순서 seed가 `[42, 0]`이므로, family 안 순서 seed `[42, family_index]`와 겹치지 않게 했다. 반복 블록 r의 seed는 `[42, 0, r]`, `[42, family_index, r]`이다.
- **결과 경로** `results/<family>/x<m>/run_<combination>_rep<r>.json`. v1 지시문을 받지 못해 "v1 그대로"라는 파일명 규칙을 확인할 수 없었고, `results/**/run_*.json`에 맞는 형태로 정했다.

### Phase 2 추가분 — 2026-10-14

| 파일 | 변경 | 이유 |
|---|---|---|
| `code/integrity.py` | 신규 | §2.4 `results/INTEGRITY.md`와 `results/integrity/*.csv` |
| `../revision_exp_adapter/run_reprofiled.py` | 신규 | §4. `revision_exp/`의 모듈을 수정 없이 import하고 `D.DS`, `D.SETS`, `D.OUT`만 실행 중 지정. 출력은 `revision_exp/out_reprofiled/`(D_macro), `revision_exp/out_reprofiled_agg/`(D_agg 참고) |
| `dataset/README.md`, `dataset/SPLIT_CHECK.md` | 신규 | §3 |

- 세 스크립트 모두 campaign 종료 뒤 첫 실행에서 오류 없이 끝났고, 실행 뒤 코드를 고치지 않았다.
- **generated 차이의 판정 규칙**: scheduled tick이 200 이상인 instance는 상대 차 2% 초과, 200 미만은 절대 차 2건 초과를 위반으로 센다(Phase 1b와 같은 규칙, campaign 전에 정함). 지시문의 "`gen_mismatch_max > 2%`"와 다른 점은 200 tick 미만 instance(GPT-2 1×·2×·3×)에 절대 기준을 쓴다는 것이다. 순수 상대값은 `results/integrity/gen_mismatch.csv`의 `gen_mismatch_rel_all`에 있다.
- Campaign 중단·재개: 없음(`logs/campaign_sessions.log`에 START 1회, END 1회).

## Phase 3 (v1.2) — 2026-10-14

새 코드는 `revision_exp/phase3/`, 산출물은 `revision_exp/out_phase3/`. `revision_exp/`의 기존 모듈, Phase 2 산출물, dataset은 바꾸지 않았다(모듈은 import만 하고 `D.DS`, `D.SETS`, `D.OUT`을 실행 중 지정).

| 파일 | 내용 |
|---|---|
| `revision_exp/phase3/common.py` | 경로·loader·Phase 2 fold mapping·3× label guard |
| `revision_exp/phase3/precheck.py` | §0-pre P1–P5 |
| `revision_exp/phase3/predictors.py` | Ridge / RF / MLP (사전 고정 grid), XGBoost −P / −I 재학습 |
| `revision_exp/phase3/baselines/{instance_rows,fit_instance,compose_select}.py` | Per-instance baseline ([14]-inspired) |
| `revision_exp/phase3/run_phase3.py` | 학습 → 설정 고정 → per-instance 선택(label 없이) → 최종 평가·표 |

- 스크립트는 자기 폴더 대신 `revision_exp/`를 `sys.path[0]`에 둔다. `phase3/baselines/` 패키지가 기존 `revision_exp/baselines.py`를 가리지 않게 하려는 것이다(`run_phase3.py`에서 assert).
- Ablation mask 확정: −P `^pairhash_\d+_on_`, −I `^views\.(sum|mean|max)\.model_hash_\d+_on_` + `^views\.(sum|mean|max)\.view\.infps_on_`. 초안과 같은 열을 잡는다.
- 첫 실행 결과를 그대로 썼고, 실행 뒤 코드·설정을 고치지 않았다.

### Phase 3 후속 — predicted score clipping (2026-10-14)

| 파일 | 내용 |
|---|---|
| `revision_exp/phase3/clip_check.py` | 저장된 예측값으로 clip 있음/없음 비교(재학습 없음) |
| `revision_exp/phase3/reaggregate.py` | 저장된 예측값으로 Phase 3 표 재집계. `--clip`으로 기존 표 재현 확인 후, clip 없는 Ŝ로 표를 다시 씀(이전 파일은 `out_phase3/clip_check/before/`) |

- 선택 점수를 모든 direct 방법에서 `T̂ − 0.3·D̂`(clip 없음)로 통일했다. `revision_exp/predict.py`는 고치지 않았고, Phase 3 재집계에서 clip을 적용하지 않는 방식이다. 상세는 `revision_exp/out_phase3/CLIP_CHECK.md`.
- `run_phase3.py`는 그대로이므로 다시 실행하면 clip 있는 표가 나온다. 최종 표는 `reaggregate.py`의 출력이다.

### Phase 3 후속 — Fig. 4 Single clip 통일, 재현 경로 정리 (2026-10-14)

| 파일 | 내용 |
|---|---|
| `revision_exp/phase3/fig4_noclip.py` | 신규. Phase 2 Single artifact로 예측만 다시 하고(재학습 없음) clip 있음/없음을 비교. clip 있음이 Phase 2 Fig. 4 Single을 재현함을 먼저 확인 |
| `revision_exp/phase3/run_phase3.py` | `CLIP_S_HAT = False`(최종 Ŝ clip 없음), 각주 문구 `SCORE_RULE`, `summary.json`에 `selection_score` 추가 |

- **원고용 Fig. 4는 `revision_exp/out_phase3/fig4_noclip/fig4_alpha_sensitivity.pdf`(.png)이다.** `revision_exp/out_reprofiled/figs/fig4_alpha_comparison.*`(Phase 2, Single에 clip 적용)는 쓰지 않는다. 데이터는 `fig4_noclip/fig4_data.csv`.
- 위 "Phase 3 후속 — predicted score clipping"의 "`run_phase3.py`는 그대로" 문장은 이 변경으로 더 이상 맞지 않는다. 이제 `run_phase3.py`를 다시 실행하면 clip 없는 최종 표가 나오고, `reaggregate.py`의 출력과 표 7개가 일치한다(`out_phase3/CLIP_CHECK.md` §7).
- `revision_exp/predict.py`와 `revision_exp/out_reprofiled/`는 바꾸지 않았다.
