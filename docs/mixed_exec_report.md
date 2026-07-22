# Mixed 셋 실행 — LLM/VLM 격리 background + 배치 거친 전환 (Q3 실행 기반)

날짜: 2026-07-22 · 범위: GPU (사용자 결정: 구현 1번=GPU만, background=llama1b만) · background 기본 OFF

## 요약
vision foreground는 그대로 두고, generative(llm/vlm) 콤보 엔트리를 **격리 프로세스**로 실제 실행해
가속기 경합을 만든다. LLM 배치(gpu↔cpu)는 BoundGuard 전환 시 함께 바뀌는 **거친 전환**(구 프로세스
종료 → 신 프로세스 기동) 대상. **V(t)·hot-swap·QoS·vision 디스패치는 불변**, background는 QoS 미측정.

## 설계 (§0 개정 반영)
- **배치 전환 포함, 거친 전환**: `BackgroundManager.sync(entries)`가 콤보의 generative 배치와 현재
  실행 상태를 대조 — device가 바뀐/사라진 모델은 종료, 새로/이동한 모델은 기동. ready-파일로 로드 완료
  게이팅(겹치기 전환·IPC·ready_event 불필요, background라 잠깐 멈춰도 무방).
- **격리 = 별도 subprocess** (`python -m runtime.bg_entry`): torch/transformers가 그 프로세스에만
  import → vision onnxruntime-gpu 워커와 cuDNN 독립. multiprocessing이 아닌 subprocess라 **Qt/vision
  main 모듈 재import 없음**.
- **toggle 기본 OFF** (`--background` 플래그 또는 `FSRR_BACKGROUND` env). OFF면 완전 no-op.

## 격리 방식 (§2)
- **GPU**: 별도 프로세스로 `LLMEngine`(transformers/torch). 검증됨.
- **NPU**: 이 박스에서 **실행 가능 확인** — `qbruntime`(정정: `mobilint_qb_runtime` 아님) 설치, NPU
  device[0] 가용, mobilint W8 `.mxq`를 aries0에 5.8s 적재 + 생성 성공. **DRAM 조회 공개 API 없음 →
  경험적 판정만.** 사용자 결정에 따라 NPU background 구현은 후속(Q5), 코드는 GPU만.

## 구현 (§5)
| 파일 | 역할 |
|---|---|
| `runtime/llm_engine.py` | mobilint에서 이식(device-agnostic gpu/cpu/npu, spec 키 전부 존재). |
| `runtime/bg_entry.py` | 격리 background 1개: 로드→ready 파일→SIGTERM까지 생성 루프→dispose. |
| `runtime/background_llm.py` | `BackgroundManager`: sync(거친 전환)·shutdown·좀비 방지. |
| `schedule_executor_main.py` | `--background`/`--background-max-new-tokens` + env, `_generative_entries`, `_run_next`서 `bg.sync`, `stop()`/종료 시 `bg.shutdown`. |
- 스냅샷: `backup/mixed_exec_20260722_133824/schedule_executor_main.py`.
- **vision 워커·디스패치·QoS·BoundGuard 불변.** `bg.sync`는 콤보 전환 지점에 추가된 별도 경로(try/except 가드).

## 검증 (§6) — 통합 파이프라인 headless 실행
2-콤보 mixed 스케줄(`yolo11s`@GPU + `llama1b`@GPU → @CPU), `schedule_executor_main --background`:

| 게이트 | 결과 |
|---|---|
| **(1) vision 무회귀 [최우선]** | `yolo11s on GPU`, view2(llama1b) **UnifiedViewer가 skip**(generative), cuDNN 폴백/에러 0. vision 220–268fps @ 3.7–4.6ms (GPU 실가속). ✅ |
| (2) BoundGuard 불변 | vision 디스패치/QoS 코드 무변경, bg는 additive. ✅ |
| (3) 기존 결과 무회귀 | **OFF**(기본): bg_entry 0개, bg-llm 로그 0줄, vision 268fps 전속. ✅ |
| (4) 간섭 유효 | LLM=GPU 경합 시 vision **98fps/10.24ms** vs OFF 268fps / LLM=CPU 220fps. ✅ |
| (5) 생명주기 | 전환·종료 시 자식 정상 종료(SIGTERM→dispose), 좀비 0, GPU mem 3893→900MiB 해제 확인. ✅ |
| (6) legacy/scripts/backup | `git diff`: 수정=`schedule_executor_main.py`만, 신규=`runtime/{llm_engine,bg_entry,background_llm}.py`. **무변경 확인.** ✅ |

## §7. 대안 도달 가능성 (Q3로 가는 다리) — **도달 가능 확정**
통합 파이프라인에서 콤보 `combo_llm_gpu`(LLM=GPU) → `combo_llm_cpu`(LLM=CPU) 전환:

| 콤보 | LLM 배치 | vision yolo11s fps | infer ms |
|---|---|---|---|
| combo_llm_gpu | GPU (경합) | **98** | 10.24 |
| combo_llm_cpu | CPU (대안) | **220** | 4.55 |

- **거친 전환(GPU→CPU)이 실제로 일어나고**(로그: stop llama1b/gpu → start llama1b/cpu, GPU mem 해제),
  그 결과 **vision이 98→220fps로 회복**(2.2x). → BoundGuard 후보에 LLM-CPU 후보를 넣으면 **검증 실패 후
  전환으로 vision QoS 회복 가능** = Q3 실험의 다리 확보.

## 사람 판단 필요 항목
1. **Q3 실험 착수**: 네 기법 비교(Adaptive=top-1 LLM-GPU에 커밋해 실패, BoundGuard=검증 실패 후 LLM-CPU 후보로
   전환해 회복) — `q3_q5_misprediction_instructions.md` 경로. 후보 열거 로직이 LLM 배치가 다른 후보를
   생성하는지 확인 필요(현 예측기는 top-1로 LLM-GPU만 반환 → 후보 시퀀스에 LLM-CPU를 포함시키는 방식 결정).
2. **Q5(NPU)**: NPU LLM 실행 가능 확인됨 → NPU background 구현 + DRAM 예산(경험적) 착수 여부.
3. **qwen2_vl**: 필요 시 non-cuDNN 경로로 background 추가(현재 llama1b만).

## 무결
- GPU 폴백 0(종료 후 GPU 897MiB idle). vision/QoS/BoundGuard/β/예측기·legacy/backup 불변. main.tex 불변.
- 재현: `--background` 켜면 콤보의 generative 엔트리대로 격리 실행, 끄면(기본) 완전 no-op.
