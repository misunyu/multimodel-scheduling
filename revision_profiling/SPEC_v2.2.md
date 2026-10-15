# CPU–GPU 재profiling 사양서 최종본 (v2.2) — Phase 1 (사양 확정·pilot)

대상 저장소: `/home/msyu/PycharmProjects/multimodel-scheduling`. 선행 문서: `revision_audit/REPORT*.md`, `revision_exp/REPORT_RETRAIN.md`, `HANDOFF_FOR_CLAUDE.md`, 사양서 v1·v2.
v2 대비 변경(v2.1): §5.1 CPU 단일 instance 기준 분리, §2.3 tick 정책, §2.4 in-flight·출력 queue 실패·worker별 모듈 검증, §4 `generated_i = 0` → invalid run, §5.5 용어·채택 근거, §3 LM 개별 처리·계측 구간 조건, §6 NPU counter 신뢰성 검증, §7 raw 복원 명시, §9 측정 순서와 중간 rate 제외 영향.
v2.1 대비 변경(v2.2, 최종): §2.4 barrier 제거 → monotonic clock 경계 기반 timestamp 집계, 출력 put 실패 = failed·consumer 실패 분리; §5.1 CPU 과부하/비과부하 구분 기준; §3.1 LM 계측 구간 조건을 완료 건수 기준으로; §2.3·§5.2 `scheduled_ticks`를 실제 계측 구간 기준으로, 요청 수에 따른 상대/절대 편차 기준; §4 동치 조건 명시·용어 수정; §5.3 β별 1× 단일 GPU 검증 대상 추가.
새 기능·실험을 추가로 확대하지 않는다.

**이 문서는 Phase 1만 지시한다. 전체 campaign(Phase 2)은 Phase 1 보고 후 사용자 승인 전까지 실행하지 않는다.**

---

## 0. 결정 사항 현황

| # | 항목 | 상태 | 내용 |
|---|---|---|---|
| D1 | LM | **Phase 1에서 시도** | GPT-2 먼저, TinyLLaMA 별도. 상한 2–3일. 결과에 따라 사용자 승인으로 포함/제외 결정 (§3) |
| D2 | Feeder | 확정 | instance별 timer thread + 사전 decode(JPEG/video decode까지만). 실현율 미달 시 process 분리 |
| D3 | 요청 rate | 확정 | placement 무관 `r_m = β × GPU isolated throughput_m`, 배수 1·2·3·4× |
| D4 | Δ | 확정 | warm-up 5 s + 계측 10 s, 실제 길이 기록 |
| D5 | 중간 rate | 확정 | Phase 2 1차 제외 |
| D6 | 반복 | 확정 | Phase 2에서 3× N≥3만 3회 |
| D7 | Family | **LM 결정 후 재계산** | 기존 29개 기준. LM 제외 시 조합 재구성 (§3.4) |
| D8 | Queue 용량 | 확정 | 2, drop-newest |
| D9 | 입력 | 확정 | 분류: imagenet sample 순환, 검출: 동영상 frame. decode만 사전 수행 |
| **D10** | **Drop metric** | **보류 (사용자 승인)** | aggregate `R/R_max` vs `D_macro` vs `D_max`. Phase 1에서 셋 다 계산해 보고 (§4) |
| D11 | β | **Phase 1에서 선택** | 후보 {0.25, 0.5, 1.0}, 2×·4×에서 물리 기준으로 (§5) |

---

## 1. 폴더·branch·코드 격리

```
revision_profiling/
  README.md  CHANGES.md          # 사양서 사본, 변경 기록(파일·line·이유)
  code/                          # 수정 runtime 사본. 원본 모듈 불변
    model_registry.yaml  feeder.py  counters.py  worker_*.py  executor.py
    lm_runner.py                 # D1
    gen_schedules.py  recompute_scores.py  run_campaign.py
  schedules/                     # Phase 2용 YAML (Phase 1에서는 pilot용만)
  offline_profile/
  pilot/                         # Phase 1 run 전부 (campaign에 섞지 않음)
    smoke/  lm/  arrival/  beta_0.25/  beta_0.5/  beta_1.0/  repeat/
  npu_compat/                    # §6 조사 결과
  logs/
  REPORT_PHASE1.md
```

- Git branch `revision-profiling` (from `ubuntu_gpu`). 원본 tracked 파일 수정 금지.
- **모듈 경로 검증**: executor 시작 시 `sys.modules`에서 runtime 모듈(`model_processors`, `unified_viewer`, `view_handlers`, 전처리)의 `__file__`이 `revision_profiling/code/` 아래인지 assert. 하나라도 원본 경로면 **실행 중단**하고 로그에 남긴다. 결과 JSON에도 로드된 모듈 경로를 기록.
- Python `-B`, `MPLCONFIGDIR` 임시 폴더.

---

## 2. Runtime·계측 수정 (v1 §2 유지 + 보완)

### 2.1 모델 registry
- `model_registry.yaml`: 이름 → onnx 경로, sha256, input_size, preprocess 종류, postprocess 종류, 요청 단위(image / lm_prompt). 매핑 없는 이름은 에러 중단. Load 시 경로·sha256을 결과 JSON에 기록.
- 기본 매핑: `models_onnx/{mnasnet, resnet50, resnext50, shufflenet-v2-12, squeezenet1.0-12, vgg19, yolov4}.onnx`. LM은 §3 결과에 따라 추가.

### 2.2 전처리 경계 (ChatGPT §6.1)
- **사전 수행(계측 밖)**: 파일 I/O, JPEG decode, 동영상 frame decode → uint8 HWC 배열을 메모리 list에.
- **계측 안(worker 경로)**: resize, normalize, layout 변환, tensor 생성, 추론, 후처리(argmax / YOLO decode). 원고의 "auxiliary stages on CPU"에 해당하므로 유지.
- Feeder는 배열 index만 queue에 넣는다.

### 2.3 Feeder
- instance별 thread, 절대 시각 스케줄(`next_t += 1/r_i`, drift 없음). 각 tick에서 `generated += 1`; `put_nowait` 성공 시 `enqueued += 1`, `queue.Full`이면 `dropped_full += 1`.
- **예정 tick과 실제 tick 구분**: `scheduled_ticks`는 **실제 계측 구간 `[t0, t1)` 안에 예정된 tick 수**(feeder의 격자 시각 중 `t0 ≤ 격자 < t1`인 것의 개수, run 종료 후 격자로부터 계산)이고, `generated`는 그 구간에 실제 실행된 tick이다. 둘을 따로 센다. Tick이 지연되어 다음 예정 시각을 지난 경우의 정책은 **skip**(밀린 tick은 생성하지 않고 `missed_ticks += 1`, `next_t`를 현재 시각 이후의 다음 격자로 재정렬)으로 고정한다. catch-up(burst 생성)은 쓰지 않는다 — 요청 rate를 부풀리지 않기 위해서다. `missed_ticks`, 최대 지연(ms), 지연 분포 p95를 instance별로 기록.
- 실현율 = `generated / scheduled_ticks`. 기준(§5.2) 미달 시 `multiprocessing` + shared memory로 분리. counter 정의 동일.

### 2.4 Counter 이벤트 정의 (ChatGPT §6.2, §6.3)
| counter | 증가 시점 |
|---|---|
| generated | feeder tick |
| enqueued | 입력 queue `put` 성공 |
| dropped_full | 입력 queue `Full` |
| completed | worker가 **후처리까지 끝내고 결과를 출력 queue에 `put` 성공한 시점** (display 유무와 무관, 한 곳에서만) |
| failed | 전처리·추론·후처리 예외(`continue` 경로) **및 출력 queue `put` 실패**. completed와 상호 배타 (put 실패는 completed 이전 실패로 센다) |
| consumer_failed | completed **이후** consumer 측(display handler, drainer) 예외. completed에서 빼지 않고 별도 기록. 0이어야 정상 |
| pending | 경계 시각의 입력 queue 길이 + in-flight(worker가 꺼냈으나 completed/failed 전). 출력 queue는 completed 이후이므로 **포함하지 않음** |

- **In-flight 상한 검증**: worker는 한 번에 요청 1개만 처리하므로 in-flight ∈ {0, 1}이어야 한다. 경계마다 assert하고, 위반하면 run invalid(worker 구조가 가정과 다르다는 뜻).
- **계측 경계와 집계 (barrier 없음)**: 모든 thread/process가 공유하는 **monotonic clock**(`time.monotonic_ns`, process 분리 시 `CLOCK_MONOTONIC` 공유 확인) 기준으로 main이 `t0`(warm-up 종료)와 `t1 = t0 + 10 s`를 정한다. 각 thread는 이벤트(generated, enqueued, dropped_full, completed, failed)마다 timestamp를 자신의 버퍼에 기록하고, **아무도 기다리지 않는다.** 집계는 run 종료 후 `t0 ≤ ts < t1`인 이벤트 수로 한다. 따라서 `t1`은 진행 중인 추론 완료를 기다리느라 밀리지 않으며, `window_sec_actual = t1 − t0`는 정확히 10 s다.
  - `pending_start`/`pending_end`: `t0`·`t1`에 main이 각 instance의 입력 queue 길이와 in-flight flag를 읽는다(원자적 읽기; 수백 μs 오차 허용, 오차 상한 기록). **`t1`에 진행 중인 요청은 pending_end에 포함**되고, 그 완료 이벤트는 `ts ≥ t1`이므로 window에 들어가지 않는다.
  - 경계 직후 완료 이벤트와 pending의 이중 집계가 없도록, 이벤트 ts는 "완료 처리 직후 한 번" 찍는다.
- Conservation(instance별, 계측 구간): `generated = enqueued + dropped_full`, `enqueued = completed + failed + (pending_end − pending_start)`. 정수 오차 0이어야 함. 경계 읽기 오차로 ±1이 나오면 그 사실과 횟수를 기록하고, ±1 초과는 invalid.
- **모듈 경로 검증 범위**: main process뿐 아니라 **worker·feeder가 별도 process로 실행되는 경우 각 process 안에서** `sys.modules`의 `__file__`을 검사해 결과에 기록한다. 하나라도 `revision_profiling/code/` 밖이면 run invalid.

### 2.5 결과 JSON (v1 §2.4 schema 유지 + 추가 필드)
- `instances[].window`에 위 counter 전부, latency p50/p95, `preprocess_ms_mean`.
- `system`: `completed`, `dropped_full`, `generated`, `window_sec_actual`, `throughput_fps = completed/window`, `drop_rate_fps = dropped_full/window`, **`drop_macro = mean_i dropped_full_i/generated_i`**, **`drop_max = max_i …`**, `service_rate_sum_legacy`(참고).
- `run_key`(schedule, combination, rate, repeat index — 고정) vs `execution_id`(timestamp — 실행마다). 재개 시 run_key로 존재 확인 + JSON 완전성(필수 필드) + conservation 통과 여부까지 확인하고, 불완전하면 재실행.
- `env`, `loaded_modules`, `git_commit`.

### 2.6 Δ·종료
- load → feeder 시작 → warm-up 5 s → `t0` 기록·pending_start 읽기 → 10 s(이벤트는 timestamp로 계속 기록) → `t1` 기록·pending_end 읽기 → feeder 정지 → drain → 저장. counter reset은 없고 timestamp 기준으로 집계한다. drain된 항목은 pending_end에 이미 반영. `t1` 이후 완료되는 in-flight 요청은 window에 들어가지 않는다(§2.4).
- GPU 부하 근거: 계측 구간 동안 `nvidia-smi --query-gpu=utilization.gpu,memory.used` 1 s 간격 sampling을 JSON에 포함.

### 2.7 Schedule 생성 (D3)
- `r_m = β × GPU_throughput_m` (offline_profile). YAML에 `base_rate`, `beta`, `multiplier`, `infps`(호환).
- Family는 기존 29개 조합을 읽어 §3.4 규칙으로 재구성. 후보 2^N 전부. Feasible 제외가 생기면(메모리 등) 이유 기록.

---

## 3. D1 — LM bounded attempt (상한 2–3일)

### 3.1 GPT-2
- 파일: `models_onnx/gpt2.onnx` (653.7 MB). 입출력 shape·dtype, past-KV 입력 유무를 `onnx` 로 읽어 기록.
- Tokenizer: 저장소 또는 `transformers` 로컬 캐시에 있는지 확인. 없으면 설치 가능 여부 보고(네트워크 정책).
- 요청 정의: 고정 prompt(토큰 수 P=32) + K=16 token greedy decode. KV cache 없이 매 step 전체 forward(길이 P+k). 요청 1건 = K step 완료.
- CPU·GPU 각각 단일 instance로 latency(요청당)·throughput 측정. `completed`는 K token 생성 완료 시점.
- 통과: 양쪽 device에서 예외 없이 실행, 요청당 latency가 안정(p95/p50 < 2), 출력 token이 비어 있지 않음, **그리고 계측 구간 조건**: 실제 requested rate(`r_m`, 1×)와 계측 구간 `completed` 건수를 함께 기록하고, `completed ≥ 20`이어야 정상 검증으로 판정한다. 20 미만이면 "계측 구간 부족(통계적 불안정)"으로 기록하고 그 device에 대해서는 통과 판정을 내리지 않는다. **K나 Δ는 사용자 승인 없이 바꾸지 않는다.** (요청당 latency만으로 완료 건수를 추정하지 않는다 — 요청 rate가 낮으면 latency가 짧아도 건수가 적을 수 있다.)

### 3.2 TinyLLaMA
- 저장소 파일은 909.7 MB(`model.onnx` + `.data`)로 원고의 2,488 MB와 다름. 먼저 (a) 이 파일이 어떤 variant인지(onnx metadata, quantization 여부), (b) 원고 서술에 맞는 파일을 확보할 수 있는지 보고. 3.1과 같은 절차와 같은 계측 구간 조건. 파일 문제가 해결되지 않으면 제외 후보.

### 3.3 판정과 승인
- GPT-2와 TinyLLaMA는 **개별 판정**한다(하나의 성패가 다른 쪽에 영향 없음). 결과를 `pilot/lm/LM_RESULT.md`에 모델별로: 실행 가능 여부, device별 latency·계측 구간 조건 통과 여부, 구현 비용, 남은 문제. **포함/제외는 모델별로 사용자 승인.**
- 기존 LM "실행 기록"(이미지 분류로 돈 것)은 어떤 형태로도 새 결과와 섞지 않는다.

### 3.4 Family 재구성 규칙 (LM 결정 후)
- 둘 다 포함: 기존 29 family 그대로.
- 일부 또는 전부 제외: 제외된 모델의 instance를 뺀 조합이 (i) 기존 다른 family와 같으면 제거(중복), (ii) N=0이면 제거, (iii) 그 외 유지. **한 모델만 제외된 경우에도** 같은 규칙을 적용해 family 목록을 정확히 재구성한다. 결과 family 수·N 분포·후보 수(Σ 2^N)·총 run 수(× 4 rates, + 반복)를 표로 보고하고, 기존 29개와의 대응(유지/변형/제거)을 명시.

---

## 4. D10 — Drop metric 정의 (Phase 1에서 셋 다 계산, 채택은 사용자 승인)

**기본 후보는 (b) D_macro다.** (a)·(c)는 비교용으로 계산한다. 보고서의 분석(best-measured 비교, α 민감도, 원고 영향)은 (b)를 중심으로 쓰고 (a)·(c)는 나란히 둔다.
**채택 근거**: D_macro를 채택하는 이유는 "application별 drop 부담을 목적함수에 반영한다"는 연구 목적이지, pilot에서 ranking reversal이 관찰되었는지 여부가 아니다. Pilot의 (a)/(b) 비교는 정의가 데이터에서 실제로 다르게 작동하는지 보는 sanity check로만 보고하고, 채택 여부는 사용자가 결정한다.

| 정의 | 식 | 범위·정규화 | 성격 |
|---|---|---|---|
| (a) aggregate (원고) | `R = Σ_i dropped_i / Δ`, `D = R / R_max` | workload별 R_max | `F + R ≈ Λ`이면 S가 F에 단조 → oracle = throughput-max, α 무의미 |
| (b) D_macro | `q_i = dropped_i / generated_i`, `D_macro = (1/N) Σ_i q_i` | [0,1], 추가 정규화 없음 | **조건부 동치**: 같은 workload의 후보 간에 instance별 generated rate가 같고(`generated_i ≈ λ_i·Δ`), failed와 Δpending이 무시 가능하면, `S = F/F_max − α·D_macro`의 최대화는 `Σ_i w_i f_i` (w_i = 1/F_max + α/(N·λ_i))의 최대화와 동치다. 이 조건에서 α는 총 throughput 대 **요청 rate가 낮은 instance**의 완료에 대한 **가중치**다(요청 rate는 모델의 무게와 별개 설정이므로 "무거운 모델"이라고 쓰지 않는다). 독립적인 두 목표가 아님과 그 조건을 원고에 명시 |
| (c) D_max | `max_i q_i` | [0,1] | 최악 instance. 비선형. 참고 후보 |

- **활성 instance에서 `generated_i = 0`이면 그 run은 invalid**다. `q_i = 0`으로 대체하지 않는다. run을 `invalid_reason = "zero_generated"`로 표시하고, 원인(feeder 미시작, tick 전부 missed, 설정 오류)을 조사해 재실행 대상으로 분류한다. invalid run은 어떤 집계에도 넣지 않는다.
- 모든 instance의 λ_i가 같으면 (b)도 F에 단조가 됨 — 같은 모델만 있는 family에서 해당. 보고서에 해당 family 목록.
- α scale: (a)와 (b)는 D의 scale이 다르므로 α=0.3을 그대로 쓸 근거가 없다. Phase 1 pilot 데이터로 α ∈ {0.1…1.0}에서 best-measured placement의 변화를 (b)·(c)에 대해 표로 보고한다. **이는 탐색적 결과이며, 유리한 α를 고르는 근거로 쓰지 않는다.** α 기본값은 Phase 2 학습 데이터(1·2·4×)에서 별도 절차로 정한다.
- **용어**: pilot은 workload의 일부 후보만 측정하므로 그 안의 최고 배치는 oracle이 아니다. 보고서에서는 **"best measured among pilot candidates"** 로 쓴다. "oracle"은 전체 feasible 후보를 측정한 Phase 2에서만 쓴다.
- Phase 1 보고에 포함: pilot run에서 (a)의 best-measured가 throughput-max와 일치하는 비율, (b)·(c)에서 달라지는 workload와 어떤 instance가 보호되는지.

---

## 5. Pilot — β 선택과 계측 안정성 (2×·4×, 3× 사용 금지)

### 5.0 사전 고정 규칙
- β 후보 {0.25, 0.5, 1.0}. 선택 기준은 아래 5.1–5.4의 **물리·계측 기준만**. Oracle 구성, CPU offloading의 score 효과, 예측기 성능은 **보지 않는다** (계산은 §4 보고용으로만 하고 β 선택 문서에는 넣지 않는다).
- 우선순위: 5.1 통과 → 5.2 통과 → 5.3에서 조건 만족 β 중 **가장 작은 값**. 동률이면 작은 β.
- Pilot run은 `pilot/` 아래에만. Phase 2 데이터에 포함하지 않음.

### 5.1 Smoke·단일 instance (1×, β=0.5 고정)
- 모델 × device 단일 instance. 요청 rate는 GPU isolated throughput 기준이므로 **CPU에서는 정상 실행이어도 완료율이 낮을 수 있다.** 따라서 완료율 기준은 device별로 다르게 두고, 나머지는 분리해서 검증한다.

| 검증 항목 | CPU | GPU |
|---|---|---|
| 모듈 경로·registry sha256 일치 | 필수 | 필수 |
| `failed = 0`, `consumer_failed = 0` | 필수 | 필수 |
| conservation 오차 0, in-flight ≤ 1 | 필수 | 필수 |
| feeder 실현율 `generated/scheduled_ticks ≥ 0.95` | 필수 | 필수 |
| isolated latency(p50)가 offline_profile ±20% | 필수 | 필수 |
| 완료율 | **과부하 여부로 구분** (아래) | `completed/Δ ≥ 0.9 × r_m` |

- CPU 완료율 기준: 요청 rate `r_m`(GPU 기준)과 CPU isolated capacity `c_m^CPU`(offline_profile)를 비교해 구분한다. 세 값(`r_m`, `c_m^CPU`, `completed/Δ`)을 항상 함께 기록한다.
  - **비과부하** (`r_m < 0.8 × c_m^CPU`): 완료율은 요청 rate에 의해 제한되므로 `completed/Δ ≥ 0.9 × r_m`이고 `dropped_full/generated < 1%`.
  - **과부하** (`r_m ≥ 1.2 × c_m^CPU`): `completed/Δ`가 `c_m^CPU`의 ±20% 안이고 `dropped_full > 0`.
  - **경계** (그 사이): 두 기준 중 하나를 만족하면 통과, 어느 쪽에 해당했는지 기록.
- "모델별 latency가 서로 달라야 한다"는 조건은 두지 않는다(registry sha256 기록이 같은 파일 load를 직접 잡는다).

### 5.2 Arrival 실현율·안정성 (β별, 2×·4×)
- N=4·N=8 family 각 1개, 전부-GPU·전부-CPU·반반 3개 배치.
- 통과: 모든 instance `generated/scheduled_ticks ≥ 0.95`, `missed_ticks/scheduled_ticks ≤ 5%` (`scheduled_ticks`는 §2.3대로 실제 계측 구간 기준). 두 번 실행해 F 상대 차이 ≤ 5%. 미달 시 feeder process 분리 후 재검증.
- **배치 간 generated 일치** (같은 workload, instance별): `scheduled_ticks ≥ 200`이면 상대 차이 ≤ 2%; `scheduled_ticks < 200`(LM 등 저-rate instance)이면 **절대 차이 ≤ 2건**. 어느 기준을 적용했는지 instance별로 기록. 기준은 pilot 전에 고정하며 결과를 보고 바꾸지 않는다.

### 5.3 GPU 부하 존재 (β별, 4×)
- 검증 대상(β별로 실행): (A) 5.2의 전부-GPU 배치(2×·4×), (B) **β별 1× 단일 GPU instance, 모든 모델** — (iii)을 실제로 확인하려면 β마다 1× 단일 run이 있어야 하므로 §5.1(β=0.5 고정)과 별도로 β=0.25·1.0에서도 단일 GPU run을 실행한다(모델당 1 run, 16 s × 모델 수 × 2 β).
- 통과: (A)에서 (i) `dropped_full > 0`인 GPU instance 존재, (ii) GPU utilization 평균 ≥ 80% **또는** GPU instance latency p50이 isolated 대비 ≥ 1.5×; (B)에서 (iii) 모든 모델의 1× 단일 GPU instance가 `dropped_full / generated < 1%`.
- queue overflow만으로 포화를 단정하지 않는다: (ii)가 없으면 CPU 전처리·feeder·worker thread 병목을 `preprocess_ms`, thread별 CPU 사용률로 분석해 원인을 적는다.

### 5.4 반복 편차 (선택된 β, 2×)
- N=6 family 1개, 후보 8개(전부-GPU, 전부-CPU, 무작위 6), 3회. F 상대 차이 중앙값·최대, q_i 차이.

### 5.5 §4 보고용 계산 (β 선택과 분리)
- 5.2–5.4 run 전체에서 세 정의의 best-measured placement(pilot 후보 안에서), `F + R` vs `Λ` 오차 분포, α별 best-measured 변화. 결과가 기대와 달라도 수정하지 않고 기록. 이 결과는 β 선택·α 선택·D10 채택의 근거로 쓰지 않는다(§4·§5.0).

---

## 6. CPU–NPU 호환성 조사 (읽기 전용, 저장 JSON)

- **데이터 수 관계**: `results_antara/` 372 rows(= 93 placements × 4 rates)와 원고의 "279 placements, 1,116 samples"의 관계를 확정한다(REPORT.md §B-2: 279 = train rows, 1,116은 데이터에 없는 수). 원고 수정용으로 정확한 집계 단위를 적는다.
- **Counter 신뢰성 검증(필수, 재계산 전)**: NPU 측정 코드(`new_main` branch 근처, REPORT.md §0.1)를 읽어 (i) `inference_count`의 집계 구간이 `window_sec=30`과 일치하는지(warm-up reset 시점, 저장 시점), (ii) `dropped_frames_due_to_full_queue`가 **모든 instance에서 실제로 집계되는지** — 특히 CPU–GPU처럼 headless 경로에서 0으로 고정되는 코드(`unified_viewer.py:1708` 상당)가 NPU branch에도 있는지, (iii) NPU에 headless instance가 정말 없는지(REPORT.md의 "NPU는 headless가 없다"를 코드와 JSON으로 재확인), (iv) throughput 필드가 `1000/latency`인지(1,015/1,020 instance에서 확인됨, 나머지 5개는 무엇인지).
- 재계산 가능성 판정: (ii)·(iii)이 통과하면 T는 `Σ completed/30`으로 재집계 가능하고, D_macro는 `generated ≈ completed + dropped` 근사(pending 무시, 오차 상한 instance당 queue 2 + 처리 중 1)만 가능하다. **(i)–(iv) 중 하나라도 신뢰할 수 없으면 근사 D_macro나 best-measured를 정식 결과로 계산하지 않고 "재계산 불가"로 보고한다.**
- 통과한 경우에만: 근사로 (a)·(b)·(c)를 계산해 3× N≥3 4 workloads의 best-measured와 저장된 Ours 선택(α=0.2·0.3)의 위치를 표로. 근사 결과임을 모든 표에 명시. 예측기 재학습은 하지 않는다(NPU artifact 불일치 문제는 별개).
- 보고: "NPU 데이터는 새 정의로 정확히 재계산할 수 없다. 선택지: (1) NPU 보드에서 재측정 372 runs ≈ 3.2 h + 새 runtime 이식, (2) 근사 재계산을 한계 명시 후 사용(counter 검증 통과 시에만), (3) NPU 평가 범위 축소." **결정은 사용자.**

---

## 7. Per-instance baseline 설계 (코드 작성 금지, Phase 2 이후 구현)

공통: instance별 feature(identity, device, λ_i, isolated capacity, co-runner 수·rate 합 같은/다른 device, N), XGBoost 2개 공유, family GroupKFold, 선택은 baseline 자신의 예측만 사용, 평가는 공통 실측 S.

| | (a) aggregate 유지 시 | (b) D_macro 채택 시 |
|---|---|---|
| Target | `t_i/λ_i`, `d_i/λ_i` | `t_i/λ_i`, **`q_i = d_i/generated_i`** |
| 합성 | `F̂ = Σ clip(t̂_i)`, `R̂ = Σ clip(d̂_i)` | `F̂ = Σ clip(t̂_i)`, `D̂_macro = mean_i clip(q̂_i, 0, 1)` |
| 정규화 | `F̂/max F̂`, `R̂/max R̂` (0이면 0) | `F̂/max F̂`만. D̂_macro는 정규화 없음 |
| Ŝ | `T̂ − α D̂` | `T̂ − α D̂_macro` |

- **복원 명시**: `t_i/λ_i`로 학습한 예측값은 `t̂_i = clip(pred, 0) × λ_i`로 raw throughput(req/s)으로 복원한 뒤 합산한다(`F̂ = Σ_i t̂_i`). (a)의 `d_i/λ_i`도 같은 방식으로 `d̂_i = clip(pred, 0) × λ_i` 복원 후 합산. (b)의 `q̂_i`는 비율 자체가 target이므로 복원 없이 `clip(·, 0, 1)` 후 평균. λ_i는 schedule의 요청 rate(placement 무관)이며 측정값이 아니다.

필요 코드: `build_instance_rows`, `fit_instance_models`, `compose_{agg,macro}`, `select` — 함수 signature는 REPORT.md §D-3 유지, compose만 두 변형. 예상 350 LOC + 변형 50 LOC.

---

## 8. Phase 1 보고서 (`revision_profiling/REPORT_PHASE1.md`)와 승인 조건

보고서 항목:
0. 환경·branch·변경 파일(`CHANGES.md` 요약), 모듈 경로 검증 결과
1. Runtime·counter 구현 결과, smoke test(5.1), conservation 오차
2. LM 시도 결과(§3), 권장안, 재구성 시 family 표
3. β pilot: 기준별 통과표, 선택된 β와 근거(물리 기준만), arrival 실현율·배치 간 generated 차이, GPU utilization, 반복 편차
4. Drop metric 분석(§4·5.5): 세 정의의 best-measured 일치/불일치(pilot 후보 안에서), `F+R` vs `Λ`, α 민감도(탐색적), instance별 q_i 분포, invalid run 목록과 사유
5. CPU–NPU 호환성(§6): counter 신뢰성 검증 결과, 데이터 수 관계, 재계산 가능/불가 판정, 선택지
6. Phase 2 예상: family·run 수, 대표 run wall-clock 기반 총 시간, 반복 포함 시간, 측정 순서
7. 원고 영향: (i) 새 정의 채택 시 교체 대상(Fig. 1(b), Fig. 4, Table 3(a)(b), §3 정의, α 근거, Abstract·Contribution의 multi-objective 서술, per-instance baseline 사양), (ii) **중간 rate(1.5/2.5/3.5×) 제외에 따라 삭제·수정해야 할 일반화 결과**: §1 56행 근처의 generalization 서술, §5.1.3 "Profiling interval and generalization evaluation"의 54 schedules/3,504 samples, §5.2 중간 rate 문단(Top-1 0.72/0.88/0.88, Top-5 0.94/1.0/0.94)과 "broad range of unseen workload scales" 결론, §7 "observed generalization to unseen workload intensities", R1-3 response의 해당 부분. 원고 .tex 줄 번호는 `revision_map.md` 기준.

**사용자 승인 항목**: LM 포함 여부와 family 구성, D10, β, CPU–NPU 처리, Phase 2 실행. 승인 후에는 모델·family·목적함수·β를 고정하고, **3× hold-out 결과를 보고 어떤 설정도 바꾸지 않는다.**

---

## 9. Phase 2 개요 (승인 후 별도 지시문)

- 전체 campaign, 측정 순서는 **1·2·4× (training rates) → 3× hold-out → 3× N≥3 반복 2회**. 3× 결과는 설정이 고정된 뒤 마지막에 측정하며, 측정 중·후에 어떤 설정도 바꾸지 않는다. `revision_profiling/results/`.
- 중간 rate는 제외(D5). 그에 따른 원고 수정 범위는 Phase 1 보고서 7(ii)에 따른다.
- `recompute_scores` 사본 → `dataset/` → `revision_exp/run_all.py --data_root …` (revision_exp는 adapter/사본으로만 변경).
- Reviewer 대응 실험(predictor 비교, ablation, per-instance baseline)은 별도 지시문.

---

## 10. 작업 규칙 요약
- 원본 파일 불변, 모든 산출물은 `revision_profiling/`. 모듈 경로 assert.
- Pilot 결과는 campaign에 섞지 않음. β 선택 문서에 best-measured·예측기 결과를 쓰지 않음. Pilot에서 "oracle"이라는 말을 쓰지 않음.
- invalid run(`zero_generated`, in-flight > 1, 모듈 경로 위반, conservation 불일치(±1 초과), consumer_failed > 0)은 집계에서 제외하고 목록·사유를 보고.
- 통과 기준에 맞추기 위해 counter 정의·기준·parameter를 바꾸지 않음. 예상과 다른 결과는 그대로 기록.
- 장시간 실행은 `tmux`/`nohup`, 예상 시간 사전 보고. `CHANGES.md`에 모든 코드 변경 기록.
- [확인] / [추정] / [확인 불가] 구분.
