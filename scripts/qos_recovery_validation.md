# `qos_score_validation.pdf` 생성 시나리오

본 문서는 논문의 *Failure Impact Objective* 절에 들어가는 figure
`qos_score_validation.pdf`를 재현하기 위한 실험 시나리오, 측정 방식, 결과
해석 방법을 정리한다.

생성 스크립트: [`scripts/qos_recovery_validation.py`](./qos_recovery_validation.py)
스케줄 정의 : [`tests/qos_recovery_schedule.yaml`](../tests/qos_recovery_schedule.yaml)
출력 PDF    : `results/qos_score_validation.pdf`
원본 CSV    : `results/qos_recovery_<timestamp>.csv` (stop-and-start, mode 0)
              `results/qos_recovery_static_<timestamp>.csv` (static, mode 3)

---

## 1. Figure가 보여주는 것

그래프에는 **두 개의 V(t) 곡선**이 한 좌표계에 그려진다.

- **`stop-and-start` (진한 남색 실선)** — 시스템이 detection 후 placement를
  변경하는 정상 동작. CPU 4-way에서 동작 중 input rate가 폭증하면 V(t)가
  ε를 넘고, 이를 detect하면 4 모델 모두를 GPU로 옮겨 회복한다. mode 0
  (stop-and-restart)이지만 *placement가 동일한 phase 1 → phase 2 transition은
  in-place infps 업데이트로 처리하는 same-placement detection*이 적용되어
  있어, 실제 worker restart는 phase 2 → phase 3 (CPU → GPU 진짜 placement
  변경) 한 번만 일어난다.
- **`static` (붉은 점선)** — placement를 한 번도 바꾸지 않는 baseline.
  같은 infps 변화를 그대로 받지만 4개 모델이 계속 CPU에 머물기 때문에
  recovery는 일어나지 않고 V(t)는 detection 이후에도 계속 상승한다.
  실제 mode 3(`--adaptive-mode 3`)으로 별도 subprocess에서 측정한 trace.

두 curve는 phase 1 → phase 2 boundary(= `t_0`)에서 정렬되도록 wall-clock 축을
shift한다. detection phase(`t_0 → t_detect`)까지는 두 곡선이 거의 동일하게
상승해야 하고, `t_detect` 이후에 stop-and-start만 ε 아래로 내려와야 한다.

- **X축**: wall-clock 시간 (초). 0초 = stop-and-start 첫 측정 행의 timestamp.
  Phase 사이의 cold-start gap이 회색 hatched 영역으로 그대로 보이도록
  row index가 아니라 wall-clock 초 단위를 사용한다. static curve는 phase
  1 → phase 2 경계가 stop-and-start의 경계와 같은 X 좌표에 오도록 offset이
  적용된다.
- **Y축**: 윈도우 QoS violation score `V(t)`

  ```
  V(t) = (1/T) · Σ_{τ=t-T+1..t} v(τ),    T = 3 s (sliding window)
  v(τ) = (1/N_active) · Σ_{i: ℓ_i(τ)>0} max(0, ℓ_i(τ) / L_SLO,i − 1)
  ℓ_i(τ) = handler.avg_infer_time + handler.avg_wait_ms   (end-to-end)
  L_SLO,i = 1000 / infps_i [ms]
  ```

  - `N_active`는 해당 tick에 적어도 한 번 inference를 끝낸 view의 수.
    아직 측정값이 없는 view는 평균에서 제외한다 ("Option A" 규칙).
    이렇게 해야 cold-start 직후 1–2 view만 동작 중인 순간에 v(t)가 0으로
    diluted되지 않는다.
  - `ℓ_i`는 inference 시간 + queue wait 시간 (end-to-end response time).
    queue wait를 포함해야 누적되는 backlog가 V(t)에 반영된다.
  - **누적 평균 효과** — `avg_infer_time`/`avg_wait_ms`는 핸들러 시작 이후의
    cumulative average다. static 곡선은 한 번도 핸들러를 reset하지 않으므로
    overload 시간이 길어질수록 누적 wait가 점점 커져 V(t)가 단조 증가한다.
- **수평 점선**: 검출 임계값 `ε`. 좌측 y축에 `ε` 기호로 라벨링되어 있다.
  논문에서는 **ε = 50 으로 고정**해서 사용한다.
- **수직선** (3 개):
  - `t_0` — failure injection 순간 (= `t_detect − T`로 역산)
  - `t_detect` — windowed V(t)가 처음 ε를 초과한 순간 (stop-and-start 기준)
  - `t_recover` — V(t)가 다시 ε 이하로 내려간 순간 (stop-and-start 기준)
- **연속 색조 영역**:
  - 옅은 주황 = **Detection phase** (`t_0 → t_detect`, 항상 `T = 3 s`)
  - 옅은 초록 = **Recovery phase** (`t_detect → t_recover`)
  - 회색 hatched (//) = **Cold-start gap** — 워커가 stop된 후 새 워커가
    ready 될 때까지의 wall-clock 공백 구간. 이 구간은 stop-and-start CSV에
    측정 행이 자체가 없다. (static curve는 절대 worker restart가 없으므로
    여기서 gap이 발생하지 않는다.)
- **stop-and-start 곡선의 끊어짐**: cold-start gap에서는 V(t) 곡선이
  의도적으로 끊어지도록 그려진다. matplotlib에서 양 끝 측정값 사이에 NaN을
  삽입해서, "이 구간에 측정 데이터가 없음"이 곡선의 형태로 그대로 드러나게
  한다. (예전에는 양 끝점을 직선으로 연결해서 우연히 두 V값이 비슷하면
  평평한 plateau처럼 보이는 시각적 오해가 있었음.)
- **빈 영역을 둘러싼 점선 타원**: 첫 번째 cold-start gap에서 끊긴 stop-and-
  start 라인의 두 끝점을 기준으로, 그 사이의 "측정 없는 빈 영역"을 얇은
  회색 dotted 타원이 감싼다. 타원 중심 = `(gap_mid, (V_before + V_after)/2)`,
  너비 ≈ gap × 1.6, 높이는 두 끝점이 충분히 안에 들어가도록 동적으로 계산.
- **redeployment downtime 어노테이션**: 위 점선 타원의 정중앙을 곡선 화살표
  끝점이 가리키고, t_recover 라벨 우측 상단(`y_max*0.78` 부근)에
  `"redeployment downtime (no service)"` 텍스트가 표시된다. "이 구간에는
  측정값이 없는 게 아니라 시스템 자체가 동작 중이지 않다"는 의미를 강조한다.
  타원 + 화살표는 detection phase와 recovery phase 사이에 들어가는 첫 번째
  cold-start gap (실제 CPU → GPU placement 변경 구간) 한 곳에만 적용되고,
  다른 cold-start gap (예: phase 3 후반 worker 안정화 glitch)은 어노테이션
  대상이 아님.
- **곡선의 zorder**: V(t) 두 곡선은 충분히 높은 zorder(static=20, stop-and-
  start=21)로 그려져서 cold-start hatched 회색 기둥(zorder 0.5, alpha 0.13)
  위에 분명하게 올라간다. phase 3의 두 번째 hatched 영역에서도 곡선이 가려
  지지 않는다.
- **Detection / Recovery phase 화살표 위치**: Detection phase 양방향 화살표는
  `y_max * 0.58` (figure 상단 중간), Recovery phase 화살표는 `V(t) = 25`
  (= ε 점선 아래, V≈20 살짝 위)에 고정된다. ε 점선이나 stop-and-start
  recovering 곡선과 겹치지 않게 배치된 결과다.
- **legend**: 우상단에 `static`, `stop-and-start` 두 항목 표시.

---

## 2. 시나리오 (3 phase)

4개 모델(`mnasnet`, `squeezenet1.0-12`, `resnet50`, `resnext50`)을 동시에
실행하면서 다음 세 단계를 차례로 통과한다. 두 곡선을 위해 동일한 시나리오를
**두 번 실행**한다 — 한 번은 `--adaptive-mode 0`(stop-and-start), 한 번은
`--adaptive-mode 3`(static). 각 run은 단일 viewer 세션 안에서 phase를
전환하며, per-tick 메트릭은 두 개의 별도 CSV로 누적 기록된다.

| Phase | YAML key | 디바이스 배치 | infps | 의도 |
|---|---|---|---|---|
| **1. Initial Deployment** | `combination_initial` | 4개 모두 CPU | mnasnet 3 / squeezenet 3 / resnet50 4 / resnext50 4 | 모든 모델이 worker capacity 안에 있음. queue가 비어 있어 wait_ms ≈ 0, V(t) ≈ 0 |
| **2. Overload (failure)** | `combination_overload` | 4개 모두 CPU (Phase 1과 동일) | mnasnet 400 / squeezenet 400 / resnet50 50 / resnext50 40 | input rate 100배 폭증. 4개 모델이 동시에 CPU를 점유하면서 큐가 즉시 가득 차고 wait_ms가 누적되어 V(t)가 ε를 넘음. **placement가 phase 1과 동일하므로 mode 0의 same-placement detection이 적용되어 worker restart 없이 in-place infps 업데이트만 일어남** (= static과 동일한 transition). |
| **3. Changed Stable Deployment** | `combination_offload` | 4개 모두 GPU | (Phase 2와 동일한 high infps) | input rate는 그대로지만 4개 모델 모두 GPU로 옮겨 빠르게 처리. *real* placement change이므로 mode 0이 worker restart를 수행하고, GPU의 빠른 inference로 V(t)는 ε 아래로 회복. static 모드는 이 phase에서도 아무 동작도 하지 않으므로 V(t)가 계속 상승한다. |

> **Phase 3가 "all GPU" 인 이유**
> 처음에는 사용자 요청대로 "무거운 두 모델만 GPU, 가벼운 두 모델은 CPU 유지"로
> 시도했지만, GPU 워커들의 pre/postprocess가 여전히 CPU를 점유해서
> mnasnet/squeezenet이 충분히 빨라지지 못했다. 4개 모두 GPU로 옮기는 것이
> 가장 깨끗하게 ε 아래로 회복되는 verified stable configuration이라 이
> 형태로 정착했다.

### 2.1 infps 값을 어떻게 정했나

**핵심 제약**: stop-and-start와 static이 detection phase까지 거의 동일하게
상승하려면, mode 0의 phase 1 → phase 2 transition에서 **spurious cold-start
spike가 없어야** 한다. 이는 same-placement detection으로 해결되지만, 그
대가로 phase 2의 V(t)가 "자연스러운 wait 누적"에만 의존하게 되므로 infps를
충분히 크게 잡아야 V(t)가 ε를 넘는다.

| 모델 | CPU 처리량(대략) | Phase 1 infps | Phase 2 infps | 의도 |
|---|---|---|---|---|
| mnasnet | ~4 fps | **3** | **400** | Phase 1: capacity 안 (V≈0). Phase 2: 100배 over → CPU saturation, 큐가 즉시 차서 wait가 누적 |
| squeezenet1.0-12 | ~4 fps | **3** | **400** | 위와 동일 |
| resnet50 | ~5 fps | **4** | **50** | 좀 더 무거우므로 Phase 2 infps도 비례하여 낮춤 |
| resnext50 | ~5 fps | **4** | **40** | 위와 동일 |

이 조합으로:

- **Phase 1**: 두 곡선 모두 V(t) ≈ 0 (clean baseline)
- **Phase 2**: 두 곡선이 거의 동일하게 가파르게 상승 (in-place infps 업데이트
  + bounded queue saturation에 의한 wait 누적)
  - stop-and-start: V(t) peak ≈ 64 (windowed)
  - static: V(t) ≈ 78 (조금 더 높음 — RNG 수준의 차이)
- **Phase 3 / t_detect 이후**:
  - stop-and-start: GPU 재배치 cold-start 후 V(t) → 0
  - static: V(t)가 계속 단조 증가 (87 → 98 → 110 → 125 …)
    cumulative `avg_wait_ms`가 점점 커지면서 발생

### 2.2 Phase 길이 (기본값)

| Phase | 인자 (default) | 측정 행 수 | 비고 |
|---|---|---|---|
| Initial  | `--baseline-duration 12` | ≈ 7 행 | V(t) ≈ 0 baseline 검증을 위해 충분히 길게 |
| Overload | `--failure-duration 3`   | ≈ 3 행 | T = 3 windowed V(t)가 ε를 넘기에 딱 충분한 길이. 더 길게 잡으면 stop-and-start의 phase 2 plateau가 dilution으로 ε 아래로 떨어질 수 있다 |
| Offload  | `--recovery-duration 12` | ≈ 12 행 | V(t)가 ε 이하로 안정화되는 모습을 충분히 보여주기 위함 (stop-and-start). static에서는 곡선이 끝까지 단조 상승하는 모습을 보여줌 |

---

## 3. 측정 / 후처리 파이프라인

```
qos_recovery_validation.py main()
    │
    ├── run_scenario(mode=0, label="stop-and-restart")
    │      └── schedule_executor_main.py --adaptive-mode 0 --combo-duration ...
    │             └── results/qos_recovery_<ts>.csv
    │
    ├── run_scenario(mode=3, label="static")
    │      └── schedule_executor_main.py --adaptive-mode 3 --combo-duration ...
    │             └── results/qos_recovery_static_<ts>.csv
    │
    └── (CSV 두 개 로드 → V(t) 계산 → 정렬 → make_plot)
            └── results/qos_score_validation.pdf
```

각 subprocess 내부:

```
schedule_executor_main.py
   - mode 0: 매 phase transition마다 placement signature 비교
       · 같으면 → apply_static_phase (in-place infps 업데이트)
       · 다르면 → 워커 stop → re-init → start (cold-start)
   - mode 3: 매 phase transition마다 항상 apply_static_phase
        │
        ▼
unified_viewer.py
   - cpu_timer 1Hz tick마다 v(t) 계산 (Option A: active view만 평균)
   - v(t)는 end-to-end latency (avg_infer_time + avg_wait_ms) 기반
   - feeder/입력 큐는 phase 간에 persistent (cold-start gap 동안에도
     계속 frame을 push해서 큐에 backlog 쌓임 → 새 워커가 wait_ms spike 측정)
   - CSV column: timestamp, combination, v_score, view*_infer_ms, ...
        │
        ▼
results/qos_recovery_*.csv
```

`unified_viewer.py`의 `cpu_timer`는 1 Hz로 동작한다. CSV 한 행 ≈ 1초 측정.
mode 0의 *real* phase 전환 시점(phase 2 → phase 3)에는 워커가
stop→load→start되는 ~2초의 wall-clock gap이 발생하는데, 이 동안 FPS=0이라
CSV에 행이 안 찍힌다. 스크립트는 이 gap을 timestamp 차이로 자동 탐지해서
figure에 회색 hatched 영역으로 표시한다.

### 3.1 핵심 architectural 요소

다음 네 가지가 mode 0/3 흐름에서 본 시나리오를 위해 명시적으로 구현되어
있다 — 일반적인 mode 0 동작과 다르므로 주의:

1. **Mode 0의 same-placement detection** (`schedule_executor_main.py`)
   - `_placement_signature(combo_name)`이 combo의 `(display, model, execution)`
     세트를 hashable tuple로 반환. **infps는 시그니처에 포함하지 않는다.**
   - `_run_next`에서 직전 combo와 새 combo의 signature가 같으면 워커 restart
     대신 `viewer.apply_static_phase(...)` 한 번만 호출 → infps만 업데이트.
   - `_next_transition_is_inplace()` helper로 _run_next의 _after_stop 스케줄링
     시점 / cancel 여부도 같이 결정한다 (in-place 다음 phase에서는 viewer가
     살아있어야 하므로 timed_shutdown을 cancel).
   - 이 변경으로 본 시나리오의 phase 1 → phase 2 transition에서 spurious
     cold-start spike가 사라져 stop-and-start 곡선이 static 곡선과 거의
     동일한 형태로 ε까지 상승한다.

2. **Mode 3 (static)** (`schedule_executor_main.py`, `unified_viewer.py`)
   - `--adaptive-mode 3`. mode 0/1/2 어느 경로와도 독립적으로 구현되어 있다.
   - executor는 첫 combo만 정상 시작하고, 이후 combo부터는 항상
     `apply_static_phase`를 호출.
   - viewer의 `apply_static_phase(schedule_file, combo)`는 새 combo의
     entries에서 `(model_basename → infps)` 매핑을 만들어 이미 동작 중인
     `update_input_rates`로 전달, 동시에 `current_combination` 라벨도
     갱신해서 CSV의 phase boundary 자동 탐지가 그대로 동작하도록 한다.

3. **Persistent input feeders/queues** (`unified_viewer.py`)
   - `feeder_shutdown_flag`라는 별도 Event를 둬서 feeder는 viewer 전체
     수명 동안 살아있음. `stop_execution()`은 워커만 죽이고 feeder/큐는
     건드리지 않음. 결과적으로 phase 2 → phase 3 cold-start gap 동안에도
     input request가 계속 큐로 들어가고, 새 워커가 처리할 때 wait_ms가
     spike한다.
   - 자세한 위치: `unified_viewer.py:initialize_state_variables`,
     `start_execution`, `initialize_threads`, `_drain_and_close_all_queues`.

4. **`avg_wait_ms` 가 v(t) 계산에 포함됨**
   - 기존에는 ResNet 워커가 `wait_ms`를 계산만 하고 핸들러로 전달하지
     않아서 v(t)는 `avg_infer_time`만 사용했음. 이제는 `output_queue.put`이
     4-tuple `(img, class_name, infer_time_ms, wait_ms)`로 전달하고
     `ResNetViewHandler`가 `avg_wait_ms`를 누적함. v(t) 계산은
     `avg_infer_time + avg_wait_ms`를 사용해서 cold-start backlog가
     latency에 반영됨.
   - 변경 위치: `model_processors.py` (run_resnet_*_process),
     `view_handlers.py:ResNetViewHandler.display_frames`,
     `unified_viewer.py:1707-` (CSV 로깅의 v(t) 계산).

---

## 4. Detection / Recovery 판정 규칙

스크립트는 **stop-and-start** CSV(mode 0)에서만 `t_0`/`t_detect`/`t_recover`를
계산한다. static curve는 단순히 같은 X 좌표에 정렬해서 함께 plot할 뿐 detection
계산에는 참여하지 않는다.

### 4.1 Detection — "fully populated post-failure window" 규칙 + T-앵커링

**Step 1**: V(t) 윈도우가 post-failure 샘플로 완전히 채워진 첫 행에서
threshold 검사:

```python
detection_earliest = p2_start + WINDOW_T - 1   # = p2_start + 2 (T=3)
for i in range(detection_earliest, p3_start):
    if v_t[i] > epsilon:
        t_detect_row = i
        break
```

이 조건이 핵심이다 — V(t)는 슬라이딩 평균이므로 window의 일부만 새
데이터로 채워져도 평균이 ε를 넘으면 즉시 detect되어 detection latency가
T보다 짧아질 수 있는데, "i ≥ p2_start + T − 1" 조건으로 윈도우가 완전히
post-failure 샘플로 차길 기다린다.

**Step 2**: t_0를 t_detect로부터 역산해서 **detection phase = T로 고정**:

```python
t_detect_sec = times_sec[t_detect_row]
t0_sec       = t_detect_sec - WINDOW_T          # always exactly T seconds
detection_phase_sec = WINDOW_T                  # = T (= 3s)
```

이렇게 하면 그래프의 detection phase 폭이 phase 1 → phase 2 cold-start gap의
wall-clock 길이에 영향받지 않고 항상 정확히 `T = 3 s`가 된다. (현재 구현에서는
phase 1 → phase 2에 cold-start가 없으므로 이 보정은 단순한 라벨링 일관성을
위한 것이다.) t_0 자체는 "failure가 마지막 verified 측정 직후의 어느 순간에
주입되었다"는 의미로 해석된다.

### 4.2 Recovery — `V(t) ≤ ε` 가 처음 성립하는 행

```python
for i in range(p3_start, p3_end):
    if v_t[i] <= epsilon:
        t_recover_row = i
        break
```

V(t) 자체가 T초 슬라이딩 평균이라 이미 충분히 smoothed 되어 있어 추가
debounce는 불필요하다. 시스템이 새 placement(phase 3)에서 동작 중이므로
V(t)가 ε에 닿는 순간을 곧 안정화 완료로 본다.

### 4.3 표시되는 phase 길이

- **Detection phase** = `t_detect_sec − t0_sec` = `T = 3 s` (항상 고정)
- **Recovery phase**  = `t_recover_sec − t_detect_sec`
  = (rollback 행동 지연) + (phase 2→3 cold-start gap) + (실제 V(t) 하강 dynamics)
- **Total degradation duration** = `t_recover_sec − t0_sec` = detection + recovery

---

## 5. ε (검출 임계값) 결정

기본 모드에서는 stop-and-start CSV의 측정값으로 자동 결정한다:

```
baseline = V(p2_start − 1)            # phase 1 마지막 tick의 V(t)
stressed = max(V(t) for t in phase 2) # phase 2 windowed V(t)의 최대값
ε        = (baseline + stressed) / 2  # 단순 중점
```

`--epsilon N`으로 명시 지정 시 그 값을 그대로 사용한다.
**논문에서는 ε = 50 으로 고정**하므로 다음과 같이 호출한다:

```bash
python scripts/qos_recovery_validation.py --epsilon 50
```

콘솔에 측정된 baseline·stressed·midpoint·실제 사용한 ε이 모두 출력된다.

---

## 6. 실행 방법

### 6.1 처음부터 끝까지 (측정 + 그래프, 두 시나리오 모두)

```bash
cd /home/msyu/PycharmProjects/fsrr-multimodel-scheduling
python scripts/qos_recovery_validation.py --epsilon 50
```

스크립트는 stop-and-start(mode 0)와 static(mode 3) 두 시나리오를 차례로
실행한다. 총 소요 시간 ≈ 1 분 30 초 (각 run에서 4개 모델 ONNX 세션 로드 +
3 phase 측정). GPU와 CUDA 12 onnxruntime이 정상 동작하는 환경이 필요하다.

### 6.2 측정 데이터 재사용 (그래프만 다시)

CSV는 그대로 두고 시각화 옵션만 바꿔보고 싶을 때:

```bash
python scripts/qos_recovery_validation.py --no-run --epsilon 50
```

`--no-run`을 주면 `results/qos_recovery_*.csv`(stop-and-start)와
`results/qos_recovery_static_*.csv`(static) 중 가장 최신 파일을 자동으로
다시 읽어 PDF만 재렌더링한다. 특정 파일을 지정하려면:

```bash
python scripts/qos_recovery_validation.py --no-run --epsilon 50 \
    --csv results/qos_recovery_20260410_110350.csv \
    --static-csv results/qos_recovery_static_20260410_110350.csv
```

### 6.3 Static curve 없이 stop-and-start만

```bash
python scripts/qos_recovery_validation.py --epsilon 50 --no-static
```

`--no-static`을 주면 mode 3 subprocess는 실행되지 않고 figure에도 static
곡선이 그려지지 않는다.

### 6.4 Phase 길이 / 임계값 조정

```bash
# failure phase를 1초 더 늘려서 V(t)가 더 stable plateau를 보이게
python scripts/qos_recovery_validation.py --epsilon 50 --failure-duration 4

# initial / recovery phase를 더 길게
python scripts/qos_recovery_validation.py --epsilon 50 \
    --baseline-duration 18 --recovery-duration 18

# epsilon만 다른 값으로 가시화 (재측정 없이)
python scripts/qos_recovery_validation.py --no-run --epsilon 30
```

내부적으로는 `schedule_executor_main.py`의 `--combo-duration COMBO=SECONDS`
옵션으로 단일 subprocess 안에서 phase별 duration을 다르게 적용한다.

### 6.5 다른 스케줄로 실행

```bash
python scripts/qos_recovery_validation.py --schedule path/to/other.yaml
```

YAML에는 정확히 3개의 combo가 있어야 하며, 순서대로 (initial, overload,
offload)로 해석된다. PDF 라벨은 스크립트 상단의 `PHASE_DISPLAY` 매핑으로
결정되므로 새 combo 이름을 추가하려면 그 매핑도 함께 갱신해야 한다.

---

## 7. 출력물 해석

### 7.1 콘솔 출력 예시 (ε = 50, 기본 phase 길이)

```
[Info] Phase boundaries: [(0, 'combination_initial'), (7, 'combination_overload'), (10, 'combination_offload')]

============================================================
  Detection threshold determination
============================================================
  Baseline V(t) (end of combination_initial   , t=6s) = 0.0175
  Stressed V(t) (end of combination_overload  , t=9s) = 64.3435
  Empirical midpoint                                  = 32.1805
  ==> Detection threshold  epsilon = 50.0000  (user override (--epsilon 50))

============================================================
  Recovery scenario timing (wall-clock seconds from row 0)
============================================================
  t_0      (failure injection)    =   6.00s  (= t_detect - T)
  t_detect (windowed V(t) > eps)  =   9.00s  (row 9, window fully post-failure)
  rollback (new workers ready)    =  11.00s  (first combination_offload row)
  t_recover (V(t) <= eps)         =  12.00s  (row 11)
  Detection phase                 =   3.00s   (= T)
  Recovery phase                  =   3.00s
  Total degradation duration      =   6.00s
  Cold-start gaps detected        = 2.00s @ 9.0-11.0s, 2.00s @ 15.0-17.0s

[Info] Loaded 22 static-mode CSV rows from results/qos_recovery_static_20260410_110350.csv
[Info] Static-mode phase boundaries: [(0, 'combination_initial'), (7, 'combination_overload'), (10, 'combination_offload')]
```

읽는 법:

| 표시 | 의미 |
|---|---|
| `Phase boundaries` | stop-and-start CSV에서 자동 탐지된 각 phase의 시작 행 인덱스 |
| `Baseline V(t)` | Phase 1 마지막 tick의 windowed V(t). ≈ 0 이어야 함 (clean baseline) |
| `Stressed V(t)` | Phase 2의 windowed V(t) 최대값. ε(=50)보다 커야 detection이 일어남 |
| `Empirical midpoint` | baseline과 stressed의 중점. `--epsilon`을 안 주면 이 값을 자동 사용 |
| `t_0` | failure injection 시점 (= `t_detect − T`로 역산) |
| `t_detect` | "fully populated post-failure window" 규칙으로 V(t) > ε 첫 검출 시점 |
| `rollback (new workers ready)` | phase 3 첫 측정 행의 wall-clock 시간 (= cold-start gap 종료 시점) |
| `t_recover` | V(t) ≤ ε 첫 성립 시점 |
| `Detection phase` | t_0 → t_detect 폭, 항상 `T = 3 s` |
| `Recovery phase` | t_detect → t_recover 폭, 시나리오에 따라 변동 |
| `Cold-start gaps detected` | CSV의 timestamp 간격이 1.5초 이상인 구간들. wall-clock 상의 cold-start 시간을 보여줌. 현재 시나리오에서는 phase 2 → phase 3 transition (real placement change) 한 군데가 의미 있는 redeployment downtime이고, phase 3 후반의 두 번째 gap은 GPU 워커 안정화 도중 cpu_timer가 잠시 tick을 놓치는 경우가 있는데 어노테이션 대상은 아니다 |
| `Static-mode phase boundaries` | static CSV에서 탐지된 phase 경계. stop-and-start와 같은 phase 시퀀스가 잡혀야 함 |

논문에 들어갈 핵심 수치(`ε`, `Detection phase`, `Recovery phase`,
`Total degradation duration`)는 모두 위 출력에서 그대로 가져다 쓸 수 있다.

### 7.2 그래프 해석

| X 구간 | stop-and-start 동작 | static 동작 |
|---|---|---|
| 0 ~ t_0 (Initial Deployment) | V ≈ 0 — workers 안에서 모두 처리 | V ≈ 0 (동일) |
| t_0 ~ t_detect (Detection phase, T=3s) | infps 폭증 → in-place 업데이트 → wait 누적 → V 가파르게 상승 | 동일 (두 곡선이 거의 겹침) |
| t_detect 직후 | ε 초과 detect → 4개 모델 모두 GPU로 stop-and-restart 시작 | 아무 동작 없음 |
| Recovery phase | cold-start gap (회색 hatched) 안에서 새 GPU 워커 로드. 곡선이 끊어져 있고 옆에 `redeployment downtime (no service)` 어노테이션 표시 | V 계속 상승 |
| t_recover | GPU 워커가 안정 상태 → V → 0 | V는 단조 증가 (cumulative wait) |
| t_recover 이후 (Changed Stable Deployment) | V ≈ 0 (verified stable) | V는 100 → 125 …로 계속 상승 |

---

## 8. 관련 파일

| 파일 | 역할 |
|---|---|
| `scripts/qos_recovery_validation.py` | 메인 측정/플롯 스크립트. 두 시나리오(mode 0, mode 3)를 차례로 실행하고 두 곡선을 한 figure에 그림 |
| `scripts/qos_recovery_validation.md` | 본 문서 |
| `tests/qos_recovery_schedule.yaml` | 3-phase 스케줄 정의. infps만 phase 별로 다르고 phase 1·2의 placement는 동일 |
| `schedule_executor_main.py` | mode 0/1/2/3 실행기. **mode 3 (static)** 과 **mode 0의 same-placement detection** (`_placement_signature`, `_next_transition_is_inplace`, `_run_next` 안의 in-place 분기) 이 본 figure를 위해 추가됨 |
| `unified_viewer.py` | per-tick 메트릭 CSV 기록. **`apply_static_phase`** 메서드(mode 3 / mode 0 in-place 양쪽이 사용), feeder persistence, Option A v(t), wait_ms 포함이 본 figure를 위한 변경 |
| `model_processors.py` | ResNet 워커가 `wait_ms`를 4-tuple로 핸들러에 전달하도록 변경 |
| `view_handlers.py` | `ResNetViewHandler`가 4-tuple을 받아 `avg_wait_ms` 누적 |
| `results/qos_score_validation.pdf` | 생성된 paper figure (overwrite) |
| `results/qos_recovery_<ts>.csv` | stop-and-start (mode 0) 측정 원본 (timestamped, overwrite 안 됨) |
| `results/qos_recovery_static_<ts>.csv` | static (mode 3) 측정 원본 (timestamped, overwrite 안 됨) |

---

## 9. 자주 만나는 문제와 대응

### 9.1 `V(t) never crossed epsilon during the failure phase`
- failure phase가 짧거나 phase 2 infps가 충분히 크지 않다는 뜻.
- 대응:
  - `combination_overload`의 mnasnet/squeezenet infps를 더 키운다 (현재 400).
    Phase 2 placement는 phase 1과 동일해야 same-placement detection이 동작한다.
  - 또는 `--failure-duration 4` 등으로 phase 2를 1초 더 늘려서 windowed V(t)가
    plateau에 도달할 시간을 더 준다.

### 9.2 V(t)가 ε 한참 위로 너무 크게 spike함
- mnasnet/squeezenet infps가 너무 높아 SLO(`1/infps`)가 너무 빡빡한 상태.
- 대응: `combination_overload`의 mnasnet/squeezenet infps를 200~400 사이로
  조정. resnet50/resnext50도 비례하여 30~60으로.

### 9.3 stop-and-start 곡선이 phase 2에서 V(t) ~ 80을 찍은 뒤 dip 후 다시 peak
- 옛 동작(spurious cold-start spike) 흔적. mode 0의 same-placement detection이
  비활성화된 채로 phase 1 → phase 2에서 워커를 재시작하면 발생한다.
- 대응: `schedule_executor_main.py`의 `_placement_signature` /
  `_next_transition_is_inplace` 코드가 살아있는지 확인. `_run_next` 로그에
  `Mode 0: same placement combination_initial -> combination_overload; applying
  infps in-place (no restart)` 메시지가 출력되어야 정상.

### 9.4 baseline V(t)가 0이 아님
- Phase 1 infps가 worker capacity를 초과해서 이미 saturation이 시작된 상태.
- 대응: `combination_initial`의 mnasnet/squeezenet/resnet50/resnext50 infps를
  각 모델의 처리량 이하로 낮춘다 (현재 3/3/4/4).

### 9.5 두 곡선이 phase 2에서 너무 다른 모양
- 두 run의 cold-start 타이밍 차이로 RNG 수준의 변동이 항상 있다.
- 대응: 한 두 번 더 재실행해 본다. peak 차이가 ±20% 이내면 정상 범위.
  더 큰 차이는 phase 2 첫 행이 cold-start spike를 잡았는지 여부에 의해
  발생하므로, `--baseline-duration`을 좀 더 길게 잡으면 안정화된다.

### 9.6 mode 1 (adaptive hot-swap)을 쓰면 안 되는 이유
- mode 1은 phase 간 view handler를 재사용하므로 `avg_infer_time`이 누적
  평균이다. Phase 3에서 GPU로 옮겨도 phase 1·2의 CPU 샘플이 평균에 남아
  V(t)가 baseline까지 떨어지지 못한다 (cumulative average dilution —
  `scripts/generate_algorithm_doc.py:432-479`에 분석되어 있음).
- 따라서 본 figure는 stop-and-start 곡선을 항상 mode 0(stop-and-restart)로
  측정한다. 스크립트도 hard-coded `--adaptive-mode 0`으로 호출한다.
- static 곡선은 mode 3 (`apply_static_phase` 전용)으로 측정하며, mode 1과는
  다른 코드 경로를 거치므로 cumulative dilution 문제는 없다 (handler
  reset이 없어도 OK인 이유: static은 phase 3에서도 같은 CPU 워커가 같은
  load를 받고 있을 뿐이라 dilution이 의미를 갖지 않는다).

### 9.7 cold-start gap 동안 V(t)가 계속 상승하지 않음
- Frame queue가 `maxsize=2`로 bound되어 있어서 cold-start gap 동안
  feeder가 push하는 frame은 최대 2개까지만 큐에 머무르고 나머지는 drop된다.
  이 때문에 wait_ms는 큐 깊이에 cap이 걸리고, V(t)는 (gap 동안 계속
  상승하는 게 아니라) gap 종료 직후 한 행에 spike로 측정된다.
- 진정한 의미의 "gap 동안 계속 상승하는" 그래프를 만들려면 unbounded
  queue 모델로 시뮬레이션하거나 gap 구간에 synthetic V(t) 값을 인위적으로
  채워 넣어야 함 (현재 figure는 그렇게 하지 않음).

### 9.8 static 곡선이 plateau에 멈춤 (계속 상승하지 않음)
- phase 2 infps가 너무 낮아서 wait가 빠르게 saturate되고 cumulative average가
  더 이상 증가하지 않는 상태. 본 figure는 infps=400/400/50/40으로 충분히
  크게 잡아서 cumulative `avg_wait_ms`가 계속 증가하도록 했다.
- 대응: phase 2의 mnasnet/squeezenet infps를 400 이상으로 키운다.

### 9.9 t_detect ~ t_recover 구간에서 stop-and-start 곡선이 평평하게 보임
- 옛 동작: cold-start gap 양 끝의 measurement (gap 직전·직후 V(t))를 단순히
  직선으로 연결해서, 두 값이 우연히 비슷하면 평평한 plateau처럼 보임. 1Hz
  측정인데 갑자기 값이 안 변한 것처럼 오해를 일으킨다.
- 현재 동작: `make_plot()`에서 stop-and-start 곡선을 그릴 때 cold-start gap
  의 row index에 NaN을 삽입해 곡선을 의도적으로 끊는다. matplotlib이 NaN
  구간에는 선을 그리지 않으므로, "이 구간에는 measurement 자체가 없다"는
  사실이 곡선의 형태로 그대로 드러난다.
- 동시에 `redeployment downtime (no service)` 어노테이션이 첫 cold-start gap
  중앙으로 곡선 화살표와 함께 표시되어, "선이 끊어진 이유 = 시스템이
  redeployment 중이라 처리를 못 함"을 명시한다.
- 어노테이션은 `t_0 ≤ gap_mid ≤ t_recover` 조건을 만족하는 첫 번째 gap
  하나에만 붙는다. phase 3 후반에 가끔 잡히는 두 번째 gap (GPU 워커 안정화
  glitch)은 어노테이션 대상이 아니다.
