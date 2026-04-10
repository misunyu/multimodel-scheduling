# `qos_score_validation.pdf` 생성 시나리오

본 문서는 논문의 *Failure Impact Objective* 절에 들어가는 figure
`qos_score_validation.pdf`를 재현하기 위한 실험 시나리오, 측정 방식, 결과
해석 방법을 정리한다.

생성 스크립트: [`scripts/qos_recovery_validation.py`](./qos_recovery_validation.py)
스케줄 정의 : [`tests/qos_recovery_schedule.yaml`](../tests/qos_recovery_schedule.yaml)
출력 PDF    : `results/qos_score_validation.pdf`
원본 CSV    : `results/qos_recovery_<timestamp>.csv`

---

## 1. Figure가 보여주는 것

- **X축**: wall-clock 시간 (초). 0초 = 첫 측정 행의 timestamp.
  Phase 사이의 cold-start gap이 회색 hatched 영역으로 그대로 보이도록
  row index가 아니라 wall-clock 초 단위를 사용한다.
- **Y축**: 윈도우 QoS violation score `V(t)`

  ```
  V(t) = (1/T) · Σ_{τ=t-T+1..t} v(τ),    T = 3 s (sliding window)
  v(τ) = (1/N_active) · Σ_{i: ℓ_i(τ)>0} max(0, ℓ_i(τ) / L_SLO,i − 1)
  ℓ_i(τ) = handler.avg_infer_time + handler.avg_wait_ms   (end-to-end)
  ```

  - `N_active`는 해당 tick에 적어도 한 번 inference를 끝낸 view의 수.
    아직 측정값이 없는 view는 평균에서 제외한다 ("Option A" 규칙).
    이렇게 해야 cold-start 직후 1–2 view만 동작 중인 순간에 v(t)가 0으로
    diluted되지 않는다.
  - `ℓ_i`는 inference 시간 + queue wait 시간 (end-to-end response time).
    queue wait를 포함해야 cold-start backlog가 V(t)에 반영된다.
- **수평선**: 검출 임계값 `ε`. 논문에서는 **ε = 50 으로 고정**해서 사용한다.
- **수직선** (3 개):
  - `t_0` — failure injection 순간 (= `t_detect − T`로 역산)
  - `t_detect` — windowed V(t)가 처음 ε를 초과한 순간
  - `t_recover` — V(t)가 다시 ε 이하로 내려간 순간
- **연속 색조 영역**:
  - 옅은 주황 = **Detection phase** (`t_0 → t_detect`, 항상 `T = 3 s`)
  - 옅은 초록 = **Recovery phase** (`t_detect → t_recover`)
  - 회색 hatched (//) = **Rollback in progress (cold start)** —
    워커가 stop된 후 새 워커가 ready 될 때까지의 wall-clock 공백 구간.
    이 구간은 CSV에 측정 행이 없으며, V(t) 곡선은 양 끝 데이터 점을
    하나의 직선으로 연결한다.
- **V(t) 곡선**: 진한 남색 단일 연속선. legend는 그래프에 표시하지 않으며
  (요청에 따라 생략), 각 phase는 X축 아래 라벨로 식별된다.

---

## 2. 시나리오 (3 phase)

4개 모델(`mnasnet`, `squeezenet1.0-12`, `resnet50`, `resnext50`)을 동시에
실행하면서 다음 세 단계를 차례로 통과한다. 같은 한 번의 viewer 세션 안에서
mode 0(stop-and-restart)로 phase를 전환하며, 모든 per-tick 메트릭은 단일
CSV로 누적 기록된다.

| Phase | YAML key | 디바이스 배치 | infps | 의도 |
|---|---|---|---|---|
| **1. Initial Deployment** | `combination_initial` | 4개 모두 CPU | mnasnet 40 / squeezenet 40 / resnet50 5 / resnext50 4 | verified placement. infps가 낮아 SLO 여유가 충분하므로 V(t) ≈ 12 (ε=50 한참 아래) |
| **2. Overload (failure)** | `combination_overload` | 4개 모두 CPU | mnasnet 100 / squeezenet 100 / resnet50 15 / resnext50 12 | input rate가 ~2.5–3배 증가. 4개 모델이 동시에 CPU를 점유하면서 큐가 쌓이고 wait_ms가 누적되어 V(t) > ε |
| **3. Changed Stable Deployment** | `combination_offload` | 4개 모두 GPU | (Phase 2와 동일한 high infps) | input rate는 그대로지만 4개 모델 모두 GPU로 옮겨 빠르게 처리. V(t)는 ε 아래로 회복 |

> **Phase 3가 "all GPU" 인 이유**
> 처음에는 사용자 요청대로 "무거운 두 모델만 GPU, 가벼운 두 모델은 CPU 유지"로
> 시도했지만, GPU 워커들의 pre/postprocess가 여전히 CPU를 점유해서
> mnasnet/squeezenet이 충분히 빨라지지 못했다 (V(t) ≈ 37에 plateau). 4개 모두
> GPU로 옮기는 것이 가장 깨끗하게 ε 아래로 회복되는 verified stable
> configuration이라 이 형태로 정착했다. (자세한 코멘트는 YAML 파일 참고)

### 2.1 infps 값을 어떻게 정했나

V(t)가 너무 극단적이 되지 않도록 (ε 대비 1.5–2배 정도 peak), 그리고 phase 2
구간에서 ε 이하로는 떨어지지 않도록 조정한 결과:

- **mnasnet/squeezenet** (가벼운 분류기, CPU 처리량 ~5 fps):
  Initial 40 → Overload 100. SLO가 25 ms → 10 ms로 줄면서 동시에
  4-way contention이 시작되어 wait_ms 누적.
- **resnet50/resnext50** (무거운 분류기, CPU 처리량 ~3 fps):
  Initial 5/4 → Overload 15/12. Phase 1에서 workers가 capacity 안에
  있도록 충분히 낮춘 값 (V(t)≈12 stable).

이 조합으로 phase 2에서 windowed V(t)가:

- 첫 phase 2 행: 49.92 (≈ ε)
- 점진 상승 → 73.86 → 93.37
- rollback gap 직후 첫 phase 3 행: 97.78 (← peak)
- 빠르게 0으로 회복

라는 단봉(single-peak) 형태가 된다.

### 2.2 Phase 길이 (기본값)

| Phase | 인자 (default) | 측정 행 수 | 비고 |
|---|---|---|---|
| Initial  | `--baseline-duration 12` | ≈ 7–8 행 | cold-start 직후 일부가 필터링됨 |
| Overload | `--failure-duration 3`   | ≈ 3–4 행 | T = 3 windowed V(t)가 phase 2에서 한 차례 peak를 찍을 만큼만 길게. 더 길게 잡으면 cumulative averaging이 V(t)를 ε 아래로 끌어내려 "두 봉우리" 형태가 됨 |
| Offload  | `--recovery-duration 12` | ≈ 12 행 | V(t)가 ε 이하로 안정화되는 모습을 충분히 보여주기 위함 |

---

## 3. 측정 / 후처리 파이프라인

```
schedule_executor_main.py (mode 0, --combo-duration ...)
        │
        ▼
unified_viewer.py
   - cpu_timer 1Hz tick마다 v(t) 계산 (Option A: active view만 평균)
   - v(t)는 end-to-end latency (avg_infer_time + avg_wait_ms) 기반
   - feeder/입력 큐는 phase 간에 persistent (mode 0 cold-start gap 동안에도
     계속 frame을 push해서 큐에 backlog 쌓임 → 새 워커가 wait_ms spike 측정)
   - CSV column: timestamp, combination, v_score, view*_infer_ms, ...
        │
        ▼
results/qos_recovery_<ts>.csv
        │
        ▼
qos_recovery_validation.py
   1. CSV 로드 + timestamp 파싱
   2. 각 행의 v_score(per-tick)를 T=3 슬라이딩 윈도우 평균으로 V(t) 계산
   3. phase 경계 자동 탐지 (combination 컬럼의 변화점)
   4. wall-clock gap 탐지 (1.5초 이상 간격 = cold-start 구간)
   5. ε 결정 (--epsilon 또는 baseline·stressed의 중점)
   6. detection earliest = p2_start + T − 1 행에서 V(t) > ε 첫 검출
   7. t_0 = t_detect_sec − T (Detection phase = T 보장)
   8. t_recover = phase 3 시작 후 V(t) ≤ ε 첫 행
   9. matplotlib으로 wall-clock 축 PDF 렌더링
        │
        ▼
results/qos_score_validation.pdf
```

`unified_viewer.py`의 `cpu_timer`는 1 Hz로 동작한다. CSV 한 행 ≈ 1초 측정.
다만 mode 0의 phase 전환 시점에는 워커가 stop→load→start되는 ~3–5초의
wall-clock gap이 발생하는데, 이 동안 FPS=0이라 CSV에 행이 안 찍힌다.
스크립트는 이 gap을 timestamp 차이로 자동 탐지해서 figure에 회색 hatched
영역으로 표시한다.

### 3.1 핵심 architectural 변경 (이 figure를 위한)

다음 두 가지가 mode 0 stop-and-restart 흐름에서 본 시나리오를 위해
명시적으로 구현되어 있다 — 일반적인 mode 0 동작과 다르므로 주의:

1. **Persistent input feeders/queues** (`unified_viewer.py`):
   - `feeder_shutdown_flag` 라는 별도 Event를 둬서 feeder는 viewer 전체
     수명 동안 살아있음. `stop_execution()`은 워커만 죽이고 feeder/큐는
     건드리지 않음. 결과적으로 cold-start gap 동안에도 input request가
     계속 큐로 들어가고, 새 워커가 처리할 때 wait_ms가 폭등.
   - 자세한 코드 위치: `unified_viewer.py:initialize_state_variables`,
     `start_execution`, `initialize_threads`, `_drain_and_close_all_queues`.

2. **`avg_wait_ms` 가 v(t) 계산에 포함됨**:
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

### 4.1 Detection — "fully populated post-failure window" 규칙 + T-앵커링

스크립트는 **두 단계**로 detection 시점을 결정한다.

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
wall-clock 길이에 영향받지 않고 항상 정확히 `T = 3 s`가 된다. t_0 자체는
"failure가 마지막 verified 측정 직후의 어느 순간에 주입되었다"는 의미로
해석되며, wall-clock 상으로 cold-start gap 안쪽에 위치할 수도 있다.

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

기본 모드에서는 한 번 실행하면서 측정한 값을 토대로 자동 결정한다:

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

### 6.1 처음부터 끝까지 (측정 + 그래프)

```bash
cd /home/msyu/PycharmProjects/fsrr-multimodel-scheduling
python scripts/qos_recovery_validation.py --epsilon 50
```

총 소요 시간 ≈ 50초 (4개 모델 ONNX 세션 로드 시간 + 3 phase 측정 시간).
GPU와 CUDA 12 onnxruntime이 정상 동작하는 환경이 필요하다.

### 6.2 측정 데이터 재사용 (그래프만 다시)

CSV는 그대로 두고 시각화 옵션만 바꿔보고 싶을 때:

```bash
python scripts/qos_recovery_validation.py --no-run --epsilon 50
```

`--no-run`을 주면 `results/qos_recovery_*.csv` 중 가장 최신 파일을 자동으로
다시 읽어 PDF만 재렌더링한다. `--csv <path>`로 특정 파일을 지정할 수도 있다.

### 6.3 Phase 길이 / 임계값 조정

```bash
# failure phase를 1초 더 늘려서 V(t)가 더 stable plateau를 보이게
python scripts/qos_recovery_validation.py --epsilon 50 --failure-duration 4

# initial / recovery phase를 더 길게
python scripts/qos_recovery_validation.py --epsilon 50 \
    --baseline-duration 18 --recovery-duration 18

# epsilon만 다른 값으로 가시화
python scripts/qos_recovery_validation.py --no-run --epsilon 30
```

내부적으로는 `schedule_executor_main.py`의 `--combo-duration COMBO=SECONDS`
옵션으로 단일 subprocess 안에서 phase별 duration을 다르게 적용한다.

### 6.4 다른 스케줄로 실행

```bash
python scripts/qos_recovery_validation.py --schedule path/to/other.yaml
```

YAML에는 정확히 3개의 combo가 있어야 하며, 순서대로 (initial, overload,
offload)로 해석된다. PDF 라벨은 스크립트 상단의 `PHASE_DISPLAY` 매핑으로
결정되므로 새 combo 이름을 추가하려면 그 매핑도 함께 갱신해야 한다.

---

## 7. 출력물 해석

콘솔 출력 예시 (ε = 50, 기본 phase 길이):

```
[Info] Phase boundaries: [(0, 'combination_initial'), (8, 'combination_overload'), (11, 'combination_offload')]

============================================================
  Detection threshold determination
============================================================
  Baseline V(t) (end of combination_initial   , t=7s) = 13.6722
  Stressed V(t) (end of combination_overload  , t=10s) = 93.3735
  Empirical midpoint                                  = 53.5228
  ==> Detection threshold  epsilon = 50.0000  (user override (--epsilon 50))

============================================================
  Recovery scenario timing (wall-clock seconds from row 0)
============================================================
  t_0      (failure injection)    =  10.00s  (= t_detect - T)
  t_detect (windowed V(t) > eps)  =  13.00s  (row 10, window fully post-failure)
  rollback (new workers ready)    =  15.00s  (first combination_offload row)
  t_recover (V(t) <= eps)         =  17.00s  (row 13)
  Detection phase                 =   3.00s   (= T)
  Recovery phase                  =   4.00s
  Total degradation duration      =   7.00s
  Cold-start gaps detected        = 4.00s @ 7.0-11.0s, 2.00s @ 13.0-15.0s
```

읽는 법:

| 표시 | 의미 |
|---|---|
| `Phase boundaries` | CSV에서 자동 탐지된 각 phase의 시작 행 인덱스 |
| `Baseline V(t)` | Phase 1 마지막 tick의 windowed V(t). Phase 1이 stable함을 검증 |
| `Stressed V(t)` | Phase 2의 windowed V(t) 최대값 |
| `Empirical midpoint` | baseline과 stressed의 중점. `--epsilon`을 안 주면 이 값을 자동 사용 |
| `t_0` | failure injection 시점 (= `t_detect − T`로 역산) |
| `t_detect` | "fully populated post-failure window" 규칙으로 V(t) > ε 첫 검출 시점 |
| `rollback (new workers ready)` | phase 3 첫 측정 행의 wall-clock 시간 (= cold-start gap 종료 시점) |
| `t_recover` | V(t) ≤ ε 첫 성립 시점 |
| `Detection phase` | t_0 → t_detect 폭, 항상 `T = 3 s` |
| `Recovery phase` | t_detect → t_recover 폭, 시나리오에 따라 변동 |
| `Cold-start gaps detected` | CSV의 timestamp 간격이 1.5초 이상인 구간들. wall-clock 상의 cold-start 시간을 보여줌 |

논문에 들어갈 핵심 수치(`ε`, `Detection phase`, `Recovery phase`,
`Total degradation duration`)는 모두 위 출력에서 그대로 가져다 쓸 수 있다.

---

## 8. 관련 파일

| 파일 | 역할 |
|---|---|
| `scripts/qos_recovery_validation.py` | 메인 측정/플롯 스크립트 |
| `scripts/qos_recovery_validation.md` | 본 문서 |
| `tests/qos_recovery_schedule.yaml` | 3-phase 스케줄 정의 |
| `schedule_executor_main.py` | mode 0/1/2 실행기. `--combo-duration` 옵션이 본 시나리오를 위해 추가됨 |
| `unified_viewer.py` | per-tick 메트릭 CSV 기록. feeder persistence + Option A v(t) + wait_ms 포함 변경 적용 |
| `model_processors.py` | ResNet 워커가 `wait_ms`를 4-tuple로 핸들러에 전달하도록 변경 |
| `view_handlers.py` | `ResNetViewHandler`가 4-tuple을 받아 `avg_wait_ms` 누적 |
| `results/qos_score_validation.pdf` | 생성된 paper figure (overwrite) |
| `results/qos_recovery_<ts>.csv` | 측정 원본 (timestamped, overwrite 안 됨) |

---

## 9. 자주 만나는 문제와 대응

### 9.1 `V(t) never crossed epsilon during the failure phase`
- failure phase가 짧거나 v(t)가 ε보다 충분히 크지 않다는 뜻.
- 대응:
  - `--failure-duration 4` 등으로 늘려서 phase 2에서 V(t)가 ε를 넘을
    기회를 더 주거나
  - `combination_overload`의 infps를 (mnasnet/squeezenet 위주로) 키워
    cold-start spike와 정상 v(t)를 모두 끌어올린다.

### 9.2 V(t)가 ε 한참 위로 너무 크게 spike함
- 현재 mnasnet/squeezenet의 infps가 너무 높음. SLO가 너무 빡빡해서
  small wait도 큰 v(t)가 됨.
- 대응: `combination_overload`의 mnasnet/squeezenet infps를 70–100 사이로
  조정. resnet50/resnext50은 비례하여 10–20.

### 9.3 V(t)가 phase 2 중간에 ε 아래로 떨어짐 (두 봉우리 형태)
- failure phase가 너무 길어서 cumulative `avg_wait_ms`가 cold-start 직후
  spike에서 정상 wait로 수렴해 가는 동안 V(t)가 점점 낮아짐. Window가
  cold-start 행을 빠져나갈 때 V(t)가 phase 2 중간에서 더 떨어졌다가
  rollback gap의 cold-start spike로 다시 한 번 올라가 두 봉우리가 보임.
- 대응: `--failure-duration 3` (기본값) 또는 더 짧게. cumulative 평균이
  너무 많이 dilute되기 전에 phase 2를 끝낸다.

### 9.4 baseline V(t)가 ε에 가까움
- Phase 1에서 이미 SLO 위반이 발생 중. infps가 너무 높거나, CPU 4-way
  동시 실행이 이미 contention을 일으키는 상황.
- 대응: `combination_initial`의 infps를 더 낮춘다 (특히 `mnasnet`,
  `squeezenet1.0-12` 쪽). 현재 40/40/5/4 정도가 V(t)≈12 baseline을 줌.

### 9.5 mode 1 (adaptive hot-swap)을 쓰면 안 되는 이유
- mode 1은 phase 간 view handler를 재사용하므로 `avg_infer_time`이 누적
  평균이다. Phase 3에서 GPU로 옮겨도 phase 1·2의 CPU 샘플이 평균에 남아
  V(t)가 baseline까지 떨어지지 못한다 (cumulative average dilution —
  `scripts/generate_algorithm_doc.py:432-479`에 분석되어 있음).
- 따라서 본 figure는 항상 mode 0(stop-and-restart)으로 측정한다.
  스크립트도 hard-coded `--adaptive-mode 0`으로 호출한다.

### 9.6 그래프에 두 개의 peak가 보임
- mode 0의 phase 전환마다 cold-start spike가 한 번씩 발생하는데, failure
  phase가 너무 길면 phase 2 cold-start spike와 phase 3 cold-start spike가
  windowed V(t) 상에서 분리된 두 봉우리로 나타난다.
- 대응: failure phase를 3초 정도로 짧게 잡으면 phase 2 plateau가 phase 3
  cold-start spike와 자연스럽게 이어져서 단봉(single-peak) 형태가 됨.

### 9.7 cold-start gap 동안 V(t)가 계속 상승하지 않음
- Frame queue가 `maxsize=2`로 bound되어 있어서 cold-start gap 동안
  feeder가 push하는 frame은 최대 2개까지만 큐에 머무르고 나머지는 drop된다.
  이 때문에 wait_ms는 큐 깊이에 cap이 걸리고, V(t)는 (gap 동안 계속
  상승하는 게 아니라) gap 종료 직후 한 행에 spike로 측정된다.
- 진정한 의미의 "gap 동안 계속 상승하는" 그래프를 만들려면 unbounded
  queue 모델로 시뮬레이션하거나 gap 구간에 synthetic V(t) 값을 인위적으로
  채워 넣어야 함 (현재 figure는 그렇게 하지 않음).
