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

- X축: 시간 (초)
- Y축: 윈도우 QoS violation score `V(t)`

  `V(t) = (1/T) · Σ_{τ = t-T+1 .. t} v(τ)`,  `T = 5 s`
  `v(τ) = (1/N) · Σ_i max(0, ℓ_i(τ) / L_SLO,i − 1)`

- 수평선: 검출 임계값 `ε` (논문에서는 30 고정)
- 수직선:
  - `t_0` — failure injection (harmful placement이 시작되는 순간)
  - `t_detect` — 시스템이 sustained degradation을 감지한 순간
  - `t_recover` — 새 placement에서 V(t)가 다시 ε 미만으로 내려가 안정화된 순간
- Shaded region 1 (주황): **detection latency** = `t_detect − t_0`
- Shaded region 2 (초록): **recovery latency** = `t_recover − t_detect`
- 합친 길이가 paper에서 말하는 *degradation duration*

---

## 2. 시나리오 (3 phase)

4개 모델(`mnasnet`, `squeezenet1.0-12`, `resnet50`, `resnext50`)을 동시에
실행하면서 다음 세 단계를 차례로 통과한다. 각 단계는 같은 한 번의 viewer
세션 안에서 mode 0(stop-and-restart) 방식으로 전환되며, 모든 per-tick
메트릭은 단일 CSV로 누적 기록된다.

| Phase | Combo (YAML key) | 디바이스 배치 | infps (요청 입력률) | 의도 |
|---|---|---|---|---|
| **1. Initial Deployment** | `combination_initial` | 4개 모두 CPU | mnasnet 30 / squeezenet 50 / resnet50 3 / resnext50 2 | verified placement. infps가 낮아 SLO 여유가 충분하므로 V(t) ≈ 0 |
| **2. Overload (failure)** | `combination_overload` | 4개 모두 CPU | mnasnet 370 / squeezenet 400 / resnet50 73 / resnext50 58 | input rate가 12–25배 급증. CPU 4-way 동시실행으로는 처리 못 함 → V(t) ≫ ε |
| **3. Changed Stable Deployment** | `combination_offload` | mnasnet/squeezenet은 CPU 유지, resnet50/resnext50을 GPU로 이동 | (Phase 2와 동일한 high infps) | input rate는 그대로지만 무거운 두 모델을 GPU로 옮겨 V(t)를 임계값 아래로 회복 |

> 모델/배치 조합은 XGBoost 예측기를 사용하지 않는다. Phase 3는 "두 무거운
> 모델은 GPU, 두 가벼운 모델은 CPU"라는 도메인 지식으로 미리 정한 verified
> stable configuration이다.

### Phase 길이 (기본값)

| Phase | `--*-duration` 인자 | 측정 행 수 | 비고 |
|---|---|---|---|
| Initial  | `--baseline-duration 12` | ≈ 7 행 | cold-start 직후 ~5초가 필터링되므로 12초 요청에 대해 보통 7–8행이 남음 |
| Overload | `--failure-duration 5`   | ≈ 6 행 | T = 5에 의해 detection이 가능한 최소 길이. 너무 늘리면 detection–rollback 사이 인공적 지연이 길어짐 |
| Offload  | `--recovery-duration 12` | ≈ 12 행 | V(t)가 다시 ε 미만으로 안정화되는 모습을 충분히 보여주기 위함 |

---

## 3. 측정 / 후처리 파이프라인

```
schedule_executor_main.py (mode 0, --combo-duration ...)
        │
        ▼
unified_viewer.py  →  매초마다 v(t)와 phase 라벨을 CSV에 기록
        │            (column: timestamp, combination, v_score, view*_infer_ms, ...)
        ▼
results/qos_recovery_<ts>.csv
        │
        ▼
qos_recovery_validation.py
   1. CSV 로드
   2. 각 행의 v_score(per-tick)를 슬라이딩 윈도우 평균으로 V(t) 계산
   3. phase 경계 자동 탐지 (combination 컬럼의 변화점)
   4. ε 결정 (--epsilon 또는 baseline·stressed의 중점)
   5. t_0 / t_detect / t_recover 계산
   6. matplotlib으로 PDF 렌더링
        │
        ▼
results/qos_score_validation.pdf
```

`unified_viewer.py`의 `cpu_timer`가 1 Hz로 동작하므로 CSV 한 행 = 1초.
즉 행 인덱스를 그대로 X축의 초 단위로 사용한다.

---

## 4. Detection / Recovery 판정 규칙

### 4.1 Detection — "fully populated post-failure window" 규칙

스크립트는 `t_detect`를 다음 조건을 모두 만족하는 가장 빠른 행으로 정의한다.

1. `i ≥ p2_start + T − 1`  (V(t) 윈도우가 post-failure 샘플로 완전히 채워짐)
2. `V(i) > ε`

여기서 `p2_start`는 `combination_overload`의 첫 번째 CSV 행 인덱스이고
`T = 5`. 조건 1이 핵심이다 — 이 조건이 없으면 V(t)가 슬라이딩 평균이라
window의 일부만 새 데이터로 채워져도 평균이 ε를 넘으면 즉시 detect되어
detection latency가 T보다 훨씬 짧아진다 (severe failure에서는 2–3초).

`t_0`는 phase 경계 직전 시점, 즉 `t_0 = p2_start − 1`로 표시한다 (failure가
"마지막 verified 측정 직후 / 첫 번째 overload 측정 직전"에 주입되었다는
의미). 이 두 컨벤션이 결합되어, severe failure에서 **detection latency가
정확히 T = 5초**가 보장된다.

### 4.2 Recovery — V(t)가 ε 까지 내려온 시점

`t_recover`는 phase 3 시작 이후 `V(t) <= ε` 가 처음 성립하는 행이다. 즉
V(t) 곡선이 임계값 ε 까지 내려오는 그 순간을 곧 회복 완료로 본다.

V(t)는 그 자체가 T = 5초 슬라이딩 평균이라 이미 충분히 smoothed 되어
있으므로 추가적인 디바운싱(예: 2 tick 연속 조건)은 필요하지 않다.
Detection처럼 "윈도우 완전 채움" 같은 보수적 대기도 적용하지 않는다 —
시스템은 이미 verified placement(Phase 3)에서 동작 중이고, 사용자 정의에
따라 V(t)가 ε에 닿는 순간이 곧 안정화이기 때문이다.

### 4.3 표시되는 latency

- **Detection latency** = `t_detect − t_0` (≥ T = 5초가 보장됨)
- **Recovery latency**  = `t_recover − t_detect`
  = (rollback 행동 지연) + (실제 V(t) 하강 dynamics)
  - rollback 행동 지연 = `p3_start − t_detect`. failure phase 길이를 짧게
    잡을수록 작아진다 (기본 5초 설정에서 1–2초).
- **Total degradation duration** = `t_recover − t_0`

---

## 5. ε (검출 임계값) 결정

기본 모드에서는 한 번 실행하면서 측정한 값을 토대로 자동 결정한다:

```
baseline = V(p2_start − 1)            # phase 1 마지막 tick의 V(t)
stressed = V(p3_start − 1)            # phase 2 마지막 tick의 V(t)
ε        = (baseline + stressed) / 2  # 단순 중점
```

`--epsilon N`으로 명시 지정 시 그 값을 그대로 사용한다. 논문에서는 ε = 30
으로 고정하므로 다음과 같이 호출한다:

```bash
python scripts/qos_recovery_validation.py --epsilon 30
```

콘솔에 측정된 baseline·stressed·midpoint·실제 사용한 ε이 모두 출력된다.

---

## 6. 실행 방법

### 6.1 처음부터 끝까지 (측정 + 그래프)

```bash
cd /home/msyu/PycharmProjects/fsrr-multimodel-scheduling
python scripts/qos_recovery_validation.py --epsilon 30
```

총 소요 시간 ≈ 50초 (4개 모델 ONNX 세션 로드 시간 + 3 phase 측정 시간).
GPU와 CUDA 12 onnxruntime이 정상 동작하는 환경이 필요하다.

### 6.2 측정 데이터 재사용 (그래프만 다시)

CSV는 그대로 두고 시각화 옵션만 바꿔보고 싶을 때:

```bash
python scripts/qos_recovery_validation.py --no-run --epsilon 30
```

`--no-run`을 주면 `results/qos_recovery_*.csv` 중 가장 최신 파일을 자동으로
다시 읽어 PDF만 재렌더링한다. `--csv <path>`로 특정 파일을 지정할 수도 있다.

### 6.3 Phase 길이 조정

```bash
# detection–rollback 사이 간격을 더 줄이고 싶다면 (↓ 단, 최소 5초 필요)
python scripts/qos_recovery_validation.py --epsilon 30 --failure-duration 5

# initial / recovery phase를 더 길게 보고 싶다면
python scripts/qos_recovery_validation.py --epsilon 30 \
    --baseline-duration 15 --recovery-duration 15
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

콘솔 출력 예시 (ε = 30, 기본 phase 길이):

```
[Info] Phase boundaries: [(0, 'combination_initial'), (7, 'combination_overload'), (12, 'combination_offload')]

============================================================
  Detection threshold determination
============================================================
  Baseline V(t) (end of combination_initial   , t=6s) = 4.5170
  Stressed V(t) (end of combination_overload  , t=11s) = 41.7074
  Empirical midpoint                                  = 23.1122
  ==> Detection threshold  epsilon = 30.0000  (user override (--epsilon 30))

============================================================
  Recovery scenario timing
============================================================
  t_0      (failure injection)    = 6s  (just before combination_overload starts)
  t_detect (windowed V(t) > eps)  = 11s
  rollback (phase 3 starts)       = 12s  (combination_offload starts)
  t_recover (V(t) <= eps)         = 15s
  Detection latency               = 5s   (>= T = 5)
  Recovery latency                = 4s
  Total degradation duration      = 9s
```

읽는 법:

| 표시 | 의미 |
|---|---|
| `Phase boundaries` | CSV에서 자동 탐지된 각 phase의 시작 행 인덱스. 이 값들이 t_0/rollback의 기준이 된다 |
| `Baseline V(t)` | Initial phase 마지막 tick의 windowed V(t). Phase 1이 stable함을 검증하는 지표 |
| `Stressed V(t)` | Overload phase 마지막 tick의 windowed V(t). Phase 2가 ε보다 충분히 큰지 검증 |
| `Empirical midpoint` | baseline·stressed의 중점. `--epsilon`을 안 주면 이 값이 자동으로 사용됨 |
| `Detection latency ≥ T` | "fully populated window" 규칙에 의해 항상 T 이상이어야 한다는 표시 |

논문에 들어갈 핵심 수치(`ε`, `detection latency`, `recovery latency`,
`total degradation duration`)는 모두 위 출력에서 그대로 가져다 쓸 수 있다.

---

## 8. 관련 파일

| 파일 | 역할 |
|---|---|
| `scripts/qos_recovery_validation.py` | 메인 측정/플롯 스크립트 |
| `scripts/qos_recovery_validation.md` | 본 문서 |
| `tests/qos_recovery_schedule.yaml` | 3-phase 스케줄 정의 |
| `schedule_executor_main.py` | mode 0/1/2 실행기. `--combo-duration` 옵션이 본 시나리오를 위해 추가됨 |
| `unified_viewer.py` | per-tick 메트릭 CSV 기록 (column `v_score` = 인스턴트 v(t)) |
| `results/qos_score_validation.pdf` | 생성된 paper figure (overwrite) |
| `results/qos_recovery_<ts>.csv` | 측정 원본 (timestamped, overwrite 안 됨) |

---

## 9. 자주 만나는 문제와 대응

### 9.1 `V(t) never crossed epsilon during the failure phase`
- failure phase가 T = 5초보다 짧거나 v(t)가 ε보다 충분히 크지 않다는 뜻.
- 대응: `--failure-duration 6` 으로 늘리거나, `combination_overload`의
  `infps`를 더 키워 v(t)를 더 강하게 만든다.

### 9.2 `V(t) did not return below epsilon by end of recovery phase`
- recovery phase 길이가 V(t) 회복에 비해 짧다.
- 대응: `--recovery-duration 15` 등으로 늘린다.

### 9.3 baseline V(t)가 0이 아닌 큰 값
- Phase 1에서 이미 SLO 위반이 발생 중. infps가 너무 높거나, CPU 4-way
  동시 실행이 이미 contention을 일으키는 상황.
- 대응: `combination_initial`의 infps를 더 낮춘다 (특히 `mnasnet`,
  `squeezenet1.0-12` 쪽).

### 9.4 mode 1 (adaptive hot-swap)을 쓰면 안 되는 이유
- mode 1은 phase 간 view handler를 재사용하므로 `avg_infer_time`이 누적
  평균이다. Phase 3에서 GPU로 옮겨도 phase 1·2의 CPU 샘플이 평균에 남아
  V(t)가 baseline까지 떨어지지 못한다 (cumulative average dilution).
- 따라서 본 figure는 항상 mode 0(stop-and-restart)으로 측정한다.
  스크립트도 hard-coded `--adaptive-mode 0`으로 호출한다.
