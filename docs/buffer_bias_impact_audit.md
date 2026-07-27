# hot-swap 큐 편향 — 오염 범위 전수 감사

날짜: 2026-07-24 · 감사 전용(코드 변경·재실행 없음) · main.tex 불변
대상 결함: [hotswap_buffer_bias_fix.md](hotswap_buffer_bias_fix.md)

> **결론 요약**
> 1. 편향은 **`adaptive_deploy` 경로(mode 1 Adaptive, mode 1+validate BoundGuard, mode 2 reactive)에만** 적용된다.
>    mode 0(Stop-restart)·3(Static)·4(restart+rollback)는 이 경로를 타지 않는다. → **"적응 ≫ 비적응" 대비가
>    구조적으로 부풀려질 수 있는 조건이 성립**한다.
> 2. **B2(논문 `tab:b2`·`fig:b2`·Q1.5)는 오염 확정 — 무효.** B=754/569 + mode 1 hot-swap 조합.
> 3. **C3(논문 `fig:fluid_validation`·Q1.4)는 무영향 — 유효.** 전 실행이 mode 0이며 hot-swap 0회.
> 4. **§4b는 유효.** 버퍼가 기본 2라 교체 큐(2)와 동일.
> 5. **감사 중 두 번째 편향 발견 — 큐 깊이 수정으로 고쳐지지 않는다.** `_hot_swap_view`가 교체 시
>    **상속 backlog를 명시적으로 폐기**하는데, mode 0 경로는 **의도적으로 보존**한다(코드 주석에 명시).
>    B2 오염의 주 성분은 이쪽으로 보인다. → §5.
>
> **[2026-07-24 후속]** §5는 수정 완료되고 B2가 재실행됐다 →
> [backlog_preserve_report.md](backlog_preserve_report.md). 결과: 예상대로 B2의 miss rate 격차가
> **완전히 소멸**(Adaptive 0.679 → 0.964)했고, `tab:b2`·`fig:b2`·Q1.5 서술의 재작성이 필요하다.
> 감사 중 **세 번째 문제**(B2 하네스의 시간 불일치 — BoundGuard에만 18초 추가)도 그 과정에서 드러나
> 시간대응본으로 교정했다.

---

## §1. 편향의 정확한 적용 범위

### 1.1 결함 지점 (수정 전 `backup/adaptive_deploy.pre_bufferfix.py`)
```
L255  new_frame_q = Queue(maxsize=2)                    # 교체 워커 (device 변경)
L323  frame_q     = Queue(maxsize=2)                    # 신규 사용 뷰
L377  v.headless_frame_queues[hid] = Queue(maxsize=2)   # headless
```
`unified_viewer.py`는 같은 큐를 `FSRR_FRAME_BUFFER`(`_fb`)로 생성하므로, **두 경로가 서로 다른 깊이**를 썼다.

### 1.2 기법별 경로 (전수 확인)

`unified_viewer.py:1155-1173` 디스패치:

| 기법 | mode | 전환 경로 | `adaptive_deploy` 경유 | **깊이 편향** |
|---|---|---|---|---|
| Static | 3 | else → `stop_execution` + `initialize_model_settings` | ✗ | **없음** |
| Stop-restart | 0 | 동일 (else) | ✗ | **없음** |
| restart+rollback | 4 | `restart_rollback_deploy` → `stop_execution`(L255) + `initialize_model_settings`(L262) | ✗ | **없음** |
| **Adaptive** | 1 | `AdaptiveDeployManager.execute()` | **✓** | **있음** |
| **BoundGuard** | 1+validate | 동일 | **✓** | **있음** |
| **reactive** | 2 | `ReactiveDeployManager` → Phase 1에서 `AdaptiveDeployManager.execute()` 위임 (`reactive_deploy.py:231`) | **✓** | **있음** |

else 경로가 안전한 이유: `initialize_processes()`가 `if not hasattr(...) or ... is None` 가드로 큐를
**한 번만 생성**하고 이후 재사용한다(`unified_viewer.py:586,592,598,604`). `stop_execution()`은 입력
프레임 큐를 건드리지 않는다(§5.1 참조). 즉 mode 0/3/4는 실행 내내 `_fb` 깊이를 유지한다.

### 1.3 뷰 단위 — 전체가 아니라 **교체된 뷰만**
`adaptive_deploy.py:196-203`: `_same_device()`면 `kept_views`에 넣고 **워커·큐를 그대로 둔다**.
device가 바뀐 뷰만 `_hot_swap_view()` → 새 큐. 신규 사용 뷰는 `_start_fresh_view()` → 새 큐.
→ 같은 실행 안에서 **뷰마다 큐 깊이가 달라진다**(예: Q4 pre-fix에서 kept 뷰 300, swapped 뷰 2).

### 1.4 핵심 판정
편향의 수혜자는 **정확히 hot-swap하는 기법**이다. B2가 주장하는 대비가 바로
`{Adaptive, BoundGuard}` vs `{Static, Stop-restart}` 이므로, **B2의 대비는 편향과 완전히 정렬된다.**

### 1.5 부수 발견 — headless 경로 불일치 (신규, 경미)
`unified_viewer.py:643`은 아직 `Queue(maxsize=2)` 하드코딩이다. 수정 전에는 양쪽 다 2라 일관됐으나,
수정 후 `adaptive_deploy.py:394`만 `_frame_buffer()`를 쓰므로 **headless 뷰의 깊이가 경로에 따라 달라진다**.
현재 실험에서 headless는 LLM background(핸들러 뷰 아님, `V(t)` 미집계)뿐이라 **결과 영향 없음**이나,
정합을 위해 `unified_viewer.py:643`도 `_fb`로 맞추는 것이 옳다(별건).

---

## §2. 실험별 오염 판정 (전수)

판정 기준: **buffer ≠ 2 AND hot-swap 발생 AND 수정 전 측정 → 무효.**

전수 census 방법: 모든 실행 로그에서 `hot-swapping` 발생 횟수를 세고, 짝이 되는 메트릭 CSV의
`viewN_q` 최대값으로 실제 버퍼 깊이를 확인했다.

| 실험 | 산출물 | buffer | hot-swap (기법) | 측정 시점 | **판정** |
|---|---|---|---|---|---|
| **B2** | 논문 `tab:b2`·`fig:b2`, `c3/b2_stay/` | **754(GPU)/569(NPU)** | **1회, Adaptive·BoundGuard만** (Static·StopRestart 0회) | 수정 전 | **❌ 무효** |
| B2 (구 run) | `c3/b2/`, `c3/b2_old_010921/` | 754/569 | 동일 | 수정 전 | ❌ 무효 (미사용) |
| **C3 fluid** | 논문 `fig:fluid_validation`, `c3/slope_*`,`vf_*`,`mueff_*`,`slope_rc_*` | 4000 / 8000 | **0회** (`--adaptive-mode 0` 고정) | — | **✅ 유효** |
| **§4b 관문** | `s4b_*_out/`, `bg/` | **기본 2** (env 미설정, `view*_q` 실측 max=2) | 3–8회 | — | **✅ 유효** (교체 큐 2 = 원큐 2) |
| phaseA (초기 4기법) | `phaseA/`, `docs/phaseA_report.md` | **기본 2** (`run_phaseA.sh`에 env 없음) | 2–4회 | — | **✅ 유효** |
| **Q3** | `q3_misprediction.pdf`, `q3_bf_out/` | 12 | 7회, BoundGuard만 | **수정 후** | **✅ 유효** |
| Q3 (구 run 전부) | `q3_out`,`q3_ord_out`,`q3_trend_out`,`q3_fix_out`,`q3_dw_out`,`q3_long_out`,`q3_final_out`,`q3_sg_out` | 12 | 7–8회 | 수정 전 | ❌ 무효 (이미 대체됨) |
| Q3-paperset | `q3v4_out`,`q3v4_trend_out` | 30 | 6회 | 수정 전 | ❌ 무효 (별도 사유로도 폐기 — `hotswap_buffer_bias_fix.md` §6c) |
| **Q4** | `q4_bounded_envelope.pdf`, `q4_bf_out/` | 300 | 7회, BoundGuard만 | **수정 후** | **✅ 유효** |
| Q4 (구 run) | `q4_out`,`q4_sg_out`,`q4_final_out` | 300/30 | 1–7회 | 수정 전 | ❌ 무효 (이미 대체됨) |
| Table I / 간섭 측정 | `docs/benchmark_model_table.*`, `latency_decomposition.md` | — | **해당 없음** (단일 배치 격리 측정, 전환 없음) | — | **✅ 유효** |
| 예측기 오프라인 감사 | `predictor_signal_audit.md`, `candidate_ranking_check.md` | — | 런타임 미실행 (오프라인 featurization) | — | **✅ 유효** |
| mixed 실행 검증 | `mixed_exec_report.md` | 기본 2 | LLM background 전환(큐 무관) | — | **✅ 유효** |

---

## §3. C3 (논문 Q1.4) — 유효, 근거

C3의 fluid 검증은 backlog 궤적이 근거이므로 큐 깊이 오염 시 **근거 자체가 무너진다**. 확인 결과:

- `c3/slope_run.sh` → `FSRR_FRAME_BUFFER=4000`, **`--adaptive-mode 0`**
- `c3/verify_run.sh` → `FSRR_FRAME_BUFFER=8000 FSRR_RATE_REPLICATE=1`, **`--adaptive-mode 0`**
- `c3/slope_rc_run.sh` → `FSRR_FRAME_BUFFER=8000 FSRR_RATE_REPLICATE=1`, **`--adaptive-mode 0`**
- **C3 디렉터리 전체 로그에서 `hot-swapping` 발생 0회** (grep 확인).

mode 0은 `adaptive_deploy`를 타지 않고 `stop_execution` 경로가 입력 큐를 보존하므로, backlog 궤적은
설정한 깊이(4000/8000) 그대로 측정됐다. **`fig:fluid_validation`·Q1.4 수치 유효. 재실행 불필요.**

(참고: `docs/c3_stress_report.md`의 B=754 항목은 C3 스트레스 스윕의 일부로 **mode 0** 실행이며,
B2와 별개다.)

---

## §4. B2 (논문 Q1.5) — 무효, 근거와 규모

### 4.1 조건 확인 (`c3/b2_run_stay.sh`)
```
FSRR_FRAME_BUFFER=$B   # B: gpu 754, npu 569        FSRR_RATE_REPLICATE=1
MODE = ( Static=3  StopRestart=0  Adaptive=1  BoundGuard=1 )
스케줄: combination_burst = yolo11s on CPU (위반)  ->  cand_1 = yolo11s on gpu/npu
```
단일 뷰(view1)이고 전환이 **cpu → 가속기 device 변경**이므로, mode 1 기법은 반드시 `_hot_swap_view`를 탄다.
로그 실측: Static 0회, StopRestart 0회, **Adaptive 1회, BoundGuard 1회**.

### 4.2 실측 궤적 — 편향이 눈에 보인다 (GPU, `view1_q`)

| 기법 | burst 구간 `view1_q` | 전환 직후 `view1_q` | V(t) |
|---|---|---|---|
| Static | 691 → **754**(포화) | — | 630 → 1688 |
| Stop-restart | 696 | **745 → 754 유지** | 1290 → 854 |
| **Adaptive** | 695 | **2 → 0** | 784 → **12.8** |
| **BoundGuard** | 714 | **2 → 0** | 800 → **11.7** |

전환 직후 적응 기법의 큐만 **754 → 0~2로 붕괴**한다. 비적응 기법은 754를 유지한다.

### 4.3 보고된 지표 (per-frame 로그에서 재계산 — 논문 `tab:b2`와 일치)

| plat/기법 | frames | miss | p99(ms) | p99.9(ms) | mean e2e(ms) |
|---|---|---|---|---|---|
| gpu/Static | 154 | 1.000 | 24612 | 24779 | 12520 |
| gpu/Stop-restart | 2187 | 1.000 | 10731 | 10748 | 7773 |
| **gpu/Adaptive** | 1473 | **0.679** | **5667** | 7712 | **147** |
| **gpu/BoundGuard** | 1768 | **0.665** | **5399** | 7933 | **130** |
| npu/Static | 154 | 1.000 | 24144 | 24311 | 12263 |
| npu/Stop-restart | 1956 | 1.000 | 10665 | 10695 | 6199 |
| **npu/Adaptive** | 1445 | **0.661** | 6208 | 8343 | **171** |
| **npu/BoundGuard** | 2932 | **0.652** | **3827** | 8147 | **90** |

**mean e2e가 7773ms → 147ms (53배)** 로 벌어진 것이 대비의 실체다. 이 차이는 적응 기법이
**축적된 backlog를 서비스하지 않았기 때문**이며(§5), 스케줄링 품질이 아니다.

### 4.4 판정
- **`tab:b2` 8행 전부·`fig:b2`·Q1.5 본문 수치 무효.**
- **"adaptive가 tail을 24s→5s로 낮춘다"는 서술은 현재 데이터로 뒷받침되지 않는다.** 편향 제거 후에도
  일부 우위가 남을 수는 있으나, **얼마가 남는지는 재실행 전까지 알 수 없다.**
- 무효 사유는 §5의 두 번째 편향이 주 성분이므로, **큐 깊이 수정만으로 재실행하면 안 된다.**

---

## §5. 감사 중 발견 — 두 번째 편향: 상속 backlog 폐기 (미수정)

### 5.1 두 경로가 정반대로 동작한다

**`adaptive_deploy._hot_swap_view()` (mode 1/2) — 폐기**
```python
# 4. Drain old queues
for q in (old_frame_q, old_output_q):
    self._drain(q)
```
새 큐로 피더를 갈아끼운 뒤 **옛 입력 큐에 쌓여 있던 프레임을 전부 버린다**. 새 큐는 비어 있다.

**`unified_viewer.stop_execution()` → `_drain_and_close_all_queues()` (mode 0/3/4) — 보존**
```
# Drain only OUTPUT queues. Input frame queues (video_frame_queue,
# view*_frame_queue) are persistent across phase transitions ...
# draining them here would discard exactly the backlog we want the next phase
# to see (and would reset the wait-time spike that drives V(t) up).
```
**입력 큐 보존이 의도된 측정 원칙임이 코드 주석에 명시**되어 있고, `adaptive_deploy`가 이를 위반한다.

### 5.2 영향
- 적응 기법은 배치를 바꿀 때마다 **무상 큐 플러시**를 받는다. 비적응 기법은 축적분을 전부 서비스해야 한다.
- B2에서 그 크기는 **약 700프레임 ≈ λ=123 기준 5.7초 분량**이다. cand_1 구간이 18s이므로 지배적이다.
- **큐 깊이 수정으로 고쳐지지 않는다.** 수정 후에도 새 큐는 754 깊이지만 **비어 있는 채로** 시작한다.
- 방향은 항상 적응 기법에 유리하다.

### 5.3 다른 실험에 대한 영향
| 실험 | 폐기되는 backlog 규모 | 영향 |
|---|---|---|
| §4b | ≤2 프레임 | 무시 가능 → **판정 유효** |
| Q3 (buffer 12) | ≤12 프레임 ≈ 50ms | 무시 가능 → **판정 유효** |
| Q4 (buffer 300) | ≤300 프레임/뷰 ≈ 0.8s, BoundGuard 4회 | 중간. **단 방향이 BoundGuard에 유리한데도 BoundGuard가 졌으므로**, "envelope 우위 없음"이라는 Q4 결론은 **더 강해질 뿐 뒤집히지 않는다** → **판정 유효** |
| **B2 (buffer 754/569)** | **~700 프레임 ≈ 5.7s** | **지배적 → 무효** |

---

## §6. 논문 반영분 중 무효인 것

| 논문 요소 | 판정 | 사유 |
|---|---|---|
| **`tab:b2`** (8행) | **무효** | §4 |
| **`fig:b2`** | **무효** | §4 |
| **Q1.5 본문 수치** (miss·p99·"24s→5s") | **무효** | §4.4 |
| `fig:fluid_validation` | **유효** | §3 (mode 0, hot-swap 0회) |
| Q1.4 본문 수치 | **유효** | §3 |
| Table I (모델 벤치마크) | **유효** | 격리 측정 |
| Q1.5 "no penalty"(§4b BoundGuard=Adaptive) | **유효** | §4b buffer 2 |

> **주의**: Q1.5는 두 주장(§4b "no penalty" / B2 "adaptive 우위")을 함께 담고 있다. 앞은 유효, 뒤는 무효다.

---

## §7. 재실행 계획과 우선순위

### 우선순위 0 (선행 결정, **재실행을 막는 게이트**)
**§5의 backlog 폐기 semantics를 먼저 정해야 한다.** 정하지 않고 B2를 재실행하면 큐 깊이만 맞춘 채
두 번째 편향이 그대로 남아 **또 무효가 되고 재실행을 두 번 하게 된다.**

선택지:
- **(a) 상속 backlog를 새 큐로 이관** — `_hot_swap_view`에서 `old_frame_q`의 항목을 `new_frame_q`로 옮긴 뒤
  나머지를 drain. mode 0의 보존 semantics와 일치. **권장** — `unified_viewer`의 명시된 측정 원칙과 정합.
- (b) 현행 폐기 유지 + mode 0도 폐기하도록 통일 — `unified_viewer`의 의도된 원칙을 뒤집게 되고,
  V(t)의 wait 스파이크가 사라져 C3 fluid 검증 근거와 충돌.
- (c) 현행 유지 + 논문에서 "hot-swap은 stale frame을 폐기하는 정책"으로 **명시**하고, 비교 시
  Stop-restart에도 동일 정책을 부여.

### 우선순위 1 — **B2 재실행** (논문 Table 직결)
- 조건: `c3/b2_run_stay.sh` 그대로(B=754/569, λ=0.9μ, 4기법 × 2플랫폼). 파라미터 불변.
- 소요: 8런 × ≤80s + 정리 ≈ **12분**.
- 산출: `tab:b2` 재작성용 miss/p99/p99.9 + `fig:b2`.
- **예상**: 적응 기법의 wait이 되살아나 miss·tail이 악화 → baseline과의 격차 축소. 격차가 얼마나
  남는지는 실행 전에는 단언할 수 없다.

### 우선순위 2 — Q4·Q3 재검증 (semantics 변경 시)
(a)를 채택하면 hot-swap 동작이 바뀌므로 재검증 필요. Q4 ≈ 11분, Q3 ≈ 9분.
§5.3 근거상 결론이 뒤집힐 가능성은 낮으나 수치는 갱신된다.

### 조치 불필요
- **C3 / `fig:fluid_validation` / Q1.4** — §3.
- **§4b / phaseA / Table I / 오프라인 감사** — §2.

### 별건 (경미)
- `unified_viewer.py:643` headless 큐 깊이를 `_fb`로 정합 (§1.5).

---

## 무결성
- 감사 전용: 코드 변경 0, 재실행 0. main.tex 불변.
- 확인 방법: 코드 경로 정독(`unified_viewer` 디스패치 → `adaptive_deploy`/`reactive_deploy`/
  `restart_rollback_deploy`), 전 실행 로그의 `hot-swapping` 카운트, 메트릭 CSV의 `viewN_q` 실측,
  per-frame 로그에서 miss/p99 재계산.
- 추정으로 판정한 항목 없음. 판정 근거를 항목별로 명시했다.
