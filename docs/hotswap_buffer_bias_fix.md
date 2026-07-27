# hot-swap 큐 깊이 편향 — 발견·수정·영향 감사

날짜: 2026-07-24 · 대상: `adaptive_deploy.py` · 스냅샷: `backup/adaptive_deploy.pre_bufferfix.py`

> Q4 결과를 해석하던 중 발견한 **측정 무결성 결함**이다. 결론을 보기 좋게 만들기 위한 변경이 아니라,
> 기법 간 비교 조건을 동일하게 맞추기 위한 수정이며, 수정 방향은 BoundGuard에 **불리하게** 작용한다
> (아래 §4).

## §1. 결함

`unified_viewer.py`는 뷰별 입력 큐 깊이(= fluid 모델의 버퍼 `B`)를 `FSRR_FRAME_BUFFER`로 파라미터화한다
(기본 2). 그런데 hot-swap 경로인 `adaptive_deploy.py`는 교체 큐를 **하드코딩된 `maxsize=2`** 로 만들고
있었다:

```python
new_frame_q = Queue(maxsize=2)     # L255 (교체 워커)
frame_q     = Queue(maxsize=2)     # L323 (신규 뷰)
v.headless_frame_queues[hid] = Queue(maxsize=2)   # L377
```

따라서 **한 번이라도 hot-swap된 뷰는 그 이후 영구히 깊이 2로 동작**한다. `FSRR_FRAME_BUFFER=300`으로
실행한 Q4에서는 교체된 뷰의 큐가 **150배 얕아진다**.

## §2. 왜 지표를 오염시키는가

`ℓ_i = infer + wait`이고 `wait`은 큐 대기시간이다. 큐가 얕으면:

1. **wait이 기계적으로 줄어든다** → `ℓ_i` 하락 → `V(t)` 하락. 배치가 좋아져서가 아니라 버퍼가 얕아서다.
2. 초과 부하가 backlog가 아니라 **드롭**으로 배출된다 → `V(t)`에 **생존편향**이 걸린다
   (느린 프레임은 완료되지 않으므로 평균에서 사라진다).
3. `backlog_slope`가 `λ−μ`를 추정하지 못한다(큐가 상한에 클리핑).

즉 [saturation_guard_report.md](saturation_guard_report.md)가 다루는 세 가지 병리가 **버퍼 설정과
무관하게, hot-swap을 했다는 이유만으로** 유발되고 있었다.

## §3. 기법 간 불공정 (핵심)

Q4(λ=90/뷰, `FSRR_FRAME_BUFFER=300`) 사전 데이터에서 확인된 실제 큐 점유:

| 기법 | hot-swap 횟수 | 실효 큐 깊이 | 관측 `viewN_q` |
|---|---|---|---|
| Static | 0 | 300 | 299–300 (포화) |
| Stop-restart | 0 (cand_1 = burst와 동일 배치) | 300 | 299–300 (포화) |
| Adaptive | 0 (동일 배치 → "keeping worker") | 300 | 299–300 (포화) |
| **BoundGuard** | cand_2~5에서 매번 1–2뷰 | **2** | **1–2** |

BoundGuard만 얕은 큐로 달렸다. 그 결과 BoundGuard의 tail `V≈5` 대 baseline `≈370`이라는 격차는
**상당 부분 큐 깊이 아티팩트**이며, 스케줄링 이득으로 해석할 수 없다. 수정 전 Q4의 envelope 비교는
따라서 **동일 조건 비교가 아니었다**.

(수정 전 Q4 cand_5 tail 실측: `backlog≈5`, `viewN_q≈1–2`, 드롭 증가율 ≈175 frame/s — 큐가 항상 상한에
닿아 있었음을 뒷받침한다.)

## §4. 수정

`adaptive_deploy._frame_buffer()`를 추가하고 세 지점 모두 이를 사용하도록 변경. `UnifiedViewer`와
**동일한 source of truth**(`FSRR_FRAME_BUFFER`, 기본 2)를 읽는다. 출력 큐는 표시 게이트이므로 `maxsize=1`
그대로.

- **기본 동작 보존**: 환경변수 미설정 시 `_frame_buffer()==2` → 종전과 바이트 단위로 동일.
- **수정 방향**: BoundGuard의 큐가 다시 깊어지므로 wait이 늘고 `V(t)`가 **올라간다**. 즉 이 수정은
  BoundGuard에 유리하지 않다.

## §5. 영향 감사 (재실행 대상)

| 실험 | `FSRR_FRAME_BUFFER` | hot-swap 발생 | 영향 | 조치 |
|---|---|---|---|---|
| §4b (`bg_scenario.py`) | 미설정 → **2** | 있음 | **없음** (교체 큐도 2로 동일) | 재실행 불필요 |
| B2 / C3 | 미설정 → 2 | — | 없음 | 불필요 |
| Δ 스윕 (`delta_sweep.py`) | 미설정 → 2 | 없음 | 없음 | 불필요 |
| **Q3** (`q3_mixed_scenario.py`) | 12 | 있음 (BoundGuard만) | **있음** (12→2) | **재실행** |
| **Q3-paperset** (`q3v4_scenario.py`) | 30 | 있음 (BoundGuard만) | **있음** (30→2) | **재실행** |
| **Q4** (`q4_scenario.py`) | 300 | 있음 (BoundGuard만) | **있음** (300→2) | **재실행** |

## §6. 재실행 결과 — 수정 전/후

### 6a. Q4 (heavy-4, λ=90, buffer=300) — **결론이 뒤집힘**
| 기법 | maxV 전 → 후 | lastV 전 → 후 | persist |
|---|---|---|---|
| Static | 372.8 → 384.0 | 367.1 → 381.7 | 89s |
| Stop-restart | 373.7 → 375.1 | 369.2 → 367.9 | 89s |
| Adaptive | 381.3 → 374.2 | 377.2 → 366.6 | 89s |
| **BoundGuard** | 275.4 → **739.6** | 5.0 → **735.8** | 61s |

- hot-swap이 없는 세 기법은 전부 오차 범위 내(±3%) — 수정이 이들에 영향을 주지 않음을 확인.
- **BoundGuard만 크게 변했고, 방향은 악화**다. 수정 전의 "낮은 envelope"은 전부 얕은 큐 아티팩트였다.
  자세한 해석은 [q4_experiment_report.md](q4_experiment_report.md) §5.
- 큐 점유가 전 기법 동일(`viewN_q`≈300)로 맞춰진 것을 CSV에서 확인 — 수정 검증 완료.

### 6b. Q3 (vision-3 + LLM, λ=80, buffer=12) — **결론 유지**
| 기법 | maxV 전 → 후 | lastV 전 → 후 | persist 전 → 후 | 회복 |
|---|---|---|---|---|
| Static | 3.74 → 3.84 | 3.26 → 3.51 | 105 → 105s | ✗ |
| Stop-restart | 3.91 → 3.67 | 3.67 → 3.38 | 106 → 106s | ✗ |
| Adaptive | 3.74 → 3.88 | 3.38 → 3.42 | 107 → 106s | ✗ |
| **BoundGuard** | 13.5 → 61.3 | 0.0005 → **0.00** | 22 → **20s** | **✓** |

Q3의 질적 결론(Adaptive 고착 / BoundGuard만 회복)은 공정 조건에서도 그대로다. BoundGuard의 탐색 중
transient `maxV`가 13.5→61.3으로 오른 것은 큐가 정상 깊이(12)로 돌아와 wait이 지표에 반영된 결과이며,
**수정이 BoundGuard에 유리하게 작용하지 않았음**을 다시 보여준다.

### 6c. Q3-paperset (vision-4, λ=52, buffer=30) — **시나리오 자체가 무효**
| 기법 | maxV | lastV | persist |
|---|---|---|---|
| Static | 0.20 | 0.18 | **0s** |
| Stop-restart | 0.21 | 0.05 | **0s** |
| Adaptive | 0.19 | 0.18 | **0s** |
| BoundGuard | 0.40 | 0.16 | **0s** |

네 기법 모두 **위반에 진입조차 하지 않는다**. 실측 지연(12–14ms / 5–7ms)이 per-model `L_SLO`
(155/209/56/32ms)보다 한 자릿수 낮고 backlog≈0, 드롭 0 — 이 워크로드는 λ=52에서 그냥 **feasible**하다.

이것은 버퍼 수정이 아니라 **Δ-창 `V(t)` 수정**([vt_definition_fix_report.md](vt_definition_fix_report.md))의
귀결이다. 이 시나리오는 그 수정 이후 재실행된 적이 없었다. 과거 이 시나리오가 보이던 `V≈7.5–13.8`,
`persist 78–79s`는 `ℓ_i`가 lifetime 누적평균이라 burst 전이가 영원히 감쇠하지 않은 **지표 아티팩트**였다.

**조치**: `docs/figures/q3_paperset_q4like.pdf`는 폐기(`_RETIRED` 접미사로 이동). Q3 시연은 실제로
infeasible해지는 vision-3 + LLM 시나리오(`q3_misprediction.pdf`)만 사용한다. 논문에서 vision-4
paper-set으로 Q3를 보이려면 **부하를 λ=52보다 높여 재설계**해야 한다(별건).


## 무결성
- 변경 파일: `adaptive_deploy.py` 단독. legacy/scripts/backup·main.tex 불변.
- 스냅샷: `backup/adaptive_deploy.pre_bufferfix.py`.
