# 전환 무손실 측정 — contribution 3의 실측 근거

날짜: 2026-07-24 · 알고리즘·파라미터 불변 · main.tex 불변 · 관련: [backlog_preserve_report.md](backlog_preserve_report.md)

> **판정**: **손실 0 · 중복 0 확인.** hot-swap 전환에서 유입=완료+드롭+잔여가 정확히 성립(balance=0),
> 상속 backlog는 `moved=N dropped=0`으로 이관. **단 "무중단"은 한정 필요**: 전환 시 **교체되는 뷰만**
> 워커 cold-start로 **~1.0s 완료 공백**이 생긴다(나머지 뷰는 계속 서빙). stop-restart는 **전 뷰가 동시에
> ~1.5–2.2s** 멈춘다. → **"무손실 + 비교체 뷰 무중단"**이 정확한 서술이며, 문자 그대로의 "no service
> interruption"은 교체 뷰에 한해 ~1s cold-start를 명시해야 한다.

## §1. 측정 설계
- **시나리오**: vision-only 3뷰(yolo11s/11m/resnet50), λ=40(μ* 여유), **대형 버퍼 B=2000**(overflow drop을
  switching loss와 분리 — §III의 "버퍼가 넘치지 않는 한" 조건). 4페이즈로 **hot-swap 다중 전환**(all-accel →
  view1 CPU → view1+2 CPU → all-accel), 양 플랫폼.
- **손실/중복 (프레임 회계)**: feeder에 `enqueue_counts` 카운터 추가(성공 enqueue). 종료 시 뷰별
  **enqueued vs completed(per-frame 로그 라인 수) + dropped + residual(qsize)** 대조. 항등식
  `enqueued − (completed + dropped + residual) = 0`이면 net 손실·중복 없음(중복이면 completed 초과 → 음수).
  - *(주의: 완료 수는 per-frame 로그로 셈. hot-swap은 핸들러를 교체하므로 현재 핸들러의 `infer_count`는
    교체 전 완료분을 놓쳐 과소계상 — per-frame 로그가 전 핸들러 완료를 포착하는 올바른 소스.)*
- **무중단 (gap)**: per-frame 완료 타임스탬프의 **뷰별 최대 연속 간격**. 서비스가 멈추면 gap이 나타난다.
- **대조군**: 동일 시나리오 stop-restart(mode 0).

## §2. 결과 (양 플랫폼)

### 2.1 손실·중복 — **balance ≈ 0, drop 0**
| plat | 기법 | Σ balance(enq−comp−drop−resid) | Σ drop | transfers(이관/드롭) |
|---|---|---|---|---|
| GPU | Adaptive | **0** | 0 | moved 553·310, dropped 0 |
| GPU | Stop-restart | **0** | 0 | (mode0, 입력큐 보존) |
| GPU | BoundGuard | **1** (in-flight 1프레임) | 0 | moved 549·306, dropped 0 |
| NPU | Adaptive | **3** (in-flight) | 0 | dropped 0 |
| NPU | Stop-restart | **0** | 0 | — |
| NPU | BoundGuard | **0** | 0 | dropped 0 |

- **balance 0–3**(스냅샷 순간 in-flight 프레임, 무시 가능) → **net 손실·중복 없음.**
- **transfers dropped=0** — 상속 backlog(수백 프레임)가 전부 새 워커로 이관, 폐기 0.
- B=2000으로 **overflow drop=0** → 관측된 무손실은 **switching 자체의 속성**(버퍼 여유가 가린 것 아님).

### 2.2 무중단 (gap) — **hot-swap은 교체 뷰만, stop-restart는 전 뷰**
per-view 최대 완료 gap(초):

| plat | 기법 | view1 | view2 | view3 | 해석 |
|---|---|---|---|---|---|
| GPU | Adaptive | **1.00** | **1.01** | 0.16 | 교체된 view1/2만 ~1s(cold-start), view3 정상 |
| GPU | Stop-restart | 1.74 | **2.06** | 1.50 | **전 뷰 동시** 1.5–2.1s(전체 재시작) |
| GPU | BoundGuard | 0.21 | 1.00 | 0.15 | 교체 뷰만 ~1s |
| NPU | Adaptive | 1.00 | 1.00 | 0.16 | 동일 |
| NPU | Stop-restart | 1.78 | **2.21** | 1.64 | 전 뷰 동시 |
| NPU | BoundGuard | 0.20 | 1.00 | 0.15 | 교체 뷰만 |

- **hot-swap**: 교체되는 뷰만 워커 cold-start로 **~1.0s gap**, **비교체 뷰는 정상 서빙**(gap 0.15–0.21s =
  정상 프레임 간격). → 시스템 전체는 무중단.
- **stop-restart**: 전 뷰가 **동시에 1.5–2.2s** 멈춤(전체 워커 재시작). hot-swap의 약 2배.

## §3. 판정 (§A3)
1. **손실 0 · 중복 0**: 실측 확인 → **contribution 3의 "no request loss/duplication" 뒷받침.** 논문에 수치 삽입 가능.
2. **"무중단(no service interruption)"은 한정 필요**: 교체되는 뷰는 워커 cold-start로 **~1.0s 공백**이 있다.
   → 서술 조정: *"non-migrated 뷰의 서비스는 중단 없이 지속되고, migrated 뷰만 워커 cold-start(~1s)
   동안 짧게 멈춘다. 전체 시스템 정지는 없다(stop-restart는 전 뷰 1.5–2.2s 정지)."*
3. **버퍼 여유 조건 명시**: 무손실은 B가 넘치지 않는 한. overflow 레짐의 drop은 Q1.5(B2)에서 별도로 다룸.

## §4. 무결성
- 코드 변경: `view_handlers.py`(VideoFeeder·ResnetImageFeeder에 `enqueue_counts` 카운터, 순수 계측·동작 불변),
  `schedule_executor_main.py`(`_current_enqueues`·`_dump_accounting`, **FSRR_ACCT 게이트**로 기본 실행 무변경).
- P1(순서)·P3(fallback=0)·P4(대형버퍼로 overflow 분리)·P7(transfer/ACCT 로깅) 확인.
- legacy/backup·main.tex 불변.
