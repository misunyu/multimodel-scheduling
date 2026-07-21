# C3 Stress Test — fluid 모델 검증 (중간 보고 + 게이트)

날짜: 2026-07-21 · 범위: vision-only × cpu_gpu·cpu_npu · 성격: 계측 + pilot + slope 진단

## 완료된 것

### 계측 (§1, `unified_viewer.py`)
- per-second metrics CSV에 **`backlog_total`, `view1..4_q`(frame-queue 큐 길이), `drop_count`** 추가(로깅만).
- **buffer 파라미터화**: `FSRR_FRAME_BUFFER`(기본 2=현행 drop-on-full). 검증: B=400에서 backlog 130→382
  누적(무손실), B=50에서 50 포화 + drop 196 계상(**drop→miss 집계 정상**).

### Pilot μ_i (§2, 실측)
| device | yolo11s | resnet50 | mobilenet_v2 |
|---|---|---|---|
| CPU | 6fps (병목) | 19fps | 193fps |
| GPU | 193fps | 424fps | 504fps |
| NPU | 263fps | 565fps | 741fps |

## fluid 모델 검증 결과

### ✅ 누적(accumulation) 모델 — 검증됨 (+1% 오차)
burst(CPU) 구간 yolo11s 큐 궤적:
| config | yolo λ_actual | acc_slope(실측) | λ−μ_cpu(예측) | 오차 |
|---|---|---|---|---|
| gpu | 30fps | 24.0/s | 23.8/s | **+1%** |
| npu | 29fps | 23.0/s | 22.7/s | **+1%** |

- **backlog 누적 기울기 = λ_actual − μ_cpu 를 +1% 오차로 실측 확인.** 누적 기울기로 μ_cpu(≈6fps)도
  교차 검증됨(λ_act 30 − slope 24 ≈ 6).
- drain 궤적도 깨끗함(gpu_l0.90: backlog_total 1284→0을 μ*−λ 속도로 배수).

### ⚠️ 게이트 — 발산 regime 도달 불가 (feeder가 λ를 제어 못 함)
**실측 도착률 λ_actual이 infps 설정과 무관**:
- **비디오 feeder(yolo): λ_actual ≈ 30fps** — 비디오 소스 fps에 묶임(infps=174/237 설정 무시).
- 이미지 feeder(resnet): λ_actual ≈ 130fps.
- 결과: μ*(GPU 193 / NPU 263) ≫ λ_actual(30) 이라 **λ = 0.7~0.95×μ* 스윕이 불가능**하고,
  fluid 모델의 핵심 signature인 **T_stable = Q/(μ*−λ) 발산(λ→μ*)을 시연할 수 없음.**

## 판정
- **fluid 누적 모델은 실측 검증됨**(+1%). 정성적 accumulate/drain 거동 확인.
- **발산 signature(R4 핵심)는 미시연** — feeder가 λ를 μ* 근처까지 밀지 못함(소스 fps 제한). 이는
  시나리오/버퍼로 안 풀리고 **feeder에 controlled arrival-rate(rate-limit 또는 frame-replication)를
  넣어야 하는 코드 변경**이 필요 → 게이트(§0 "계측만" 범위 밖).

## 게이트에서 결정할 것
발산 regime(λ→μ*)을 시연하려면:
1. **feeder rate-control 추가** — 비디오/이미지 feeder가 목표 λ(프레임 복제 포함)를 실제로 밀도록. 그래야
   λ=0.7~0.95×μ* 스윕과 T_stable 발산 overlay(§6a 핵심 그림)가 가능. → 코드 변경, 사람 확인.
2. 또는 **μ*를 낮춰 λ 도달 가능하게** — 더 무거운 모델/더 큰 배치로 μ*를 feeder 도달 범위(~30-130fps)로.
   (모델 구성 변경 — 신규 셋 재조정)
3. 또는 **누적 모델 검증(+1%)으로 R4 부분 대응** + 발산은 해석적 외삽으로 논문 기술(실측 없이).

## 무결·불변
- 전 slope run cuDNN 폴백 **0**(GPU 무결). NPU .mxq 정상. mode 확인.
- 계측 스냅샷 `backup/c3_instrument_20260721_175310/`. legacy/scripts/backup 무변경.

---

# 업데이트 (feeder rate-control 추가 후)

## feeder rate-control (선택지 1, `view_handlers.py`)
- `FSRR_RATE_REPLICATE=1`: 두 feeder(VideoFeeder/ResnetImageFeeder)가 경과시간×목표율만큼 **프레임 복제
  enqueue** → λ_actual = 목표 infps(소스/루프 fps 초과 가능). 기본(unset)=현행 throttle-only(불변).
- **복제=실제 부하 검증**: 목표 λ=120 → backlog slope 114.2/s + proc 6fps → **λ_actual=120fps 정확**,
  backlog 선형 증가(각 복제본이 full forward pass = 실추론).

## ✅ fluid 누적 모델 — 검증됨 (평균 오차 0.3%, rate-control 후 유지)
| config | λ_target | λ_actual | acc_slope(실측) | λ−μ_cpu(예측) | 오차 |
|---|---|---|---|---|---|
| gpu 0.70/0.90/0.95× | 135/174/183 | 136/174/183 | 130/168/177 | 129/168/177 | +1/0/0% |
| npu 0.70/0.90/0.95× | 184/237/250 | 185/237/250 | 179/231/244 | 178/231/244 | +0/0/-0% |

- rate-control로 λ가 정확히 제어되고, **backlog 누적 기울기 = λ−μ_cpu가 전 스윕에서 평균 0.3% 오차**로
  실측 일치. μ_cpu(≈5-8fps)도 누적 기울기로 교차 검증. **R4 fluid 모델(누적)의 실증.**

## ⚠️ drain/발산 regime — 부하 의존 실효 μ*로 미해결 (후속)
- drain 단계 기울기가 예측 −(μ*−λ)과 불일치(양수, backlog 성장). 원인: **실효 μ*가 부하에 의존**.
  - 직접 측정(saturated single, backlog 없음): μ*_eff ≈ GPU 257 / NPU 267fps.
  - 스윕 drain 단계(대형 backlog + replication feeder): μ*_eff ≈ 138 / ~104fps로 저하.
  - 즉 **대형 backlog + 고율 프레임 복제(720p, 470MB/s copy) + preprocess가 CPU를 경합**해 drain 구간
    실효 처리율이 떨어짐. fluid 모델의 constant-μ* 가정이 이 regime에서 흔들림(실제 시스템 동역학).
- 결과: drain 기울기→0 발산 overlay는 **아직 깨끗이 추출 못 함.** 필요 조치(후속):
  - config별 offload steady-state μ*_eff를 측정해 그 값 기준으로 drain 예측, 또는
  - feeder 복제 오버헤드 축소(프레임 다운스케일/공유 버퍼)로 μ*_eff를 부하-독립적으로.

## 판정
- **누적 fluid 모델: 실측 검증(0.3%)** — R4 핵심의 절반(accumulation) 실증 완료.
- **발산 signature: 미해결** — 부하 의존 μ*가 원인, 후속 필요(steady-state μ*_eff 측정 또는 feeder 경량화).
- GPU 폴백 0·NPU 무결 유지. 계측·rate-control 스냅샷 backup/에 보존.

---

# 진단 결과 + drain/발산 검증 (해결)

## 진단: 부하 의존 μ = (A) 복제 아티팩트 vs (B) 시스템 성질
같은 backlog를 feeder ON(복제 λ=200) vs OFF(순수 drain λ=2)로 배수해 μ* 비교:
| plat | feeder OFF μ* | feeder ON μ* | 직접(추론-only proc_fps) |
|---|---|---|---|
| GPU | 117fps | 137fps | 255fps |
| NPU | 105fps | 103fps | ~285fps |

- **feeder ON≈OFF** (GPU 117≈137, NPU 105≈103) → **복제 아티팩트 아님. (B) 실제 시스템 성질.**
- 핵심: proc_fps(추론-only)=255이나 **실제 파이프라인 drain rate ≈120fps**. GPU 추론은 3.9ms로 빠르지만
  per-frame 파이프라인(dequeue+frame.copy+letterbox+NMS+handler)이 CPU-bound → 실효 μ*가 추론율의 절반.
- **즉 μ*는 파이프라인-제한이며 복제와 무관하게 대략 일정.** 앞선 drain 불일치의 원인은 잘못된 μ*
  (추론 기반 193/257)를 쓴 것 — 올바른 μ*는 **파이프라인 drain rate(GPU≈137, NPU≈103fps)**.

## ✅ drain/발산 검증 (올바른 파이프라인 μ* 기준, 이론=실측)
λ = frac × μ*_pipeline:
| config | λ | drain 실측 | μ*−λ 예측 | 오차 |
|---|---|---|---|---|
| gpu 0.70/0.90/0.95× | 96/123/130 | -41.7/-14.2/-8.0 | -41/-14/-7 | +2/+1/+14% |
| npu 0.70/0.90/0.95× | 72/93/98 | -33.5/-11.9/-7.1 | -31/-10/-5 | +8/+19/+43% |

- **drain 기울기 = −(μ*−λ)** 정합(GPU 2-14%, NPU 8-43%; 0.95×는 절대값이 작아 %오차 증폭).
- **drain 기울기 → 0 (λ→μ*)**: GPU 41.7→14.2→8.0, NPU 33.5→11.9→7.1 — **T_stable=Q/(μ*−λ)→∞ 발산
  signature 양 플랫폼 실증.**

## 최종 판정 (C3 fluid 모델)
- **누적 검증**: slope=λ−μ_cpu, 평균 0.3% 오차.
- **drain 검증**: slope=−(μ*−λ), GPU 1-14% / NPU 8-43%(절대값 작아 증폭), **발산 추세 명확**.
- **μ 성질**: 파이프라인-제한(추론율의 ~절반), 복제-독립(B). **§II fluid 모델은 μ*를 추론율이 아닌
  파이프라인 service rate로 명시해야 함**(이론-실측 정합의 전제).
- **R4 대응**: fluid 모델(누적+drain+발산) 예측 vs 실측 정합 실증 완료. GPU 폴백 0·NPU 무결.

## 남은 것 (§6a 그림화 + B2 지표)
- 위 slope 데이터로 Q(k) overlay 그림(예측 vs 실측) + T_stable 발산 곡선 작성 → §IV, "Scope of the model" 빨강 해제.
- buffer 스윕(1.0/0.5×도달 backlog)으로 drop→miss, B2 지표(miss rate·drop 스택·p99/p999) 산출 — 계측·rate-control 인프라 재사용.

---

# C3 마무리 — buffer 스윕 + B2 + overlay 그림

## §2 Buffer B 재산출 (유한 10s 구간 도달 backlog, 파이프라인 μ*)
| plat | λ(0.7/0.9/0.95×) | Q_obs | B 0.5×(drop 유발) |
|---|---|---|---|
| GPU | 96/123/130 | 1151/1509/1621 | 575/754/810 |
| NPU | 72/93/98 | 836/1139/1202 | 418/569/601 |

## §3 drop→miss 집계 검증 ✅
0.5×B(GPU 0.90×, B=754): backlog가 B=754에서 포화 → drop_count 0→58→175→…→646(burst)로 정확 계상.
metrics drop_count == results dropped_frames == 큐-full 이벤트. (mode-0 전환 시 feeder 재생성으로 카운터
리셋되므로 phase별 합산 필요.) **집계 정확 — B2 drop 성분 신뢰 가능.**

## §4 B2 지표 (데모, 0.5×B GPU 0.90×)
- N_arrived≈1442, N_drop=646, N_late≈42 → **miss rate ≈ 48%**, **drop이 miss의 94%**(tight buffer).
- 완전한 4기법 비교 + p99/p999 tail은 **per-frame latency 로깅 추가 필요**(현재 per-second 집계만) — 후속.

## §5a fluid overlay 그림 ✅ `docs/figures/c3_fluid_validation.pdf`
3-panel: (a) 누적 dQ/dt=λ−μ_cpu, (b) drain μ*−λ, (c) 발산 drain→0(λ→μ*). 수치표:
| plat | λ | acc 실측/예측(err) | drain 실측/예측(err) |
|---|---|---|---|
| GPU | 96/123/130 | +1/+1/+0% | +2/+1/+14% |
| NPU | 72/93/98 | +1/+0/−1% | +8/+19/+42% |
- NPU drain 고λ %오차는 **절대값이 작아 증폭**(0.95×: 실측 7.1 vs 예측 5.0). 발산 추세(drain→0)는 양
  플랫폼 일관, **GPU(1~14%)가 주 증거**.

## §6 §IV 논문 반영 — 블로커
- **main.tex가 이 저장소에 없음** → §IV 텍스트/그림 반영 불가. 논문 저장소 경로 필요.
- 반영 준비물(확정): 파이프라인 μ*(GPU 137/NPU 103), λ=μ* 상대 설정, B 스윕값, fluid overlay 그림,
  예측vs실측 오차표.

## 남은 것 (후속)
1. **per-frame latency 로깅** 추가 → 4기법 B2(miss rate·drop 스택·p99/p999) 완성.
2. **논문 저장소 경로 확인** → §IV 반영(남색).

---

# B2 지표 완성 (per-frame latency 로깅 + 4기법 buffer 스윕)

## per-frame 로깅 (계측만, `view_handlers.py`)
- `FSRR_PERFRAME_LOG=<path>`: 각 완료 프레임을 `view,model,infer_ms,wait_ms,e2e_ms,deadline_ms,late`로 기록.
  e2e=infer+wait, deadline=slo_ms 또는 1000/infps, late=e2e>deadline. drop은 feeder가 별도 집계(§3 검증됨).
- sanity: 무부하(λ30≪μ*137) late 4%(콜드스타트), 과부하(λ60≫μcpu6) late 100%(e2e p50 5685ms). 정상.

## B2 결과 (0.5×B drop regime, λ=0.9×μ*, 4기법 × 양 플랫폼)
| config | miss rate | drop:late (of miss) | p99 | p999 |
|---|---|---|---|---|
| GPU Static | 100% | 94:6 | 24.5s | 24.8s |
| GPU StopRestart | 100% | 26:74 | 11.0s | 11.1s |
| GPU Adaptive | 84% | 50:50 | 5.8s | 7.9s |
| GPU BoundGuard | 83% | 48:52 | 5.2s | 7.9s |
| NPU Static | 100% | 91:9 | 24.3s | 24.6s |
| NPU StopRestart | 100% | 18:82 | 10.5s | 10.5s |
| NPU Adaptive | 73% | 29:71 | 6.0s | 8.2s |
| NPU BoundGuard | 76% | 35:65 | 6.9s | 8.4s |

- **서열**: Static≈StopRestart(100% miss, p99 24s/11s) ≫ **Adaptive≈BoundGuard(73-84% miss, p99 5-8s)**.
  적응(hot-swap)이 miss rate·tail을 극적으로 개선. Static은 drop-지배(위반 placement 고정→backlog 무한),
  StopRestart는 late-지배(재시작 downtime).
- **BoundGuard ≈ Adaptive**: 이 시나리오는 예측기 top-1이 정확해 동점(예상). BoundGuard의 distinctive
  우위(오예측 하 후보 사이클)는 Q3 오예측 시나리오 후속(§7).
- **높은 절대 miss(73-100%)**: tight 0.5×B + 고부하(λ=0.9×μ*)라 burst 누적 backlog가 대부분 late. 적응은
  tail·최종 miss를 낮추나 burst 구간 backlog는 불가피.

## 그림
- `docs/figures/c3_fluid_validation.pdf` (5a: 누적·drain·발산 overlay)
- `docs/figures/c3_b2_metrics.pdf` (5b: miss rate drop/late 스택 + p99/p999 tail, 양 플랫폼)

## 무결
- 전 8 run cuDNN 폴백 0(GPU 무결). NPU 무결. mode 1(BoundGuard) 확인. per-frame 로깅=계측만(제어 불변).
- 변경: `view_handlers.py`(rate-control + per-frame 로깅), `unified_viewer.py`(backlog 로깅 + buffer). legacy/scripts/backup 무변경.

## 논문 반영 준비물 (별도 대화)
- 4기법 × 양 플랫폼 miss rate/drop-late split/p99·p999 표(위), B2 그림 PDF.
- fluid overlay 그림 + 예측vs실측 오차표. 파이프라인 μ*(GPU 137/NPU 103), λ=μ* 상대, B 스윕값.
- miss 정의: (N_late + N_drop)/N_arrived, late=e2e>L_SLO, drop=큐-full(집계 검증됨).
