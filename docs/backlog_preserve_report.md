# 상속 backlog 보존 수정 + B2 재실행

날짜: 2026-07-24 · 스냅샷: `backup/backlog_preserve_20260724_*/` · main.tex 불변
선행 감사: [buffer_bias_impact_audit.md](buffer_bias_impact_audit.md) §5

> **핵심 결과**: B2에서 **"적응 ≫ 비적응"의 강한 형태(miss 0.68 vs 1.00)는 남지 않는다.**
> 이전 격차는 hot-swap이 상속 backlog를 폐기해 얻은 **무상 큐 플러시**였다. 남는 것은
> (a) Static 대비 압도적 우위, (b) Stop-restart 대비 **p99 약 18%** 우위다.
> 이 수정은 적응 기법에 **불리하게** 작용한다 — 결과를 좋게 만들기 위한 변경이 아니다.

## §1. 수정 설계 (게이트 확인 결과)

### 1.1 보존 방식: **재사용이 아니라 이관**
옛 입력 큐를 새 워커에 그대로 넘기는 방식(재사용)은 채택하지 않았다. 새 워커는 옛 워커가 정지되기
**전에** 생성·ready되므로, 큐를 공유하면 겹침 구간에 **두 워커가 같은 큐를 동시에 소비**해 프레임이
두 디바이스로 갈린다. 2-큐 + 원자적 스왑 구조가 전환을 정의 가능하게 만드는 부분이므로 구조는 유지하고
**내용만 옮긴다**.

### 1.2 이관 시점: **피더 스왑 직전**
```
1. _transfer(old_frame_q -> new_frame_q)   # new_frame_q가 아직 비어 있음
2. _swap_feeder_queue(...)                 # 이후 도착분은 새 큐로
3. handler.result_queue = new_output_q
4. 옛 워커 정지 (shutdown + join)
5. _transfer(old_frame_q -> new_frame_q)   # 스왑 중 들어온 잔여분 수거
   _drain(old_output_q)                    # 출력 큐만 비움
```
이 순서의 근거:
- **FIFO 순서 보존** — 옛 프레임이 새 도착분보다 먼저 들어간다.
- **용량 부족 없음** — 이관 시점에 새 큐가 비어 있고 깊이가 같다(`_fb`).
- 겹침 구간에 옛 워커가 몇 프레임을 가져가는 것은 무해하다(그 시점 배치는 아직 옛 배치).

### 1.3 출력 큐: **계속 drain**
mode 0(`_drain_and_close_all_queues`)도 출력 큐만 비우고 입력 큐는 보존한다. 동일하게 맞췄다.

### 1.4 큐 깊이 정합
- `adaptive_deploy._frame_buffer()`와 신규 `unified_viewer._frame_buffer()`가 **동일한
  `FSRR_FRAME_BUFFER`(기본 2)** 를 읽는다. 두 경로가 어긋나면 뷰 깊이가 "hot-swap된 적 있는가"에
  좌우된다.
- `_start_fresh_view()`: 뷰에 기존 입력 큐가 있으면 **재사용**(mode 0의 create-once-then-reuse와 동일),
  없을 때만 생성.
- **headless 정합**(감사 §1.5): `unified_viewer.py`의 `Queue(maxsize=2)` 하드코딩을 `_frame_buffer()`로
  교체. 현재 headless는 LLM background뿐이라 결과 영향은 없으나 잠재 편향을 제거.

## §2. 구현
- `adaptive_deploy.py`: `_transfer(src, dst)` 추가(이동 수·손실 수 반환), `_hot_swap_view._watcher()`
  재배열(§1.2), `_start_fresh_view()` 큐 재사용, 전환마다 로그
  `backlog preserved across hot-swap (moved=N dropped=M)`.
- `unified_viewer.py`: 모듈 레벨 `_frame_buffer()` 추가(단일 source of truth), inline env 파싱 대체,
  headless 큐 깊이 정합.
- 판정 로직·파라미터 불변: N_cand·α·β·ε·T·T_v·Δ·θ·추세 판정·포화 가드·auto-advance·commit-and-stay 무변경.

## §3. 검증 — backlog 연속성

**단위 검증**: 순서 보존(`[0,1,2,3]`), 오버플로 계수(`moved=2 lost=3`), None 안전 확인.

**실행 검증** (B2, B=754/569, 전 hot-swap 실행):
```
[AdaptiveDeploy] view1: backlog preserved across hot-swap (moved=754 dropped=0)   # gpu Adaptive
[AdaptiveDeploy] view1: backlog preserved across hot-swap (moved=754 dropped=2)   # gpu BoundGuard
[AdaptiveDeploy] view1: backlog preserved across hot-swap (moved=569 dropped=4)   # npu Adaptive
[AdaptiveDeploy] view1: backlog preserved across hot-swap (moved=570 dropped=5)   # npu BoundGuard
```
큐 전체가 손실 없이 이관됐다(dropped 0~5는 스왑 중 새 큐가 일시적으로 찬 경우로, 이관량의 1% 미만).

**mode 1과 mode 0의 backlog 거동 비교** (GPU, `view1_q` 궤적):
| 기법 | 전환 전 | 전환 후 배수 궤적 |
|---|---|---|
| Stop-restart (mode 0) | 713 | 751 → 679 → 608 → 531 → 456 → 388 |
| **Adaptive (mode 1)** | 703 | **753 → 684 → 615 → 543 → 476 → 406** |

**정성적으로 동일**하다. 수정 전에는 Adaptive가 전환 직후 `2 → 0`으로 붕괴했다. 목표 달성.

## §4. 재실행

### 4a. B2 — 논문 `tab:b2`·`fig:b2`·Q1.5

조건: 4기법 × 2플랫폼, drop regime(λ=0.9μ*, B=754(GPU)/569(NPU)), `FSRR_RATE_REPLICATE=1`.

#### 4a-1. 원 조건 그대로 (참고) — **시간 불일치가 있어 BoundGuard 행은 비교 불가**
| run | 실행s | served | late | drop | miss | p99 | p99.9 | mean |
|---|---|---|---|---|---|---|---|---|
| gpu/Static | 21 | 157 | 157 | 2277 | 1.000 | 25082 | 25213 | 12737 |
| gpu/Stop-restart | 21 | 2713 | 2713 | 0 | 1.000 | 10582 | 10610 | 5581 |
| gpu/Adaptive | 20 | 2531 | 2531 | 307 | **1.000** | 8582 | 8617 | 5254 |
| gpu/BoundGuard | **39** | 5207 | 5075 | 289 | 0.975 | 8727 | 8785 | 3142 |
| npu/Static | 20 | 155 | 155 | 1615 | 1.000 | 24381 | 24512 | 12399 |
| npu/Stop-restart | 20 | 1987 | 1987 | 0 | 1.000 | 10399 | 10432 | 5932 |
| npu/Adaptive | 20 | 1891 | 1891 | 224 | **1.000** | 8539 | 8573 | 5424 |
| npu/BoundGuard | **38** | 3836 | 3836 | 210 | 1.000 | 8353 | 8408 | 3588 |

> **감사 중 발견한 별개 결함**: `b2_run_stay.sh`가 BoundGuard에만 `cand_2=18`을 추가로 주어
> **BoundGuard 38–39s vs 나머지 20–21s**로 실행된다. 시간이 길수록 backlog가 더 배수되므로 BoundGuard
> 수치가 그만큼 유리해진다. **이 시간 불일치는 이번 수정과 무관하게 기존 발표 수치에도 있었다.**
> 따라서 아래 시간대응 결과를 정본으로 삼는다.

#### 4a-2. **시간대응 (전 기법 44s) — 정본**
| run | 실행s | served | late | drop | miss | p99 | p99.9 | mean |
|---|---|---|---|---|---|---|---|---|
| gpu/Static | 39 | 262 | 262 | 4295 | 1.000 | 41798 | 42133 | 21168 |
| gpu/Stop-restart | 40 | 5361 | 5160 | 0 | 0.963 | 10516 | 10563 | 3221 |
| gpu/Adaptive | 40 | 5228 | 5042 | 306 | 0.964 | **8736** | 8797 | 3014 |
| gpu/BoundGuard | 39 | 5229 | 5063 | 313 | 0.968 | **8607** | 8657 | 3019 |
| npu/Static | 39 | 264 | 264 | 3235 | 1.000 | 41174 | 41474 | 20850 |
| npu/Stop-restart | 38 | 3910 | 3910 | 0 | 1.000 | 10389 | 10444 | 4022 |
| npu/Adaptive | 39 | 3705 | 3705 | 232 | 1.000 | **8613** | 8654 | 4384 |
| npu/BoundGuard | 38 | 3791 | 3791 | 209 | 1.000 | **8418** | 8456 | 3866 |

(`N_arrived ≈ served + drop`. miss = late/served.)

#### 4a-3. 이전(무효) 수치와의 차이

| 지표 | 이전(무효) | 보존 후(정본) | 해석 |
|---|---|---|---|
| gpu/Adaptive miss | 0.679 | **0.964** | 격차 소멸 — 이전 값은 backlog를 서비스하지 않아 나온 것 |
| npu/Adaptive miss | 0.661 | **1.000** | 동일 |
| gpu Adaptive mean e2e | 147ms | **3014ms** | 20배. 이전의 "Stop-restart 대비 53배" 격차는 전부 무상 플러시 |
| gpu Adaptive p99 | 5667 | 8736 | tail 악화 |
| Static p99 (GPU) | 24612 | 41798 | 실행시간이 길어져 backlog가 더 쌓임(시간대응 효과) |

**차이의 출처는 단 하나**: 이전에는 적응 기법이 전환 시점에 **약 700프레임(≈5.7초 분량)을 폐기**했고,
지금은 그것을 **서비스한다**. 폐기된 프레임은 완료되지 않으므로 miss·latency 통계에서 사라졌다
(생존편향). 지금은 늦게라도 완료되어 late로 계상된다.

#### 4a-4. 판정 — **게이트 발동**

지시문의 게이트는 "격차가 사라지거나 서열이 뒤집히면 멈추고 보고"였다.

- **서열은 뒤집히지 않았다.** 여전히 `Static ≪ Stop-restart ≲ Adaptive ≈ BoundGuard`.
- **그러나 miss rate 격차는 완전히 사라졌다** (0.963 / 0.964 / 0.968 — 사실상 동일).
- 남는 유의미한 차이는 **p99 약 18%** (8.4–8.7s vs 10.4–10.5s)와 **Static 대비 5배**(8.6s vs 41.8s).
- **"adaptive가 tail을 24s→5s로 낮춘다"는 서술은 성립하지 않는다.** 실제는 Static 41.8s → 적응 8.6s
  (시간대응 기준), Stop-restart 대비로는 10.5s → 8.6s.

**주의(정직 보고)**: Stop-restart의 `drop=0`은 재시작 다운타임 동안 피더가 프레임을 받지 않기 때문이며,
적응 기법은 전환 중에도 수용하다 ~300건을 버린다. 즉 **miss의 분모가 기법마다 다르다.** 이 때문에 miss
단일 수치로 우열을 논하기보다 `served/late/drop` 원수치를 함께 제시해야 한다(위 표에 포함).

### 4b. Q3 재검증 — **결론 불변** (그림 `docs/figures/q3_misprediction.pdf` 갱신)

| 기법 | maxV | lastV | persist | 회복 | (직전) |
|---|---|---|---|---|---|
| Static | 3.65 | 3.42 | 106s | ✗ | 105s |
| Stop-restart | 3.92 | 3.52 | 105s | ✗ | 106s |
| Adaptive | 3.78 | 3.55 | 106s | ✗ | 106s |
| **BoundGuard** | 60.23 | **0.00** | **21s** | **✓** | 20s |

판정 경로도 동일하다 — cand_1 auto-advance, cand_2/3/4 `saturated advance`, cand_5 회복.
이관량이 0–12프레임(`moved=2/3/0/5/11/12`)으로 작아 판정에 영향이 없었다. 감사 §5.3의 예상과 일치.
cand_3은 `slope=−0.71/s`(배출 중으로 보임)인데 view3이 61프레임을 버리는 중이라 포화 가드가 정확히 기각.

### 4c. Q4 재검증 — **결론 강화** (그림 `docs/figures/q4_bounded_envelope.pdf` 갱신)

| 기법 | maxV | lastV | persist | (직전 maxV) |
|---|---|---|---|---|
| Static | 369.4 | 367.1 | 89s | 384.0 |
| Stop-restart | 376.3 | 373.7 | 89s | 375.1 |
| Adaptive | 382.6 | 378.9 | 88s | 374.2 |
| **BoundGuard** | **746.3** | **743.1** | 60s | 739.6 |

- 이전에는 backlog 폐기가 **BoundGuard에 유리**하게 작용했는데도 BoundGuard가 졌다. 보존 후 그 이점이
  사라져 746.3으로 **소폭 더 악화**됐다 — 감사 §5.3의 예상대로 **뒤집히지 않고 강화**됐다.
- 판정 경로: cand_1 auto-advance, cand_2/3/4 전부 `saturated advance`(오커밋 0건).
  이관 실측 `moved=301/300/301/299/300/300/6`.
- **bound 유지**: 탐색 구간(cand_1 진입 → cand_5 적용) = **12.0s ≤ N_cand(T_v+δ) = 20s**. 수정 전과 동일.
- **Q4 판정 불변**: (i)(ii)(iii)(v) 충족, **(iv) 불충족**. "bound는 탐색 시간만 보장하고 종료 배치
  품질은 무보장" — [q4_experiment_report.md](q4_experiment_report.md) §5 그대로.

### 4d. §4b · C3 무회귀 — **통과**

**§4b** (buffer 기본 2, 이관량 ≤2프레임):
| 기법 | GPU persist/lastV | NPU persist/lastV |
|---|---|---|
| Static | 59s / 9.2 | 60s / 9.0 |
| Adaptive | **11s / 0.0** | **12s / 0.0** |
| **BoundGuard** | **11s / 0.0** | **13s / 0.0** |

**BoundGuard = Adaptive 유지 → Q1.5의 "no penalty" 주장 불변.**

**C3 스팟체크** (`slope_gpu_l0.90`, mode 0, B=8000):
- `hot-swapping` 발생 **0회**, headless 워커 **0개** → 이번 수정과 **구조적으로 무관** 확인.
- burst 누적 1936→3259, offload 배수 4582→3865. 정성적 accumulate→drain 재현.
  (절대값은 실행 간 변동이 있으나 mode 0 경로는 변경 대상이 아니다.)
- **`fig:fluid_validation`·Q1.4 무영향 재확인.**

## §5. 논문 반영 판단

### 5.1 `tab:b2`·`fig:b2`에 넣을 최종 수치
**§4a-2 시간대응 표를 정본으로 채택.** 그림은 `docs/figures/b2_metrics.pdf`(miss + p99, 양 플랫폼).
`served/late/drop` 원수치를 표 또는 캡션에 함께 실을 것 — miss의 분모가 기법마다 다르기 때문이다(§4a-4).

### 5.2 Q1.5 서술 방향 — **수정이 필요하다**

Q1.5는 두 주장을 담고 있으므로 분리해야 한다.

| 주장 | 판정 | 근거 |
|---|---|---|
| §4b "well-predicted 케이스에서 BoundGuard는 Adaptive 대비 no penalty" | **유지** | §4d (11s vs 11s, 12s vs 13s) |
| B2 "adaptive가 non-adaptive 대비 miss·tail을 크게 낮춘다" | **약화 — 재서술 필요** | §4a |

재서술 시 사실관계:
- **성립**: `Static` 대비 우위는 크고 명확하다. p99 **41.8s → 8.6s (약 5배)**, 양 플랫폼 동일 경향.
  적응/재시작 계열이 배치를 옮기는 것 자체의 효과다.
- **성립하지 않음**: `Stop-restart` 대비 **miss rate 우위**. 0.963 / 0.964 / 0.968로 사실상 동일하다.
- **약하게 성립**: `Stop-restart` 대비 **tail(p99) 우위 약 18%** (10.5s → 8.6s GPU, 10.4s → 8.4s NPU).
  재시작 다운타임을 피한 만큼으로 해석되며, 이는 hot-swap의 고유 이점이다.
- **폐기해야 할 서술**: "24s→5s". 실측 근거가 없다.

> 즉 논문의 대비 축을 **"적응 vs 비적응"에서 "재배치 vs 고정(Static)"으로 옮기고**, Stop-restart 대비
> 우위는 **tail 한정, 약 18%** 로 정직하게 축소 기술하는 것이 데이터에 부합한다.

### 5.3 남은 주의사항 (논문에 명시 권장)
1. **miss 분모 비대칭**: Stop-restart는 재시작 다운타임 동안 프레임을 수용하지 않아 `drop=0`인 반면,
   적응 기법은 전환 중에도 수용하다 ~300건을 버린다. miss 단독 비교는 오해를 부른다.
2. **B2 하네스 시간 불일치**(§4a-1)는 이번에 시간대응으로 교정했다. 기존 수치는 BoundGuard에만
   18초를 더 준 상태였다.

### 5.4 main.tex
**불변.** 위 수치의 반영은 별도 대화에서 진행한다.

## 무결성
- 변경 파일: `adaptive_deploy.py`, `unified_viewer.py`. legacy/scripts/backup·main.tex 불변.
- 스냅샷: `backup/backlog_preserve_20260724_*/`.
- P1(실행 순서)·P2(mode)·P3(fallback=0, 전 런 확인)·P4(큐 깊이·backlog 거동 전 기법 동일)·
  P6(B·λ 명시)·P7(판정 경로 로깅)·P9(commit-and-stay) 확인.
- **P8 관련 신규 이슈**: B2 하네스의 시간 불일치(§4a-1) — 기존 수치에도 있던 문제. 시간대응본을 정본으로 채택.
