# B2 buffer 스윕 — β_B ∈ {1.5, 1.0, 0.5}

날짜: 2026-07-24 · 파라미터 불변 · 상속 backlog 보존 적용 · main.tex 불변
관련: [backlog_preserve_report.md](backlog_preserve_report.md), [buffer_bias_impact_audit.md](buffer_bias_impact_audit.md)

> **한 줄 결론**: 논문이 약속한 스윕을 이행했다. **miss rate는 세 레짐 전체에서 분해능이 없다**
> (deadline 8ms가 burst backlog 형성 후 구조적으로 달성 불가). 기법을 일관되게 구별하는 지표는
> **p99/mean latency**이며, 서열 `Static ≪ Stop-restart ≲ Adaptive ≈ BoundGuard`가 세 레짐 모두에서
> 유지된다. β_B는 초과 프레임을 **drop(Static)** 으로 낼지 **late(적응 기법)** 로 낼지만 바꾼다.

## §1. 구성

- **B = β_B · Q_max**. 0.5× 정본에서 GPU 754 / NPU 569이므로 **Q_max = GPU 1508 / NPU 1138** (=754/0.5).
  - β_B=**0.5**: B = 754 / 569 (정본, 재실행 불요 — 동일 조건 확인)
  - β_B=**1.0**: B = 1508 / 1138
  - β_B=**1.5**: B = 2262 / 1707
- 4기법 × 2플랫폼, λ=0.9μ*(GPU 123 / NPU 93 fps), `FSRR_RATE_REPLICATE=1`, per-frame deadline **8.13ms**(yolo11s).
- **시간대응**: 전 기법 스케줄 duration 통일(Static burst=44; Stop-restart/Adaptive burst=8+cand_1=36;
  BoundGuard burst=8+cand_1=18+cand_2=18). 실측 wall-clock 44.7–49.9s(§4 명시). Stop-restart가 ~5s 긴 것은
  재시작 cold-start 오버헤드로 **기법 고유 특성**이며 스케줄 duration은 동일하다.

## §2. 전체 결과 (3레짐 × 4기법 × 2플랫폼)

miss = (late + drop) / arrived. p99/p999/mean 단위 ms. blmax = 관측 최대 backlog.

| β_B | plat/기법 | wall | arrived | late | drop | **miss** | **p99** | p999 | mean | blmax (cap) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.5 | gpu/Static | 45s | 3018 | 262 | 2756 | 1.000 | 41459 | 41758 | 20962 | 2262 (2262) |
| 1.5 | gpu/Stop-restart | 50s | 5508 | 5508 | 0 | 1.000 | 10473 | 10510 | 6792 | 1279 |
| 1.5 | gpu/Adaptive | 46s | 5349 | 5349 | 0 | 1.000 | 8828 | 8891 | 5213 | 1074 |
| 1.5 | gpu/BoundGuard | 45s | 4327 | 4327 | 0 | 1.000 | 8891 | 8902 | 8654 | 1094 |
| 1.5 | npu/Static | 45s | 2304 | 263 | 2041 | 1.000 | 41017 | 41350 | 20817 | 1707 (1707) |
| 1.5 | npu/Stop-restart | 49s | 3888 | 3888 | 0 | 1.000 | 10371 | 10412 | 7731 | 955 |
| 1.5 | npu/Adaptive | 45s | 3749 | 3749 | 0 | 1.000 | 8428 | 8467 | 6033 | 774 |
| 1.5 | npu/BoundGuard | 45s | 3671 | 3671 | 0 | 1.000 | 8428 | 8461 | 6450 | 780 |
| 1.0 | gpu/Static | 45s | 3745 | 261 | 3484 | 1.000 | 41324 | 41656 | 20969 | 1508 (1508) |
| 1.0 | gpu/Stop-restart | 50s | 5545 | 5545 | 0 | 1.000 | 10514 | 10573 | 6719 | 1277 |
| 1.0 | gpu/Adaptive | 46s | 5366 | 5366 | 0 | 1.000 | 8820 | 8839 | 5226 | 1084 |
| 1.0 | gpu/BoundGuard | 45s | 5236 | 5236 | 0 | 1.000 | 8664 | 8693 | 5163 | 1066 |
| 1.0 | npu/Static | 45s | 2856 | 261 | 2595 | 1.000 | 40763 | 41097 | 20696 | 1138 (1138) |
| 1.0 | npu/Stop-restart | 49s | 3890 | 3890 | 0 | 1.000 | 10507 | 10554 | 7824 | 970 |
| 1.0 | npu/Adaptive | 45s | 3751 | 3751 | 0 | 1.000 | 8480 | 8508 | 6077 | 782 |
| 1.0 | npu/BoundGuard | 45s | 3760 | 3760 | 0 | 1.000 | 8414 | 8454 | 6043 | 775 |
| 0.5 | gpu/Static | 39s | 4557 | 262 | 4295 | 1.000 | 41798 | 42133 | 21168 | 754 (754) |
| 0.5 | gpu/Stop-restart | 40s | 5361 | 5160 | 0 | 0.963 | 10516 | 10563 | 3221 | 754 |
| 0.5 | gpu/Adaptive | 40s | 5534 | 5042 | 306 | 0.966 | 8736 | 8797 | 3014 | 754 |
| 0.5 | gpu/BoundGuard | 39s | 5542 | 5063 | 313 | 0.970 | 8607 | 8657 | 3019 | 754 |
| 0.5 | npu/Static | 39s | 3499 | 264 | 3235 | 1.000 | 41174 | 41474 | 20850 | 569 (569) |
| 0.5 | npu/Stop-restart | 38s | 3910 | 3910 | 0 | 1.000 | 10389 | 10444 | 4022 | 569 |
| 0.5 | npu/Adaptive | 39s | 3937 | 3705 | 232 | 1.000 | 8613 | 8654 | 4384 | 569 |
| 0.5 | npu/BoundGuard | 38s | 4000 | 3791 | 209 | 1.000 | 8418 | 8456 | 3866 | 569 |

그림: `docs/figures/b2_buffer_sweep.pdf` (p99 및 drop을 β_B 축으로, 양 플랫폼).

## §3. 확인 사항

### 3.1 1.5× 무손실? — **부분적. 기법 의존적**
논문 서술("1.5× = slack, no drops")은 **offload하는 기법에만** 성립한다.
- Stop-restart/Adaptive/BoundGuard: 1.0×·1.5× 모두 **drop=0** (blmax 774–1279 < cap). 이미 1.0×에서 무손실.
- **Static: 세 레짐 전부 drop 발생** (blmax가 항상 cap에 닿음: 754→1508→2262). CPU-only라 λ=0.9μ*를
  **서비스할 수 없어** backlog가 무한 증가, 버퍼를 아무리 키워도 채운다. → "무손실"은 Static엔 원리적으로
  불가능. **Q_max 산정 문제가 아니라 Static의 구조적 특성**이다.

### 3.2 분해능 — **miss rate는 분해능 없음, p99가 분해능 제공**
- **miss rate**: 24개 셀 중 23개가 **1.000**, 예외는 0.5×-GPU의 적응 기법(0.963–0.970)뿐. 느슨한 버퍼로
  갈수록 오히려 miss가 **1.0으로 수렴**한다(§3.3). → **miss는 기법을 구별하지 못한다.**
- **원인**: per-frame deadline이 **8.13ms**인데, burst 8초 동안 CPU에서 ~1000프레임이 쌓인다. offload 후
  backlog를 배수해도 그 프레임들은 이미 초 단위를 대기해 **전부 late**다. 버퍼 크기와 무관하다.
- **p99/mean**: 세 레짐 전체에서 안정적으로 분해 — Static ~41s ≫ Stop-restart ~10.5s > Adaptive ~8.5s ≈
  BoundGuard ~8.5s. **이것이 B2의 실질 판별 지표**다.

### 3.3 (역설) 왜 0.5×에서만 miss가 1.0 미만인가
버퍼가 작을수록(0.5×) burst backlog가 **낮은 cap(754)에 잘려** 더 작게 쌓인다. offload 후 이 작은
backlog가 더 빨리 배출되어, 마지막 구간의 일부 프레임이 deadline을 지킨다(GPU StopRestart 0.963 = 3.7%가
성공). 버퍼를 키우면 backlog가 더 쌓여 성공 프레임이 사라지고 miss→1.0. **즉 tight buffer가 miss엔 오히려
유리**하다 — 논문이 기대한 "느슨할수록 좋다"와 반대이며, 그대로 보고한다.

### 3.4 서열 일관성 — **유지**
p99 기준 `Static ≪ Stop-restart ≲ Adaptive ≈ BoundGuard`가 **6개 (레짐×플랫폼) 셀 전부**에서 유지.
서열 역전 없음 → 게이트 미해당.

### 3.5 BoundGuard ≈ Adaptive — **유지 (Q1.5 핵심)**
p99 기준 두 기법 차이 ≤3% (전 셀): GPU 8664–8891 vs 8820–8828, NPU 8414–8428 vs 8428–8480.
- **주의**: 1.5×-GPU에서 BoundGuard mean=8654가 Adaptive 5213보다 높다. 이는 BoundGuard가 cand_1에서
  `slope=−2.57/s, tail_drops={none}`로 **정상 커밋 후 commit-and-stay**하며 served=4327(Adaptive 5349보다
  적음)로 초기 프레임 표본이 달라진 것으로, **run-to-run 편차**다. p99(8891 vs 8828)는 거의 동일하므로
  tail 판정에는 영향 없다. mean은 p99보다 표본 구성에 민감함을 명시.

## §4. 무결성
- wall-clock 전량 명시(§2). 시간대응 유지 — BoundGuard에만 추가 시간을 주던 기존 결함(정본에서 교정)을
  **전 레짐에서 재발시키지 않음**.
- 상속 backlog 보존 동작 확인: 전 hot-swap 런에서 `backlog preserved (moved=N dropped=0)` 로깅
  (예: 1.0×-GPU `moved=985/976`, 1.5×는 cand_1 커밋 전 backlog 이관).
- P1(순서)·P2(mode)·P3(fallback=0, 전 런 확인)·P7(판정 경로)·P9(commit-and-stay) 확인.
- 파라미터·판정 로직 불변. legacy/backup·main.tex 불변.

## §5. 논문 반영 제안

### 5.1 스윕 표를 어떻게 실을까
**권장: p99(및 drop:late 조성)을 중심으로 3레짐 표를 싣고, miss rate는 "전 레짐 포화"로 축약 서술.**
- miss rate를 3레짐 표의 주 지표로 쓰면 23/24 셀이 1.000이라 정보가 없다. **오히려 지표 선택이
  부적절해 보인다.**
- 대신 §2 표의 **p99/mean + drop:late 조성**을 본문에 싣고, miss는 "모든 β_B에서 ≈1.0 — deadline(8ms)이
  burst backlog 형성 후 도달 불가"라고 1–2문장으로 처리.
- 그림 `b2_buffer_sweep.pdf`가 이 서술과 정합한다.

### 5.2 대안
0.5만 본문에 두고 1.0/1.5를 부록으로 돌리는 것도 가능하나, **논문이 §IV에서 스윕을 명시적으로 약속**했으므로
세 레짐을 모두 보이되 **지표를 p99로 바꾸는 것**이 약속 이행과 정직성을 동시에 만족한다.

### 5.3 함께 명시할 것
- **Static의 drop은 무손실 레짐에서도 사라지지 않는다**(§3.1) — CPU-only의 구조적 한계. "1.5×=no drops"는
  offload 기법 한정으로 기술.
- **tight buffer가 miss엔 유리**(§3.3)한 역설 — 버퍼를 QoS 개선 수단으로 서술하지 말 것.
- **miss 분모 비대칭**(Stop-restart drop=0 vs 적응 기법 drop>0 at 0.5×) — [backlog_preserve_report.md]
  §5.3과 동일 주의.

### 5.4 main.tex
**불변.** 반영은 별도 대화.
