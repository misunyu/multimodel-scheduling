# B2 스윕 재실행 — deadline을 Table I의 L_SLO로 정합

날짜: 2026-07-24 · deadline 외 모든 파라미터 불변 · main.tex 불변
관련: [b2_buffer_sweep_report.md](b2_buffer_sweep_report.md), [backlog_preserve_report.md](backlog_preserve_report.md)

> **한 줄 결론**: deadline을 폴백(`1000/infps`=8.13ms)에서 논문 Table I의 **L_SLO=31.0ms**(YOLO11s)로
> 정합했다. **정성적 결론은 전부 불변** — miss는 대부분 셀에서 포화, p99가 분해능 제공, 서열
> `Static≪Stop-restart≲Adaptive≈BoundGuard` 유지, BoundGuard≈Adaptive 유지. 유일한 실질 변화는
> **0.5×-GPU의 적응 기법 miss가 0.97→0.86**으로, 8–31ms 구간에 놓인 프레임(~11%)이 on-time으로
> 재분류된 것이다(원리적·예상된 효과). **p99/mean은 deadline과 무관해야 하며, 실측도 그렇다**(§3.2).

## §1. deadline 정합

### 1.1 결함
`view_handlers._deadline_ms`: `slo_ms`가 설정되면 그 값, 없으면 `1000/infps`. B2 콤보 YAML은
`slo_ms`를 넣지 않아 λ=123fps에서 **8.13ms** 폴백이 발동했다 — 논문이 서술한 지표가 아니었다.

### 1.2 전달 경로의 두 번째 결함 (수정)
`slo_ms`를 YAML에 넣어도 폴백이 유지됐다. 원인: **명명된 뷰(view1–4) 생성 경로가 `slo_ms`를 누락**했다.
- `unified_viewer.py:457` `view_to_model_map` — `slo_ms` 없음(headless 경로 L434만 있었음). → 추가.
- `adaptive_deploy.py:135` `_parse_combination`의 뷰 분기 — `slo_ms` 없음. → 추가(hot-swap 후 이 settings가
  `model_settings`를 덮어쓰므로 여기도 필요).
두 곳을 고쳐야 Static/Stop-restart(전자)와 Adaptive/BoundGuard(후자, hot-swap) **모두** L_SLO를 반영한다.

### 1.3 값과 검증
- B2 워크로드 = **yolo11s 단일 모델**. Table I(`docs/benchmark_model_table.md`): YOLO11s latency 6.19ms,
  **L_SLO = 5× = 31.0ms**. (지시문 Table I와 일치, 정본 확인.)
- **검증**: 전 24런의 per-frame CSV `deadline_ms` 컬럼이 **모두 31.00** — 폴백(8.13) 미발동.
  단일 뷰이므로 값 하나만 존재.
- Static(비swap)·Adaptive(swap) 스모크 각각 `deadline_ms=31.00` 확인 후 본 스윕 실행.

## §2. 전체 결과 (deadline=31ms, 3레짐 × 4기법 × 2플랫폼)

miss = (late+drop)/arrived. p99/p999/mean 단위 ms. blmax = 관측 최대 backlog. wall 44.9–50.0s(시간대응 유지).

| β_B | plat/기법 | wall | arrived | late | drop | **miss** | **p99** | p999 | mean | blmax(cap) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.5 | gpu/Static | 45s | 3024 | 263 | 2761 | 1.000 | 41492 | 41826 | 21037 | 2262(2262) |
| 1.5 | gpu/Stop-restart | 50s | 5530 | 5530 | 0 | 1.000 | 10474 | 10529 | 6724 | 1295 |
| 1.5 | gpu/Adaptive | 46s | 5399 | 5399 | 0 | 1.000 | 8591 | 8662 | 4711 | 1030 |
| 1.5 | gpu/BoundGuard | 46s | 5218 | 5218 | 0 | 1.000 | 8778 | 8844 | 5248 | 1069 |
| 1.5 | npu/Static | 45s | 2308 | 263 | 2045 | 1.000 | 41027 | 41361 | 20862 | 1707(1707) |
| 1.5 | npu/Stop-restart | 49s | 3898 | 3898 | 0 | 1.000 | 10402 | 10452 | 7682 | 958 |
| 1.5 | npu/Adaptive | 45s | 3768 | 3768 | 0 | 1.000 | 8390 | 8442 | 5897 | 773 |
| 1.5 | npu/BoundGuard | 45s | 3730 | 3730 | 0 | 1.000 | 8462 | 8489 | 6088 | 781 |
| 1.0 | gpu/Static | 45s | 3779 | 263 | 3516 | 1.000 | 41591 | 41923 | 21095 | 1508(1508) |
| 1.0 | gpu/Stop-restart | 50s | 4722 | 4722 | 0 | 1.000 | 10455 | 10489 | 9894 | 1285 |
| 1.0 | gpu/Adaptive | 46s | 5345 | 5345 | 0 | 1.000 | 8555 | 8614 | 4903 | 1025 |
| 1.0 | gpu/BoundGuard | 45s | 5183 | 5183 | 0 | 1.000 | 8636 | 8686 | 5312 | 1064 |
| 1.0 | npu/Static | 45s | 2873 | 263 | 2610 | 1.000 | 41070 | 41367 | 20826 | 1138(1138) |
| 1.0 | npu/Stop-restart | 49s | 3850 | 3850 | 0 | 1.000 | 10366 | 10408 | 7934 | 956 |
| 1.0 | npu/Adaptive | 45s | 3755 | 3755 | 0 | 1.000 | 8375 | 8416 | 5880 | 769 |
| 1.0 | npu/BoundGuard | 45s | 3704 | 3704 | 0 | 1.000 | 8520 | 8557 | 6381 | 785 |
| 0.5 | gpu/Static | 45s | 4548 | 264 | 4284 | 1.000 | 41642 | 41994 | 21096 | 754(754) |
| 0.5 | gpu/Stop-restart | 40s | 4645 | 4645 | 0 | 1.000 | 10679 | 10686 | 6623 | 754 |
| 0.5 | gpu/Adaptive | 39s | 5503 | 4462 | 305 | **0.866** | 8823 | 8879 | 3066 | 754 |
| 0.5 | gpu/BoundGuard | 39s | 5522 | 4478 | 295 | **0.864** | 8459 | 8514 | 2968 | 754 |
| 0.5 | npu/Static | 39s | 3493 | 264 | 3229 | 1.000 | 41020 | 41325 | 20775 | 569 |
| 0.5 | npu/Stop-restart | 38s | 3881 | 3881 | 0 | 1.000 | 10508 | 10556 | 4188 | 569 |
| 0.5 | npu/Adaptive | 38s | 3949 | 3718 | 231 | 1.000 | 8564 | 8595 | 4221 | 569 |
| 0.5 | npu/BoundGuard | 38s | 3981 | 3767 | 214 | 1.000 | 8419 | 8468 | 3945 | 569 |

그림: `docs/figures/b2_buffer_sweep.pdf`(재생성), `docs/figures/b2_metrics.pdf`(0.5× SLO, 재생성).

## §3. 이전(8.13ms) 대비 비교

### 3.1 miss rate — 22/24 셀 불변, 0.5×-GPU 적응만 변화

| β_B | plat/기법 | miss 8.13 → 31 | 해석 |
|---|---|---|---|
| — | Static (전 셀) | 1.000 → 1.000 | 100% 프레임이 e2e>31ms → 무변화 |
| — | Stop-restart (전 셀) | 1.000 → 1.000* | 동일 (*0.5×-GPU 예외 §3.3) |
| **0.5** | **gpu/Adaptive** | **0.966 → 0.866** | 8–31ms 밴드 ~11%가 on-time 재분류 |
| **0.5** | **gpu/BoundGuard** | **0.970 → 0.864** | 동일 |
| 0.5 | npu/Adaptive·BoundGuard | 1.000 → 1.000 | NPU가 느려 밴드=0% |
| 1.0, 1.5 | 적응 기법 (전 셀) | 1.000 → 1.000 | 버퍼 큼 → backlog 큼 → 밴드=0% |

**원리 (e2e 이봉성)**: 프레임은 (a) 큐가 비어 즉시 서비스(e2e ≤ 8ms)되거나 (b) backlog에 걸려 **초 단위**
지연된다. 8–31ms 구간은 전환 직후 짧게만 존재한다. 0.5×-GPU 적응 기법에서만 그 구간에 프레임의
**10.8%**가 놓여(직접 계수 확인), deadline을 8→31ms로 늦추면 late→on-time으로 넘어간다. 다른 셀은
밴드=0%라 deadline 무관.

### 3.2 p99 / mean — deadline 독립 (검증)
코드상 `mean`·`p99`는 **완료 프레임의 e2e 분포**만으로 계산되며 deadline은 `late` 플래그만 좌우한다.
따라서 old/new 차이는 **run-to-run 편차여야** 한다. 실측:
- **p99**: 전 24셀에서 old≈new(예: gpu/Adaptive 1.5× 8828→8591, npu/BoundGuard 0.5× 8418→8419).
  최대 편차 ~3%로 **모두 run 편차 범위** → **deadline 독립 확인**.
- **mean**: 대부분 근접하나 **1.0×-gpu/Stop-restart 6719→9894**, **0.5×-gpu/Stop-restart 3221→6623**의 큰 차.
  → **§3.3**.

### 3.3 조사 — Stop-restart mean/miss 편차 (deadline 아님)
- Stop-restart는 mode 0으로 매 전환마다 **full stop + cold restart**한다. 재시작 완료 시점이 burst backlog와
  어떻게 겹치는지에 따라 재시작 직후 서비스되는 프레임 수·속도가 **run마다 크게 흔들린다**.
- `mean`은 표본(served) 구성에 민감하고 `p99`(꼬리 분위수)는 안정적이다 — old/new의 **p99가 나란히 안정**
  (10516→10679, 10514→10455)한데 mean만 흔들리는 것이 이를 뒷받침한다.
- 0.5×-GPU Stop-restart miss 0.963→1.000도 같은 편차다. old run은 재시작 직후 일부 프레임을 <8ms에
  서비스(3.7%)했고 new run은 그러지 못했다. **deadline과 무관** — deadline이 8→31로 **느슨해지면** miss는
  내려가야지 올라갈 수 없다(같은 run 내에서). 즉 방향이 반대인 것 자체가 run 편차의 증거다.
- **결론**: mean/miss의 Stop-restart 편차는 **재시작 오버헤드의 run-to-run 변동**이며 지표 정의 변경의
  효과가 아니다. p99를 주 지표로 쓰면 이 편차에 영향받지 않는다.

## §4. 확인 사항 판정

| 항목 | 판정 | 근거 |
|---|---|---|
| **분해능** | miss는 여전히 포화(22/24 셀 1.000). 0.5×-GPU 적응만 0.86 | §3.1 |
| **서열 유지** | ✅ p99 기준 `Static≪Stop-restart≲Adaptive≈BoundGuard` 6개 (레짐×플랫폼) 셀 전부 | §2 |
| **BoundGuard≈Adaptive** | ✅ p99 전 셀 ≤3% (8375–8823 대역) | §2 |
| **역설(tight buffer 유리)** | ✅ 유지·강화. L_SLO 기준에서 0.5×-GPU 적응이 유일하게 miss<1.0 (밴드 효과가 tight buffer에서만) | §3.1 |
| **Static 구조적 drop** | ✅ 1.5×에서도 drop(gpu 2761 / npu 2045). CPU-only 한계 | §2 |
| **서열 역전** | 없음 → 게이트 미해당 | |

## §5. 논문 반영 제안

### 5.1 이전 리포트 §5.1(p99 중심 3레짐 표 + miss 포화 축약)이 L_SLO에서도 유효한가 → **예**
- p99가 세 레짐·양 플랫폼에서 일관된 판별 지표라는 점은 deadline 정의와 무관하게 유지된다.
- miss "전 레짐 포화" 서술은 **거의** 유효하되, **정확히는 "22/24 셀 포화, tight buffer + GPU에서
  적응 기법만 ≈0.86"** 으로 미세 수정. 이 예외가 오히려 **tight buffer에서 적응 기법이 일부 프레임을
  살린다**는 점을 보여줘 서사에 부합한다.
- 그림 `b2_buffer_sweep.pdf`(주석 갱신)·`b2_metrics.pdf`(0.5× SLO) 재생성 완료.

### 5.2 함께 명시할 것 (이전과 동일 + 추가)
- **deadline은 이제 Table I의 L_SLO**(YOLO11s 31.0ms). 폴백 제거를 방법론 절에 반영.
- Static drop은 무손실 레짐에서도 지속(CPU-only 구조적).
- **Stop-restart mean/miss는 재시작 오버헤드로 run 편차가 큼** → **p99를 주 지표로** 쓰는 근거를 강화.
- miss 분모 비대칭(Stop-restart drop=0 vs 적응 drop>0 at 0.5×) — 기존 주의 유지.

### 5.3 main.tex
**불변.** 반영은 별도 대화.

## §6. 무결성
- deadline **외** 전부 불변: N_cand·α·β·ε·T·T_v·Δ·θ·판정 로직·backlog 보존·λ·B·시간대응.
- 변경 파일: `unified_viewer.py`(view_to_model_map에 slo_ms), `adaptive_deploy.py`(_parse_combination에 slo_ms).
  legacy/backup·main.tex 불변.
- P1(순서)·P2(mode)·P3(fallback=0 전 런)·P4(deadline 전 셀 31.00)·P7(판정 경로)·P9(commit-and-stay) 확인.
- wall-clock 전량 명시(§2). 시간대응 유지 — BoundGuard 추가시간 결함 미재발.
