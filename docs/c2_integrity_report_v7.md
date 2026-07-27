# C2/Q6 무결성 v7 (최종) — Q5 출처 확인 · Q1.3 wall-clock 확정 · 수치 확정표

날짜: 2026-07-26 · 재실험 없음(코드 판독+재집계) · main.tex 불변 · 파라미터 불변
선행: v2 §7, [c2_integrity_report_v6.md](c2_integrity_report_v6.md), [metric_definitions.md](metric_definitions.md)

> **판정 요약** (작업 22~25):
> - **작업 22**: Q5 774의 `15.2`=샘플수 persist(pre-burst stable 위반 포함해 과다), `14.2`=burst→적용(wall-clock).
>   Q5는 cand_2 즉시 회복(drain≈0)이라 **수렴=T_valid=14.2±0.4 단일값**. 15.2는 샘플카운트 아티팩트 → 774행
>   두 값을 하나로 재구성. (Q3와 동형 + 즉시회복 특수성)
> - **작업 23**: Q1.3 회복 3기법 wall-clock = **11–13s**(SR 10.8–12.6·Ad 11.0–11.8·BG 11.2–12.0). 샘플 "9–12"에서
>   SR만 −2s(재시작 누락), BG/Ad는 누락 0이라 동일. Q3와 같은 규약(wall-clock)으로 통일.
> - **작업 24**: v6 19-2의 Q3 wall '114'는 full-CSV(stable 포함). from-burst dur = 106.8/106.8/107.0/83.0(v3 정합).
>   누락 결론(0 vs 5) 불변.
> - **작업 25**: **`main.tex` 수치 확정표** 완성(아래 §25). 확정표에 없는 수치는 논문에 넣지 않는다.

## 작업 22 — Q5 `15.2`/`14.2` 출처 (작업 15 동일 방식)

`b2_analyze.py`: `persist=sum(1 for x>ε)`(샘플수), `Tvalid=rec_apply−burst`(타임스탬프). b2rep_q5 런별:

| rep | [orig]persist(샘플수) | [orig]Tvalid(burst→적용) | [mine]persist_burst(burst→V≤ε) | 누락 | recov |
|---|---|---|---|---|---|
| 1 | 15 | 15.0 | 15.0 | 7.0 | cand_2 |
| 2–5 | 15,15,15,16 | 14.0×4 | 14.0×4 | 5–6 | cand_2 |
| **평균±SD** | **15.2±0.40** | **14.2±0.40** | **14.2±0.40** | 5.8 | cand_2 |

- **15.2 = 샘플수 persist**(main.tex 774), **14.2 = burst→적용**(main.tex T_valid). **판정 22-3 1행**(Q3 동형).
- 특수성: Q5는 cand_2에서 **즉시 회복**(drain≈0) → burst→적용 = burst→V≤ε = 14.2. 즉 **수렴과 T_valid가 같은
  회복 사건**. 15.2가 14.2보다 큰 이유는 **pre-burst stable 위반 샘플**(Q5 stable V≈7, 작업14)이 persist 카운트에
  포함되기 때문(과소집계 아니라 과다). 누락 5.8(LLM 전환)은 회복 시점이라 위반 카운트에 기여 안 함.
- **조치**: 774행을 **단일값 "converges in 14.2±0.4 s (from burst)"**로 재구성. 15.2(샘플수, stable 포함) 폐기.
  t0 기준(21.0) 미사용(작업17).

## 작업 23 — Q1.3 회복 3기법 wall-clock 확정

| plat | 기법 | wall-clock persist | 샘플수(현행) | 누락 | 회복 |
|---|---|---|---|---|---|
| gpu | Stop-restart | 12.6±2.4 | 10.8 | 1.0 | 5/5 |
| gpu | Adaptive | 11.0±0.6 | 11.0 | 0.0 | 5/5 |
| gpu | BoundGuard | 11.2±0.7 | 11.2 | 0.0 | 5/5 |
| npu | Stop-restart | 10.8±1.5 | 9.4 | 1.8 | 5/5 |
| npu | Adaptive | 11.8±0.4 | 11.8 | 0.0 | 5/5 |
| npu | BoundGuard | 12.0±0.0 | 12.0 | 0.0 | 5/5 |
| both | Static | —(미회복) | 59.4 | 0.0 | 0/5 |

- **BG/Adaptive는 누락 0 → 샘플수=wall-clock.** Stop-restart만 재시작 pause로 −2s 과소집계. → 회복 3기법
  wall-clock 범위 **11–13s**(샘플 "9–12" 대체). **Static은 미회복 → "회복 없음"(v3 §8-4), 값 교체 아님.**

## 작업 24 — v6 19-2 표 참조점 정정
- v6의 Q3 wall '114/90'은 full-CSV(stable 포함). **from-burst dur = Static/SR 106.8·Ad 107.0·BG 83.0**(v3 작업8 정합).
- 누락(전환 변형만: BG 5, 나머지 0) 결론 불변. "114"는 Q5 baseline 복사가 아니라 Q3 full-CSV wall이었음(우연히 근접).

## §25. `main.tex` §IV 수치 확정표 (copy-ready)

> 정의 표기: **[wc]**=wall-clock 타임스탬프, **[cnt]**=V>ε 샘플수, **[int]**=시간가중 적분. 전부 5회(명시 예외).

### Q3 (720행) — CPU–GPU 오예측
| 항목 | 확정 수치 | 정의 | 출처 |
|---|---|---|---|
| BoundGuard persist | **25.4±1.4 s** [wc] (= search 22.6 + drain 2.8) | t0→V≤ε | b2rep_q3 5런 |
| BoundGuard T_valid | **24.6±0.5 s** [wc] | burst→회복적용 | b2rep_q3 5런 |
| 상한 대조 | 관측 **search ≤ 23 s** vs worst-case **24–26 s**(런별 k·δ) | — | 35런(b2rep5+C2 30) |
| baseline (Static/SR/Adaptive) | **회복 없음** (lastV 2.7–2.9, 창 ~107 s) | censored | b2rep_q3 |

### Q4 (749행) — 유계 탐색
| 항목 | 확정 수치 | 정의 | 출처 |
|---|---|---|---|
| 첫위반→reversion | **24 s = bound 24 s (tight)** (sample-gap 0) | wc | q4_bsf_out(논문) |
| C2 재현 | 22–23 s (5런) | wc | c2_final2/q4 |

### Q5 (774행) — CPU–NPU
| 항목 | 확정 수치 | 정의 | 출처 |
|---|---|---|---|
| 수렴(=T_valid) | **14.2±0.4 s** [wc] (단일 회복사건) | burst→V≤ε(=적용, drain≈0) | b2rep_q5 5런 |
| ~~persist 15.2~~ | **폐기** (샘플수, pre-burst stable 위반 포함) | cnt | — |
| baseline | **회복 없음** (lastV ~33, 창 ~107 s) | censored | b2rep_q5 |

### Q1.3 (630행) — 정상상태/failure persistence
| 항목 | 확정 수치 | 정의 | 출처 |
|---|---|---|---|
| 회복 3기법 persistence | **11–13 s** [wc] (SR 10.8–12.6·Ad 11.0–11.8·BG 11.2–12.0) | t0→V≤ε | b3rep gpu+npu |
| Static | **회복 없음** (lastV 9.0, 창 58.4 s) | censored | b3rep |
| 누적위반 ∫V 배수 | **6.2× (gpu) / 6.3× (npu)** [int] (창 정합 시 ~5.7×; **Static 창 더 짧아 보수적**) | Σv·dt | b3rep |

### Q5·§4b 공통 문장 (stable 위반, v6 21-3)
> *"In Q5/§4b the all-CPU stable phase is itself infeasible for the workload (V≈7>ε before the burst). By
> experimental design the controller is enabled at the burst injection; recovery times are reported from that
> point, and the pre-burst violation reflects the stable placement, not detection latency."*

### 관측 창·t0 기준 (v4 작업14)
| 실험 | stable V | t0 위치 | 창 |
|---|---|---|---|
| Q3 | 0.00 | burst+2s | ~107s |
| Q5 | 6.9(>ε) | burst−6.8s(stable) | ~107s |
| §4b | ~6(>ε) | burst−7s(stable) | 고정100s |
| Q4 | ~0 | burst+1s | 고정100s |

### Q6 완성표 (v2 §7 정정본, 5변형 × 3시나리오, 5회)
| 변형 | §4b 회복 | Q3 회복 | Q4 회복 | 비고(정정 반영) |
|---|---|---|---|---|
| BoundGuard | 5/5 | 5/5 | 0/5 유계 | Σδ_vis §4b 680·Q3 1719 ms |
| Adaptive | 5/5 | **0/5** | 0/5 유계 | baseline |
| A. no-dwell | **4/5**(hs 9.4) | **1/5** | 0/5 유계 | **§4b 4/5(1/5 아님)**; Σδ_vis §4b **3919**·Q3 **3496**(BG의 2.4×) |
| B. re-invoke | 5/5 | **0/5** | 0/5 유계 | ≡Adaptive(격자 모서리) |
| C. hybrid | 5/5 | **0/5** | 0/5 유계 | **Q3 0/5(1/5 아님)** |
- lastV: Q3 A 19.9±9.9·B 1.25·C 1.19·Adaptive 1.14; Q4 A 97.6±52.6(최악)·BG 54.8. hotswaps: Q3 BG 7·A 8.8.
- 상세·δ 전열 → v2 §7(정정 주석 포함).

## 무결성
- 재실험 없음(작업 22~25 전부 코드·로그 판독+재집계). 알고리즘·파라미터·main.tex·legacy/backup 불변. 코드 무변경.
- v7 = 확정표 라운드. v2~v6 판정 유지, v6 참조점(114)·Q5 15.2·Q1.3 9–12만 wall-clock으로 정정. **§II bound 불변.**
