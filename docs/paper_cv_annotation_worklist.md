# `%!cv` 주석 부착 목록 (v36 작업 190 · v37 작업 196 확장)

**이 목록은 저장소 밖 정본에 적용한다.** `paper/main.tex` 는 배포본이며 편집하면
`scripts/check_paper.py` 의 해시 검사가 실패한다 — 그것이 의도된 동작이다.

주석 18개 · 추적 값 86건(고유 74건).
대상 `main.tex` sha256 `0c7c3a1c0cdf2d65…`

적용:

```bash
python3 scripts/apply_cv_annotations.py --tex /경로/정본/main.tex
```

각 항목의 `line_sha256` 가 대상 줄과 일치할 때만 붙는다. 줄 번호는 편집으로 밀리고,
엉뚱한 줄에 붙은 주석은 **그 문장이 하지도 않은 주장에 값을 매다는** 조용한 오염이 된다.

---

### L365 — §III — what a hot-swap costs in memory (v38/v39)

```
\textcolor{navyblue}{We measured what it costs. Two kinds of transition occur in our syste…
```

줄 끝에 붙일 것:

```latex
%!cv hs_vision_vram_incr_max,hs_gen_rss_incr_max,hs_gen_rss_headroom
```

| id | 확정표 값 |
|---|---|
| `hs_vision_vram_incr_max` | 574.0 MiB |
| `hs_gen_rss_incr_max` | 3099.0 MiB |
| `hs_gen_rss_headroom` | 18.1 x |

### L558 — Q1 — recovery totals, lead, and the part of the lead that precedes activation

```
\textcolor{navyblue}{Recovery completes in $30.4\pm0.5$\,s on CPU--GPU and $27.2\pm0.4$\,s…
```

줄 끝에 붙일 것:

```latex
%!cv q3_bg_persist,q5_bg_persist,q3_lead,q5_lead,q5_lead_pre_activation
```

| id | 확정표 값 |
|---|---|
| `q3_bg_persist` | 30.4 s |
| `q5_bg_persist` | 27.2 s |
| `q3_lead` | 9.0 s |
| `q5_lead` | 22.4 s |
| `q5_lead_pre_activation` | 9.6 s |

### L580 — Q1.1 — steady-state throughput, inference latency, CPU occupancy

```
\emph{Steady state.} We hold the workload in a feasible state throughout ($V(t)=0$, no rec…
```

줄 끝에 붙일 것:

```latex
%!cv q11_gpu_boundguard_throughput,q11_gpu_boundguard_infer_ms,q11_gpu_static_infer_ms,q11_npu_adaptive_infer_ms,q11_npu_boundguard_infer_ms,q11_gpu_boundguard_cpu,q11_gpu_static_cpu,q11_npu_static_cpu,q11_npu_adaptive_cpu
```

| id | 확정표 값 |
|---|---|
| `q11_gpu_boundguard_throughput` | 12.085 fps |
| `q11_gpu_boundguard_infer_ms` | 2.381 ms |
| `q11_gpu_static_infer_ms` | 2.561 ms |
| `q11_npu_adaptive_infer_ms` | 2.772 ms |
| `q11_npu_boundguard_infer_ms` | 2.887 ms |
| `q11_gpu_boundguard_cpu` | 16.496 percent |
| `q11_gpu_static_cpu` | 16.676 percent |
| `q11_npu_static_cpu` | 17.437 percent |
| `q11_npu_adaptive_cpu` | 17.72 percent |

### L601 — Q1.2 — detection lag and recovery latency across the window sweep

```
\textcolor{navyblue}{Figure~\ref{fig:detection_sensitivity} reports four quantities. No se…
```

줄 끝에 붙일 것:

```latex
%!cv q12_detect_lat_T1_eps1.0,q12_detect_lat_T3_eps1.0,q12_detect_lat_T5_eps1.0,q12_recover_lat_T1_eps1.0,q12_recover_lat_T5_eps1.0
```

| id | 확정표 값 |
|---|---|
| `q12_detect_lat_T1_eps1.0` | 0.64 s |
| `q12_detect_lat_T3_eps1.0` | 2.0 s |
| `q12_detect_lat_T5_eps1.0` | 4.0 s |
| `q12_recover_lat_T1_eps1.0` | 17.96 s |
| `q12_recover_lat_T5_eps1.0` | 21.84 s |

### L603 — Q1.2 — the T=1 vs T=3 comparison the section's conclusion turns on

```
Our configuration ($T=3$\,s, $\epsilon=1.0$) sits in the conservative corner of this space…
```

줄 끝에 붙일 것:

```latex
%!cv q12_detect_lat_T1_eps1.0,q12_detect_lat_T3_eps1.0,q12_recover_lat_T1_eps1.0,q12_recover_lat_T3_eps1.0
```

| id | 확정표 값 |
|---|---|
| `q12_detect_lat_T1_eps1.0` | 0.64 s |
| `q12_detect_lat_T3_eps1.0` | 2.0 s |
| `q12_recover_lat_T1_eps1.0` | 17.96 s |
| `q12_recover_lat_T3_eps1.0` | 20.24 s |

### L626 — Q1.3 — Static terminal V, cumulative violation, persistence range, ratios

```
Figures~\ref{fig:q13-persistence} and~\ref{fig:q13-cumulative} report both metrics for the…
```

줄 끝에 붙일 것:

```latex
%!cv q13_static_lastV,q13_gpu_static_cumV,q13_npu_static_cumV,q13_gpu_bg_persist,q13_gpu_sr_persist,q13_gpu_bg_cumV,q13_gpu_sr_cumV,q13_gpu_cumV_ratio,q13_npu_cumV_ratio
```

| id | 확정표 값 |
|---|---|
| `q13_static_lastV` | 13.75 V |
| `q13_gpu_static_cumV` | 1279 V*s |
| `q13_npu_static_cumV` | 1295 V*s |
| `q13_gpu_bg_persist` | 13.2 s |
| `q13_gpu_sr_persist` | 15.2 s |
| `q13_gpu_bg_cumV` | 147 V*s |
| `q13_gpu_sr_cumV` | 162 V*s |
| `q13_gpu_cumV_ratio` | 8.3 x |
| `q13_npu_cumV_ratio` | 8.4 x |

### L628 — Q1.3 — the search that is lead, and the no-dwell variant's counts

```
We do not read the small differences among stop-and-restart, Adaptive hot-swap, and BoundG…
```

줄 끝에 붙일 것:

```latex
%!cv q13_gpu_bg_search,q4b_a_recoveries,q4b_a_hotswaps,q4b_bg_hotswaps
```

| id | 확정표 값 |
|---|---|
| `q13_gpu_bg_search` | 10.8 s |
| `q4b_a_recoveries` | 2 of5 |
| `q4b_a_hotswaps` | 15 swaps |
| `q4b_bg_hotswaps` | 6 swaps |

### L645 — Q1.4 — sweep regression slopes against the predicted 1.0

```
The sweep separates the model's two halves, and they do not fare the same. Accumulation is…
```

줄 끝에 붙일 것:

```latex
%!cv c3_gpu_dslope_dlambda,c3_npu_dslope_dlambda
```

| id | 확정표 값 |
|---|---|
| `c3_gpu_dslope_dlambda` | 1.007 1 |
| `c3_npu_dslope_dlambda` | 1.083 1 |

### L647 — Q1.4 — the three service rates bracketing the effective drain rate

```
The reason is the one Eq.~(\ref{eq:tstable}) now names. Inverting the measurements for the…
```

줄 끝에 붙일 것:

```latex
%!cv c3_gpu_mu_star_sat,c3_npu_mu_star_sat,c3_gpu_mu_star_raw,c3_npu_mu_star_raw,c3_gpu_mu_star_eff,c3_npu_mu_star_eff
```

| id | 확정표 값 |
|---|---|
| `c3_gpu_mu_star_sat` | 79.0 fps |
| `c3_npu_mu_star_sat` | 65.0 fps |
| `c3_gpu_mu_star_raw` | 144.0 fps |
| `c3_npu_mu_star_raw` | 231.3 fps |
| `c3_gpu_mu_star_eff` | 112.7 fps |
| `c3_npu_mu_star_eff` | 96.7 fps |

### L685 — Q1.5 — buffer sweep: drops fall, miss rate barely moves

```
Two observations follow, and both are stable across the buffer sweep of Figure~\ref{fig:b2…
```

줄 끝에 붙일 것:

```latex
%!cv q15_gpu_static_b0.5_drop,q15_gpu_static_b1.5_drop,q15_gpu_static_b0.5_miss,q15_gpu_static_b1.5_miss
```

| id | 확정표 값 |
|---|---|
| `q15_gpu_static_b0.5_drop` | 1979 frames |
| `q15_gpu_static_b1.5_drop` | 958.0 frames |
| `q15_gpu_static_b0.5_miss` | 0.6316 ratio |
| `q15_gpu_static_b1.5_miss` | 0.5332 ratio |

### L687 — Q1.5 — the reconfigure-vs-stay-put division

```
Second, both metrics separate the methods, and they draw the same line: the primary divisi…
```

줄 끝에 붙일 것:

```latex
%!cv q15_gpu_static_b0.5_miss,q15_gpu_static_b0.5_p99,q15_npu_adaptive_b0.5_p99,q15_gpu_stoprestart_b0.5_p99
```

| id | 확정표 값 |
|---|---|
| `q15_gpu_static_b0.5_miss` | 0.6316 ratio |
| `q15_gpu_static_b0.5_p99` | 33656.0 ms |
| `q15_npu_adaptive_b0.5_p99` | 6849.0 ms |
| `q15_gpu_stoprestart_b0.5_p99` | 10050.0 ms |

### L689 — Q1.5 — stop-restart vs in-place tail

```
Within the reconfiguring group a second, smaller division appears, and it is the one this …
```

줄 끝에 붙일 것:

```latex
%!cv q15_gpu_stoprestart_b0.5_p99,q15_gpu_boundguard_b0.5_p99,q15_gpu_adaptive_b0.5_p99
```

| id | 확정표 값 |
|---|---|
| `q15_gpu_stoprestart_b0.5_p99` | 10050.0 ms |
| `q15_gpu_boundguard_b0.5_p99` | 7426.0 ms |
| `q15_gpu_adaptive_b0.5_p99` | 7571.0 ms |

### L701 — Q2 / prediction-match — the recovery times that are indistinguishable

```
\textcolor{navyblue}{This scenario and the prediction-match control of Q6 are the same con…
```

줄 끝에 붙일 것:

```latex
%!cv q4b_adaptive_persist,q4b_c_persist
```

| id | 확정표 값 |
|---|---|
| `q4b_adaptive_persist` | 13.0 s |
| `q4b_c_persist` | 13.8 s |

### L713 — Q3 — the level the baselines sit at while in violation (peak, not terminal)

```
Figure~\ref{fig:misprediction} shows the resulting QoS trajectories on the CPU--GPU platfo…
```

줄 끝에 붙일 것:

```latex
%!cv q3_static_maxV
```

| id | 확정표 값 |
|---|---|
| `q3_static_maxV` | 4.1 V |

### L715 — Q3 — persist, search, drain, lead, envelope, baseline terminal V

```
BoundGuard instead treats each ranked placement as a hypothesis. It rejects the top-ranked…
```

줄 끝에 붙일 것:

```latex
%!cv q3_bg_persist,q3_bg_search,q3_bg_drain,q3_lead,q3_bound,q3_baseline_lastV
```

| id | 확정표 값 |
|---|---|
| `q3_bg_persist` | 30.4 s |
| `q3_bg_search` | 26.6 s |
| `q3_bg_drain` | 3.8 s |
| `q3_lead` | 9.0 s |
| `q3_bound` | 58.0 s |
| `q3_baseline_lastV` | 3.5 V |

### L742 — Q4 — terminal and peak violation across the five variants (CORRECTS the annotation now in the distribution copy, which names q4_b_lastV: 294.2 lies inside the printed range 293--315 and so appears nowhere on the line)

```
Figure~\ref{fig:bounded} shows the outcome. BoundGuard evaluates each candidate in turn; a…
```

줄 끝에 붙일 것:

```latex
%!cv q4_bg_lastV,q4_c_lastV,q4_adaptive_lastV,q4_bg_maxV,q4_b_maxV,q4_adaptive_maxV
```

| id | 확정표 값 |
|---|---|
| `q4_bg_lastV` | 362.5 V |
| `q4_c_lastV` | 292.6 V |
| `q4_adaptive_lastV` | 315.1 V |
| `q4_bg_maxV` | 480.3 V |
| `q4_b_maxV` | 298.2 V |
| `q4_adaptive_maxV` | 322.8 V |

### L776 — Q5 — persist, lead, the pre-activation part of the lead, baseline terminal V

```
Figure~\ref{fig:npu_violation} shows the outcome. Static, stop-and-restart, and Adaptive h…
```

줄 끝에 붙일 것:

```latex
%!cv q5_bg_persist,q5_lead,q5_lead_pre_activation,q5_baseline_lastV,q5_adaptive_lastV,q5_sr_lastV
```

| id | 확정표 값 |
|---|---|
| `q5_bg_persist` | 27.2 s |
| `q5_lead` | 22.4 s |
| `q5_lead_pre_activation` | 9.6 s |
| `q5_baseline_lastV` | 26.7 V |
| `q5_adaptive_lastV` | 26.7 V |
| `q5_sr_lastV` | 26.9 V |

### L818 — Q6 — variant A's terminal violation, integrated violation, and transition counts

```
Removing dwell fails differently, and the difference is the more instructive result. Under…
```

줄 끝에 붙일 것:

```latex
%!cv q6_a_lastV,q6_a_cumV,q6_adaptive_cumV,q6_a_hotswaps,q6_bg_hotswaps,q4b_a_recoveries,q4b_a_hotswaps
```

| id | 확정표 값 |
|---|---|
| `q6_a_lastV` | 51.7 V |
| `q6_a_cumV` | 2226 V*s |
| `q6_adaptive_cumV` | 264 V*s |
| `q6_a_hotswaps` | 7 swaps |
| `q6_bg_hotswaps` | 6 swaps |
| `q4b_a_recoveries` | 2 of5 |
| `q4b_a_hotswaps` | 15 swaps |
