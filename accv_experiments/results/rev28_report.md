# rev28 — evaluation protocol systematically fails (re-analysis)

_Pure CPU re-analysis of existing threads=4 yolo11s CSVs. NO new measurement. Run after rev27 completed (no concurrent GPU/NPU work). core scripts / paper / results unchanged._

## A1 — Ranking reversal frequency (isolated prescription vs deployment-optimal)

Isolated single-camera profiling routes **every** stream to the GPU (Table 1: GPU ≥ NPU at every size), so the isolated prescription is **All-GPU** in every scenario. Deployment-optimal = the deployable strategy maximizing worst-stream sAP under the actual co-tenant. Reversal = optimum ≠ All-GPU.

Source: `rev20_5strat_heavybg.csv` (worst/mean per co-tenant×N×strategy, threads=4, 3 reps). **Granularity = (co-tenant × N) cells**, not per-stream combinatorial scenarios (per-stream sAP under VLM/LM is not in existing data → that richer sampling would need re-measurement; deferred).

| co-tenant | cells (N=2/4/8) | reversed | frac |
|---|---|---|---|
| CNN | 3 | 2 | 67% |
| LM | 3 | 2 | 67% |
| VLM | 3 | 3 | 100% |

Per-cell detail:

| co-tenant | N | isolated Rx (worst) | deployment-best (worst) | reversed? |
|---|---|---|---|---|
| CNN | 2 | All-GPU (0.1149) | All-GPU (0.1149) | no |
| CNN | 4 | All-GPU (0.0915) | Cont-aware (0.0985) | YES |
| CNN | 8 | All-GPU (0.0570) | All-NPU (0.0826) | YES |
| LM | 2 | All-GPU (0.0751) | Cont-aware (0.0751) | YES |
| LM | 4 | All-GPU (0.0618) | All-GPU (0.0618) | no |
| LM | 8 | All-GPU (0.0441) | All-NPU (0.0537) | YES |
| VLM | 2 | All-GPU (0.0181) | All-NPU (0.0829) | YES |
| VLM | 4 | All-GPU (0.0159) | All-NPU (0.0833) | YES |
| VLM | 8 | All-GPU (0.0136) | All-NPU (0.0826) | YES |

## A2 — Oracle vs isolated-prescription gap along the sweep (N=4)

Source: `rev24_sweep_byN_points.csv`. Isolated = All-GPU; Oracle ≈ max(All-GPU, All-NPU) (only these two measured along the sweep — approximation noted). gap grows with GPU skip:

| GPU skip % | isolated (All-GPU) | Oracle≈max | gap |
|---|---|---|---|
| 0 | 0.1149 | 0.1149 | +0.0000 |
| 42 | 0.0879 | 0.0879 | +0.0000 |
| 58 | 0.0765 | 0.0836 | +0.0071 |
| 67 | 0.0739 | 0.0836 | +0.0097 |
| 72 | 0.0719 | 0.0836 | +0.0117 |
| 74 | 0.0700 | 0.0836 | +0.0136 |
| 79 | 0.0663 | 0.0836 | +0.0173 |

The isolated prescription (All-GPU) is optimal at low contention (gap 0) but diverges from the optimum as GPU skip rises past the crossover — the protocol's error grows with deployment contention.

## A3 — Metric sensitivity (All-GPU, high GPU contention, N=4, GPU skip ≈80%)

Source: `rev22_perstream_sap.csv` (threads=4 per-stream, ResNet sweep, highest GPU-pressure point — the reversal regime). **N=4 → only 4 per-stream values, so p10 ≈ worst (coarse percentile).**

| metric | All-GPU | All-NPU | exposes failed camera? |
|---|---|---|---|
| mean | 0.1070 | 0.1263 | hides |
| median | 0.0866 | 0.1152 | partial |
| p10 | 0.0720 | 0.0912 | exposes |
| worst | 0.0660 | 0.0836 | exposes |

All-GPU per-stream sAP at this point: `[np.float64(0.066), np.float64(0.0861), np.float64(0.0871), np.float64(0.1888)]`. mean is buoyed by the less-starved streams; worst (and p10, at N=4) exposes the failed camera. Honest: at N=4 p10 and worst nearly coincide — worst is the most conservative/sensitive, but we do not overclaim p10 is uniquely blind.

## A5 — Worst-stream dominance (leave-one-out)

Source: same per-stream cell (All-GPU, high contention).

- mean 0.1070, worst 0.0660, 2nd-worst 0.0861
- worst/mean = 0.617 (worst is 1.6× below the mean)
- dropping the worst stream lifts the system mean by +0.0137 (0.1070 → 0.1207) — one camera dominates the safety-relevant signal.

## Honest limitations

- Emulated multi-camera (24 forward-camera logs replayed) limits combinatorial diversity; A1 is at (co-tenant×N) granularity, not arbitrary per-stream scenarios.
- A2 Oracle is approximated as best-of-{All-GPU,All-NPU} along the sweep (full 2^N not swept).
- A3/A5 use the ResNet high-GPU-contention point as the reversal-regime proxy (threads=4); the VLM point itself lacks persisted per-stream sAP. N=4 makes p10 coarse.
- No new measurement; all from existing threads=4 yolo11s CSVs (rev20/rev22/rev24).

## Files
- rev28_reversal_frequency.csv, rev28_oracle_gap_sweep.csv, rev28_metric_sensitivity.csv, rev28_worst_dominance.csv, rev28_report.md
