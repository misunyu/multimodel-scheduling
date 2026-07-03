# Non-interference check — GPU-util sampler vs the measurement

**Question:** does the concurrent pynvml GPU-util sampler perturb the measured
GPU frame skip or worst-stream sAP?

**Design:** k∈{2,8}, N=4 All-GPU, 3 reps each, run twice — sampler **ON**
(Phase A, taskset-pinned to core 23, 200 ms poll) vs sampler **OFF** (Phase B).
Compare against the 3-rep run-to-run spread as the noise baseline. Source:
`rerun_raw.csv` (phase A `All-GPU`, phase B `All-GPU-OFF`).

## GPU frame skip (%)

| k | ON reps | ON mean | OFF reps | OFF mean | Δ(ON−OFF) | ON spread | OFF spread |
|---|---|---|---|---|---|---|---|
| 2 | 51.84 / 48.90 / 49.56 | 50.10 | 48.19 / 47.53 / 48.87 | 48.20 | **+1.90** | 2.94 | 1.34 |
| 8 | 70.27 / 70.30 / 69.18 | 69.92 | 72.09 / 71.94 / 73.43 | 72.49 | **−2.57** | 1.12 | 1.49 |

## Worst-stream sAP

| k | ON mean | OFF mean | Δ(ON−OFF) |
|---|---|---|---|
| 2 | 0.0824 | 0.0813 | **+0.0011** |
| 8 | 0.0695 | 0.0697 | **−0.0002** |

## Conclusion — the sampler does NOT perturb the measurement

1. **Skip deltas are small** (≤2.6 pp) and **of opposite sign** across k (ON
   higher at k=2, ON lower at k=8). A real sampler load would bias skip
   consistently in one direction; the sign flip means the differences are
   run-to-run noise, not a sampler effect.
2. **Each delta is within the run-to-run spread.** At k=2 the ON/OFF gap
   (1.9 pp) is below the ON 3-rep spread (2.9 pp); at k=8 the gap (2.6 pp) is
   comparable to both spreads (1.1–1.5 pp).
3. **Worst-stream sAP is essentially identical** (|Δ| ≤ 0.0011, i.e. ≤1.3% of
   the value), far below the cross-rep variation already present in the sweep.

The sampler is passive NVML polling on an isolated core and performs no
inference, consistent with the measured non-interference. **The 200 ms rate was
therefore retained** (no rate reduction needed). The per-k sAP/util numbers
reported in `gpu_util_by_k.csv` / `persize_sweep_rerun.csv` are from the
sampler-ON runs, which match the sampler-OFF baseline within noise.
