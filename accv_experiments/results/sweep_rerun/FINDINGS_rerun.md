# FINDINGS — ResNet50 sweep re-run with per-size sAP + GPU utilization

Re-executed the N=4 All-GPU ResNet50 contention sweep (k∈{0,1,2,3,4,6,8}, 3 reps)
adding (a) per-size sAP at every k and (b) concurrent GPU-utilization sampling.
`main_vision.tex` and all existing results were **not** modified; everything new
is under `accv_experiments/results/sweep_rerun/`.

**How (no core-script edits):** new driver `phase_rerun_persize_util.py` reuses
the unmodified `measure_multistream` — which already computes per-stream
`sap_s/m/l` and `frame_skip_pct`; the original rev22 simply did not log the size
fields. Same config: YOLOv11s, Argoverse-HD PANEL4=[2,22,3,21], N=4, threads=4,
30-frame warm-up, full-log replay, ResNet50 k-loop lever via `BG_VARIANTS`
mutation. GPU util sampled by a separate pinned `pynvml` process
(`gpu_util_sampler.py`, core 23, 200 ms). All numbers trace to produced files.

## 1. Reproduction check (k=0,1,2 vs stored rev25) — config did NOT drift

Per-size sAP (All-GPU, mean of 3 reps):

| k | rerun skip% | rerun s/m/l | stored rev25 skip% | stored s/m/l | verdict |
|---|---|---|---|---|---|
| 0 | 0.3 | 0.0078 / 0.168 / 0.458 | 0.0 | 0.0078 / 0.1681 / 0.4587 | **exact** (Δl≤0.0003) |
| 1 | 30.9 | 0.0072 / 0.145 / 0.411 | 31.1 | 0.0071 / 0.1436 / 0.3995 | match (same ~31% skip; Δl +0.011, within noise) |
| 2 | 50.1 | 0.0063 / 0.132 / 0.376 | 58.8 | 0.0063 / 0.1285 / 0.3635 | **skip lower** (50 vs 59); sAP consistent vs *measured* skip |

**Interpretation (important):** k=0 reproduces the stored values **exactly**, so
the detector/dataset/eval config did not drift. But at mid/high k the **same k
yields a lower measured skip** in this run than in the original rev22 (k=2: ~50%
vs ~59%; k=8: ~70% vs ~79%). This is the **33 ms-boundary instability the paper
already flags** (`main_vision.tex:513`): near the deadline the k→skip mapping is
machine-state-dependent. The per-size sAP is therefore best read against the
**measured GPU frame skip**, not against k. Plotted that way, the rerun lands on
the same sAP-vs-skip relationship as before (e.g. rerun k=2 at 50% skip has
sap_l 0.376; stored ~52% point ≈ 0.378).

## 2. New per-size numbers at every k (All-GPU, mean of 3 reps)

Source: `persize_sweep_rerun.csv` (per cell) + figure `persize_sweep.pdf`.

| k | GPU skip% (mean, lo–hi) | sAP small | sAP medium | sAP large | loss_l vs k0 | worst sAP |
|---|---|---|---|---|---|---|
| 0 | 0.3 (0.0–0.7) | 0.0078 | 0.168 | 0.458 | 0.000 | 0.115 |
| 1 | 30.9 (30.4–31.4) | 0.0072 | 0.145 | 0.411 | 0.048 | 0.094 |
| 2 | 50.1 (48.9–51.8) | 0.0063 | 0.132 | 0.376 | 0.083 | 0.082 |
| 3 | 54.1 (53.1–54.9) | 0.0065 | 0.131 | 0.368 | 0.090 | 0.081 |
| 4 | 60.2 (59.3–61.0) | 0.0062 | 0.126 | 0.350 | 0.108 | 0.076 |
| 6 | 65.0 (64.2–65.8) | 0.0059 | 0.123 | 0.335 | 0.123 | 0.072 |
| 8 | 69.9 (69.2–70.3) | 0.0058 | 0.120 | 0.333 | 0.125 | 0.069 |

k=3,4,6,8 are **new** (previously only k=0,1,2 had per-size data). The loss is
**large-biased throughout** (loss_l ≫ loss_m ≫ loss_s) — large drops 0.458→0.333
(27%) by k=8 while small barely moves — exactly the staleness signature of
`tab:persize-contention`. **All-NPU reference is flat**: large sAP 0.432 (k=0) vs
0.433 (k=8), worst 0.083 both, NPU skip ≤0.4% — GPU contention does not touch it.

## 3. GPU utilization per k (NEW — was previously unlogged)

Source: `gpu_util_by_k.csv` (sampler-ON, util sliced to each window minus 1 s warm-up).

| k | GPU skip% | util mean | util p50 | util p95 | mem (MiB) | power (W) |
|---|---|---|---|---|---|---|
| 0 | 0.3 | 39.3 | 54 | 69 | 2883 | 156 |
| 1 | 30.9 | 61.2 | 73 | 85 | 3022 | 210 |
| 2 | 50.1 | 60.9 | 71 | 79 | 3554 | 223 |
| 3 | 54.1 | 59.9 | 70 | 76 | 4087 | 232 |
| 4 | 60.2 | 60.3 | 72 | 78 | 4874 | 244 |
| 6 | 65.0 | 61.9 | 73 | 81 | 5945 | 264 |
| 8 | 69.9 | 63.0 | 75 | 80 | 7006 | 274 |
| VLM (ref) | 100.0 | 82.3 | — | — | — | — |

**Key observation:** mean GPU util **saturates near ~60%** from k=1 onward and
barely rises with more ResNet loops, whereas **power (156→274 W) and memory
(2.9→7.0 GB) climb monotonically with k**. So "GPU frame skip" keeps rising
(more co-tenants → longer queueing → more deadline misses) even though the
coarse util% is already near its plateau — util% is a poor proxy for contention
here; the deadline-miss rate (frame skip) and power/memory track it better. The
VLM saturation point (~82% util, separate `cpuload_raw.csv` run) is listed only
as a labeled reference, not part of the ResNet series. (Non-interference of the
sampler verified — see `noninterference_check.md`.)

## 4. Paper-change options (described, NOT applied)

1. **`tab:persize-contention` rows.** Now defensible with **stable, evenly-spaced
   points from one run**: k=2 (~50%), k=4 (~60%), k=8 (~70%) — or k=1/k=4/k=8 for
   ~31/60/70%. **90% is still unreachable by ResNet** (max ~70% here, ~79%
   originally); the saturated row would remain the VLM point. The per-size losses
   above can replace/extend the current 3 rows, read against *measured* skip.
2. **Per-size figure (`persize_sweep.pdf`).** Now a **dense 7-point** curve
   (was a sparse 4-point sketch) — feasible to add as an optional figure: three
   large/medium/small loss curves vs measured GPU frame skip, large-biased, with
   the ~48% crossover band marked.
3. **GPU-util column/row.** Now feasible per k (`gpu_util_by_k.csv`). Honest
   framing: report power/memory alongside util%, and note util% saturates ~60%
   while skip and power keep rising — supports the claim that contention is real
   even where util% looks moderate.

## Files produced (new dir only)
- `phase_rerun_persize_util.py`, `gpu_util_sampler.py` — driver + sampler (scripts/).
- `rerun_raw.csv` — 33 cells (per rep): skip, per-stream/size sAP, worst, t_start/t_end.
- `util_samples.csv` — 2219 raw pynvml samples (epoch, util, mem, power), 445 s span.
- `persize_sweep_rerun.csv` — per cell with util sliced to its window + loss-vs-k0.
- `gpu_util_by_k.csv` — per-k skip + util mean/p50/p95 + mem + power (+VLM ref row).
- `persize_sweep.pdf` — dense per-size loss curves vs measured GPU frame skip.
- `noninterference_check.md` — sampler ON-vs-OFF deltas (within noise).

## Caveats (labeled)
- **Measured** (not inferred): all skip, per-size sAP, util, power, memory above.
- **Inference:** the k→skip downshift vs rev22 is attributed to 33 ms-boundary
  run-to-run variance (supported by exact k=0 reproduction + consistent
  sAP-vs-measured-skip), not config drift. The VLM ~82% util is from a separate
  run (`cpuload_raw.csv`), not this sweep — labeled REFERENCE. No values fabricated.
