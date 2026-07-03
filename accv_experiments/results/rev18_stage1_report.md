# rev18 STAGE 1 — optimized-postprocess lever gate (G2 FAIL → STAGE 2 not run)

_3 reps each, 24-log single-stream L0, yolo11s, mxq `b2441f9d`, global8. No core eval script modified (lever = process-level `torch.set_num_threads`). rev12/T-D0/rev14/15/16/17 artifacts & paper unchanged._

## The lever (root cause of the 27 ms postprocess)

rev16 found NPU postprocess ≈ 27 ms. rev18 probe localized it: **torch CPU thread thrashing on the tiny YOLO11 head tensors** (max 80×80×80). The decode math is already vectorized; with the default 24 threads, parallelizing these tiny ops costs more than it saves.

| torch threads | postprocess time |
|---|---|
| 24 (default) | 25.8 ± 14.2 ms |
| 4 | **1.33 ± 0.18 ms** |
| 1 | 1.81 ms |

`torch.set_num_threads(4)` → 19× faster postprocess, **identical computation** (same math, only thread count).

## Gate measurement (24-log L0, 3 reps)

| setting | NPU infer | NPU skip | small gap | medium gap | large gap |
|---|---|---|---|---|---|
| threads=24 (thrash) | 36.5 ms | 55.1% | −0.0090 (−56.4%) | −0.0657 (−35.7%) | −0.0974 (−20.4%) |
| **threads=4 (fast)** | **10.1 ms** | **0.0%** | −0.0077 (−48.4%) | −0.0362 (−19.7%) | **+0.0005 (+0.1%)** |

Reproducibility (3 reps): threads=4 latency = [10.12, 10.12, 10.09] ms, large gap = [+0.0005, +0.0005, +0.0005] — std ≈ 0. Fully reproducible (NOT a transient).

## Gate verdict

| gate | result |
|---|---|
| G1 (NPU L0 lat < 33.3 ms & skip < 5%) | **PASS** (10.1 ms, 0%) |
| G2 (large gap −0.096 ± 0.005 preserved) | **FAIL** (gap moved −0.0974 → +0.0005) |
| G3 (≥3 reps reproducible) | **PASS** (std ≈ 0) |
| **OVERALL** | **FAIL → STAGE 2 NOT run** |

Per spec, G2 failure forbids STAGE 2 (a faster pipeline that changes the comparison must not be used to re-measure Table 3). **STAGE 2 was not run.**

## Why G2 "failed" — the important finding

G2 was designed to catch an optimization that *corrupts detections*. That is **not** what happened. The thread lever does not change any detection — per processed frame the boxes are bit-identical (same postprocess math). What changed is **frame-skip**: at 36 ms the NPU misses the 33.3 ms budget and skips 55% of frames (staleness + recall loss); at 10 ms it keeps up and skips 0%.

So the lever isolates the two loss axes cleanly, by turning streaming-skip on/off:

| size | pure quantization (threads=4, skip=0) | + skip/staleness (threads=24 − threads=4) |
|---|---|---|
| small | **−0.0077 (−48.4%)** | −0.0013 |
| medium | −0.0362 (−19.7%) | −0.0295 |
| large | **+0.0005 (+0.1%)** | **−0.0979** |

- **Pure INT8 quantization** (skip-free): steeply small-concentrated (−48% small), **≈ 0 on large**.
- **Skip/staleness**: large-concentrated (−0.098 large), ≈ 0 on small.

This is the two-mechanism thesis in its cleanest form — but it also means:

### ⚠️ The paper's Table 1 large-object number is skip/staleness, not quantization

Table 1 (threads=24 state) reports large NPU−GPU = **−0.096 (−20.1%)** and the corrected text calls it quantization. rev18 shows pure quantization on large is **+0.0005 (≈ 0)**; the −0.096 is almost entirely the frame-skip/staleness of the thread-thrashing postprocess. The genuine size-stratified **quantization** signal is small −48%, medium −20%, large ≈ 0.

This also re-frames the earlier rev14/rev15 state labels:
- the "fast 10–12 ms / gap≈0" state (rev7, rev9) is **reproducible** (threads=4, std≈0), i.e. the *correctly-configured* host state — **not** a one-off transient as rev13/rev14 concluded.
- the "36 ms / 55%-skip / gap −0.096" state (rev10, rev12, T-D0) is the *thread-thrashing misconfiguration* — also reproducible, but an artifact of 24-thread torch on tiny tensors.

Both are reproducible; they differ by a host torch thread setting, not by device capability or driver drift.

## Decision required (human) — which is the paper's operating point?

This is a framing decision, not an auto-fix:

- **Option A — adopt threads=4 as the proper normal state.** The NPU keeps up (10 ms, 0% skip). Table 1 becomes: quantization small −48% / medium −20% / large ≈ 0; the device has essentially no large-object quantization loss. The reversal story must then rest on contention staleness vs the *small/medium* quantization (still valid — small-rich carry the quantization, large-rich don't), and the single-stream "large −20%" claim is dropped. This is the more defensible, correctly-configured state.
- **Option B — keep threads=24 as the operating point**, but relabel Table 1's per-size NPU rows honestly as quantization+streaming-skip (not pure quantization), and present the threads=4 column as the pure-quantization decomposition. Then large −0.096 is explicitly skip/staleness.
- Either way, **STAGE 2 (5-strategy Table 3 re-measure) should be run in the threads=4 state** if a fair, NPU-keeps-up comparison is wanted — but that requires first choosing the operating point (this gate stopped before it).

Per the spec, STAGE 2 is held pending this decision; G2 as literally written failed, and the choice of operating point is the user's.

## Files
- `results/rev18_postproc_levers.csv` (per-size, both thread settings, 3 reps)
- `results/rev18_stage1_report.md` (this file)
- `accv_experiments/scripts/phase_rev18_stage1.py` (gate harness; lever = `torch.set_num_threads`)
- STAGE 2 outputs: none (not run). core scripts / prior artifacts / paper unchanged.
