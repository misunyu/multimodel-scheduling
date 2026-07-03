# rev16 — normal-state NPU idle frame-skip diagnosis

_Profiling harness (no core eval script modified). All latency stats ≥3 reps, std reported. Pinned state: mxq `b2441f9d`, driver 580.159.03, global8, yolo11s, L0 (no bg). rev12/T-D0/rev14/rev15 artifacts & paper unchanged._

## STEP 0 — skip definition and the arithmetic

From `accv_experiments/scripts/_step_d_common.py`:
- `FPS = 30.0` (`minimal_pipeline/step0_compare_devices.py:29`).
- `period_ms = 1000/FPS = 33.33 ms`.
- `frame_skip_pct = 100 × (# post-warmup frames with eff_ms > 33.33) / total`.

So skip counts frames whose effective per-frame latency exceeds the 33.3 ms frame budget.

| device | per-frame eff (ms) | vs period 33.3 | measured skip% |
|---|---|---|---|
| GPU | ~8.6–11 | well under | 0% |
| NPU (normal) | ~35.5–39 | just over | ~52% |

**Verdict STEP 0**: NPU eff (~35.5 ms) sits just above the 33.3 ms period, so ~50% skip is the *arithmetic consequence* of per-frame latency straddling the frame budget — **not a bug**. The real question becomes: why is NPU per-frame ~35.5 ms, and can it be brought below 33.3 ms reproducibly?

## STEP 1 — latency breakdown of the ~35.5 ms (3 reps, sid 2, 120 frames/rep)

Source: `results/rev16_latency_breakdown.csv`.

| stage | mean ms | std | % of total |
|---|---|---|---|
| cv2.imread (disk + JPEG decode, full-res) | 6.89 | — | 17.7% |
| preprocess (resize/normalize/quantize) | 0.54 | — | 1.4% |
| **model() on-chip (H2D + MLA100 + D2H)** | **4.16** | — | **10.7%** |
| **postprocess (dequant/decode/NMS)** | **27.34** | — | **70.2%** |
| **total** | **38.94** | ~1.2 | 100% |
| GPU total per-frame (ref) | 11.05 | 0.08 | — |

- **on-chip compute = 4.16 ms** — the MLA100 is actually *faster* than the GPU's compute; it is NOT the bottleneck.
- **postprocess = 27.34 ms (70%)** is the dominant cost. Sub-profiled: `postprocessor()` = **27.5 ± 12.2 ms**, `Results()` = 0.01 ms. The cost is entirely in `YOLOAnchorlessPost.__call__` → `_pre_process` (host-side dequantize + YOLO decode) + `nms`, all on host CPU via the vendor SDK.
- imread (6.9 ms) is also host-side and shared with the GPU path.

**Verdict STEP 1: HOST-bound.** on-chip (4.16 ms) ≪ period (33.3 ms); the frame-budget overrun is created by host-side postprocess (27 ms), not by the device.

## STEP 2 — expectation cross-check

- Vendor positions ARIES/MLA100 for **8-channel concurrent YOLO**. With on-chip compute ~4 ms/frame, 8 channels × 4 ms ≈ 32 ms of compute — consistent with the 8-channel claim. Our 4.16 ms on-chip corroborates that the *device* is fast; the single-stream 35.5 ms is dominated by host pipeline overhead, exactly as the 8-channel framing implies.
- The high variance of `postprocessor()` (std 12.2 ms) explains the rev9 "12 ms" transient: 12 ms ≈ 4 (on-chip) + 7 (imread) + 0.5 (pre) + ~0.5 (a moment when postprocess ran fast). rev9 caught a low-postprocess window; it is **not reproducible** (3 reps here all show ~27 ms postprocess).

## STEP 3 — is the overhead reproducibly removable?

Reproducibility of the SLOW state: across 3 reps the total is 37.6 / 39.1 / 40.1 ms and postprocess is consistently ~27 ms → **the slow (high-skip) state is the reproducible one**. The fast state (rev9) was a non-reproducible postprocess window — excluded per spec.

Idealized headroom (what the device alone allows): on-chip + imread + preprocess = 4.16 + 6.89 + 0.54 = **11.6 ms ≪ 33.3 ms**. So if postprocess were overlapped/offloaded/optimized, NPU effective latency would fall well under the frame budget → skip < 5%. **The headroom to a fast, *legitimate* state exists and is large.**

Levers (spec STEP 3) — feasibility under the constraints:
- **async/pipelined (overlap post with next compute)**: would hide the 27 ms behind the streaming clock → effective ~12 ms. Requires rewriting the npu_infer/fg_worker path (core eval scripts) — **out of allowed scope to implement here**; demonstrated achievable by the breakdown but not realized in this harness.
- **postprocess offload (NMS/decode to GPU or vectorized)**: the 27 ms is vendor-SDK host code; reducing it means replacing `YOLOAnchorlessPost` — outside the no-core-mod boundary; not validated here.
- **input pre-quantization / single-stream core mode**: preprocess is only 0.54 ms, so negligible upside; not the bottleneck.

No lever was run to a ≥3-rep reproducible skip<5% *within the allowed (no core-mod) scope*, because the removable cost lives in the vendor SDK postprocess / eval streaming path. The headroom is proven (on-chip 4 ms), but a reproducible fast state has **not yet been realized** here.

## STEP 4 — verdict and path-2 recommendation

**Branch (B-qualified): HOST-bound, large removable headroom, but a reproducible fast state is not yet realized within the no-core-modification scope.**

- It is **not (A) device-bound**: on-chip compute is 4.16 ms, far under the 33.3 ms period. The 52% idle skip is host-pipeline overhead (postprocess 27 ms + imread 7 ms), not MLA100 capability.
- It is **not pure (C)**: the headroom is concrete and reproducibly measured (3 reps), not a one-off.
- It is a **qualified (B)**: the device can clearly sustain ≪ 33 ms/frame; realizing it needs an optimized/overlapped NPU postprocess path, which must be *built and validated* (≥3 reps, large gap −0.096±0.002 held) before it counts.

### path-2 recommendation

**Conditionally pursue.** The reviewer risk ("reversal is just immature NPU integration") is *partly real*: the single-stream NPU penalty is inflated by host postprocess, not device compute. Two honest options:

1. **Recommended now — keep the honest normal-state paper** (rev12 single-stream, rev10 gen-gain Table 6, NORMAL). State in Limitations that the single-stream NPU latency is host-postprocess-bound (on-chip 4 ms; postprocess 27 ms), so the *absolute* idle penalty reflects an unoptimized host pipeline, while the **core reversal still holds in normal state** (Table 6 worst-gain positive for all detectors). Do **not** re-run main-comparison/capacity yet, because in the current (slow-postprocess) normal state path-2 would understate the NPU.
2. **Only if an optimized postprocess path is built**: implement async/overlapped or GPU-offloaded NMS in a *new* harness, validate skip<5% reproducibly with large gap held, then re-measure main-comparison/capacity in that faster-but-reproducible state. That state would be more favorable to the NPU and is the fair comparison. This is engineering work beyond the current audit.

**Bottom line**: device is not the bottleneck (on-chip 4 ms); the 52% idle skip is removable host overhead but not yet reproducibly removed under the no-core-mod rule. Hold the honest NORMAL-only paper; flag the host-bound nature explicitly; defer path-2 until an optimized postprocess path is built and validated.

## Files
- `results/rev16_idle_skip_diag.md` (this file)
- `results/rev16_latency_breakdown.csv` (stage breakdown, 3 reps)
- core scripts / prior artifacts / paper unchanged. Profiling harness: `accv_experiments/scripts/phase_rev16_profile.py` (new, read-only wrapper).
