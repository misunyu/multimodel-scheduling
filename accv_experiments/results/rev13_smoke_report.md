# rev13 R1 smoke — gate FAIL, R1 not proceeded

_Smoke test only. No R1 full sweep performed. No R2._

## Smoke test setup

- detector: yolo11s, mxq sha `b2441f9d`, infer_mode `global8`
- sid: 2 (small-rich anchor)
- bg levels: L0, L1_light
- environment at start: driver `580.159.03`, gpu_util `12%`, gpu_temp `45°C`
- git: `dc478b1f88cd`
- ts: `2026-06-05T17:23:00`
- entry point: `phase_rev6_sweep.measure_single_stream` (identical function used by rev9 partA, rev10/rev12 single-stream, T-D0).

## Smoke results (sid=2)

| device | bg | sap_5095 | NPU latency | frame_skip_pct | wall |
|---|---|---|---|---|---|
| GPU | L0       | 0.1214 | 9.3 ms  | 0.0%   | 5.4s  |
| NPU | L0       | 0.0873 | 35.8 ms | **52.8%** | 15.8s |
| GPU | L1_light | 0.1186 | 21.7 ms | 6.4%   | 11.4s |
| NPU | L1_light | 0.0722 | 65.9 ms | **97.6%** | 16.7s |

## Gate verdict

**R1 gate (NPU frame_skip_pct < 5% across all bg levels): FAIL**

Even at L0 (no contention), NPU skip = 52.8% (≫ 5%). Higher bg levels worse.

## Why the gate fails

Comparison of NPU latency across runs (all use same legacy mxq `b2441f9d`, same `global8` mode, same `measure_single_stream` entry point):

| run | bg | NPU latency | NPU skip% | wall/sid |
|---|---|---|---|---|
| **rev9 partA** (the target clean pipeline) | L1_light | **12 ms** | **0.7%** | 5.8s |
| T-D0 (8-run normal-state anchor) | L0 | ~36 ms | ~52% | ~20s |
| rev6 baseline | L0 | 36 ms | 55% | 19.9s |
| rev10 gen_single | L0 | 35.5 ms | 52% | 19.7s |
| rev10 cstar | L1_light | 66 ms | 96% | 15.6s |
| **rev13 smoke (current)** | L0 / L1_light | 36 / 66 ms | 53 / 98% | 16-17s |

Diagnosis: rev9 partA's NPU latency (12 ms) is an outlier. T-D0, rev6, rev10, and now rev13-smoke all measure ~36 ms NPU latency at L0. The current system state is the **same as T-D0/rev10 state**, not the rev9-partA state. The rev9-partA "clean" pipeline appears to depend on a transient driver/SDK condition that is not currently reproducible.

## What this means for the paper

Per spec §R1: when low-skip pipeline is unachievable, 일반화는 **clean N-axis** (rev9_capacity) 결과로 대체.

- The N-axis analysis is **already clean** (rev9_capacity.csv) and shows:
  - small-rich C* ≈ N=3.98 at L1_light
  - large-rich C* ≈ N=3.96 at L1_light
  - both essentially tied, crossing between N=3 and N=4
- The bg-axis curve from rev10 cstar (in `figures/gen_cstar.pdf`) shows NPU values contaminated by frame_skip (96-100% at L1_light through L2_LM). The "near saturation crossing" claim should be **rephrased**:
  - the bg-ladder gap appears to cross only between L2_LM and L3_VLM in the rev10 pipeline, but this is in the high-skip regime; the rev9-partA-like pipeline would likely cross sooner.
  - **Recommended paper edit (separate task)**: cite N-axis crossover (rev9_capacity, clean) as the authoritative crossover evidence; demote the bg-ladder curve to qualitative or remove.

## What was NOT done

- R1 full sweep (4 dets × 5 bg × 2 dev × 24 sid) — **not started**; gate would fail.
- R2 fine-grain first measurement — **not started**; same state issue + spec recommends skipping.

## Files produced

- `results/rev13_smoke.csv` — 4 rows (this smoke test only)
- `results/rev13_smoke_report.md` — this file

`paper/main_vision.tex` and `paper/tables/*.tex` NOT modified.
