# rev24 — N=2/4/8 saturation sweep per-point values (supplementary Table 4, reviewer #5)

_Extraction only (0 new measurements). All points from `rev22_vlm_sweep.csv` (both devices threads=4, ResNet50 GPU-pressure lever, 3 reps/point). x-axis = measured GPU skip%. NPU skip recorded per point. mxq `b2441f9d`, global8, yolo11s. rev22 sanity pre/post: NPU infer ~14 ms, skip 0% (threads=4 held). core scripts / paper unchanged._

Source: each point = mean over 3 reps of All-GPU and All-NPU worst-stream sAP at a given ResNet50 GPU-load. `results/rev24_sweep_byN_points.csv` (21 points).

## N = 2
| GPU skip% | NPU skip% | All-GPU worst | ±std | All-NPU worst |
|---|---|---|---|---|
| 0.0 | 0.0 | 0.1149 | 0.0000 | 0.0836 |
| 0.1 | 0.0 | 0.1149 | 0.0000 | 0.0836 |
| 4.8 | 0.0 | 0.1122 | 0.0012 | 0.0836 |
| 13.6 | 0.0 | 0.1055 | 0.0018 | 0.0836 |
| 23.1 | 0.0 | 0.1031 | 0.0042 | 0.0836 |
| 41.2 | 0.0 | 0.0885 | 0.0042 | 0.0836 |
| 54.5 | 0.0 | 0.0793 | 0.0002 | 0.0836 |

**crossover ≈ 48% GPU skip** (All-NPU flat 0.0836).

## N = 4
| GPU skip% | NPU skip% | All-GPU worst | ±std | All-NPU worst |
|---|---|---|---|---|
| 0.1 | 0.0 | 0.1149 | 0.0000 | 0.0836 |
| 42.0 | 0.0 | 0.0879 | 0.0063 | 0.0836 |
| 57.9 | 0.0 | 0.0765 | 0.0027 | 0.0836 |
| 67.3 | 0.0 | 0.0739 | 0.0006 | 0.0836 |
| 71.5 | 0.1 | 0.0719 | 0.0013 | 0.0836 |
| 73.5 | 0.0 | 0.0700 | 0.0015 | 0.0836 |
| 79.2 | 0.0 | 0.0663 | 0.0022 | 0.0836 |

**crossover ≈ 48% GPU skip** (All-NPU flat 0.0836).

## N = 8
| GPU skip% | NPU skip% | All-GPU worst | ±std | All-NPU worst |
|---|---|---|---|---|
| 63.8 | 0.2 | 0.0701 | 0.0017 | 0.0836 |
| 71.9 | 1.0 | 0.0556 | 0.0007 | 0.0833 |
| 75.9 | 3.0 | 0.0554 | 0.0010 | 0.0814 |
| 78.4 | 4.2 | 0.0559 | 0.0009 | 0.0802 |
| 82.0 | 4.7 | 0.0561 | 0.0009 | 0.0805 |
| 83.9 | 4.1 | 0.0532 | 0.0017 | 0.0815 |
| 87.9 | 4.7 | 0.0507 | 0.0015 | 0.0811 |

**No in-range crossover — All-NPU (flat ≈0.081) wins the entire measured range** (GPU skip 64–88%). At N=8 the 8 foreground streams alone push GPU skip to 64% even with zero co-tenant load, which is already past the ~48% crossover. NPU skip rises slightly to ~4.7% at the heaviest co-tenant (8 NPU streams + host load), but All-NPU still dominates.

## Crossover summary (reviewer #5: stable across N)

| N | crossover (GPU skip%) | All-NPU flat |
|---|---|---|
| 2 | ~48% | 0.0836 |
| 4 | ~48% | 0.0836 |
| 8 | < 64% (already past at min load) | 0.0817 |

The worst-stream device crossover is **stable at ~48% GPU skip for N=2 and N=4**; at N=8 the foreground contention alone exceeds it, so the NPU is preferred throughout — consistent with "crossover stable across stream counts," now backed by actual per-point values rather than a single number.

Reproducibility: **max std across all 21 points = 0.0063** (most ≤ 0.002), far below the All-GPU↔All-NPU separation — the curves are well-resolved.

## Provenance / status
- All 21 points extracted from `rev22_vlm_sweep.csv` (columns: N, resnet_k, gpu_skip, npu_skip, strategy, worst_sap, mean_sap; 3 reps). **No new measurement.**
- `allgpu_mean` / `allnpu_mean` also in `rev24_sweep_byN_points.csv` for the supplementary table.
- NPU skip ≤ 0.1% at N=2/4 (GPU lever does not touch the NPU host path); ≤ 4.7% at N=8 (8 NPU streams) — noted per point.

## Files
- `results/rev24_sweep_byN_points.csv` (N, resnet_k, gpu_skip, npu_skip, allgpu_worst/std/mean, allnpu_worst/std/mean, reps)
- `results/rev24_report.md` (this file)
- core scripts / prior artifacts / paper unchanged.
