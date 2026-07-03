# EXP-NORM v2 — Summary

**Question (W5 defence).** Is the worst camera in Table 5 worst because of GPU contention, or
because it is an inherently difficult log? We add a normalized (Δ-vs-uncontended) view.

**Pre-registered verdict: Outcome A.** The worst-camera *identity* is content-driven (the same log
is worst in every condition), but its contended sAP carries a large, statistically robust,
device-specific contention loss that the NPU does not — and that loss is large-object-biased. The
worst-stream signal under co-tenancy therefore reflects genuine contention damage, not merely log
difficulty.

---

## 1. Environment + gate
See `env_snapshot_norm.md`. EXP-QOS artifacts present; MPS not running; 0 GPU compute procs at
start; no drift. All measurements plain (PRIO/MPS off). Panel = [2,22,3,21] = Table 5 logs.
Table-5 identity confirmed: tab:metric-sensitivity worst (All-GPU 0.066 / All-NPU 0.084) ==
rev22 resnet_k=8 → high-contention point = resnet_k=8 (GPU skip ~80%).

## 2. Inventory (§2)
Existing per-stream files (rev22_perstream_sap.csv) carry only sap+skip at rep0 and **no per-size**;
no plain All-GPU/All-NPU per-size per-stream exists at k0/k8. Per §2 (no back-derivation), all four
cells were **re-measured** at full per-stream + per-size granularity, 3 reps. Each cell's worst-stream
sAP was sanity-checked against the rev22 reference (±0.006) — all PASS:

| cell | re-measured worst (mean) | rev22 ref | \|Δ\| | sanity |
|---|---|---|---|---|
| U-GPU (All-GPU, k0) | 0.1147 | 0.1149 | 0.0002 | PASS |
| U-NPU (All-NPU, k0) | 0.0835 | 0.0836 | 0.0001 | PASS |
| C-GPU (All-GPU, k8) | 0.0682 | 0.0663 | 0.0019 | PASS |
| C-NPU (All-NPU, k8) | 0.0834 | 0.0836 | 0.0002 | PASS |

## 3. Per-stream sAP and Δ (mean over 3 reps); Δ = uncontended − contended
streams in panel order; sid = Argoverse-HD log index.

**All-GPU**
| sid | U-GPU | C-GPU | Δ_GPU | Δ std | C-GPU skip% |
|---|---|---|---|---|---|
| 2  | 0.1214 | 0.0872 | 0.0342 | 0.0014 | 72.2 |
| 22 | 0.2110 | 0.1908 | 0.0202 | 0.0037 | 73.6 |
| 3  | 0.1557 | 0.0903 | **0.0654** | 0.0016 | 71.7 |
| 21 | **0.1147** | **0.0682** | 0.0465 | 0.0005 | 73.2 |

**All-NPU** (contention-invariant)
| sid | U-NPU | C-NPU | Δ_NPU | C-NPU skip% |
|---|---|---|---|---|
| 2  | 0.1089 | 0.1089 | 0.0000 | 0.6 |
| 22 | 0.1913 | 0.1912 | 0.0000 | 0.5 |
| 3  | 0.1214 | 0.1215 | -0.0001 | 0.5 |
| 21 | 0.0835 | 0.0834 | 0.0001 | 0.4 |

All All-NPU Δ ≤ 0.0001 ≤ 0.005 → the NPU path is contention-invariant (separate compute path).

## 4. Worst-stream identity (rank) — content-driven
sid21 is the worst stream in **all four** cells (U-GPU, C-GPU, U-NPU, C-NPU). So *which* camera is
worst is determined by content difficulty: sid21 is the inherently hardest log (lowest sAP even
uncontended, and lowest on the NPU). Rank is preserved.

## 5. Δ structure — non-uniform, and the worst stream's loss is real and large-biased
- Δ_GPU is **non-uniform**: 0.0202 (sid22) … 0.0654 (sid3), range 0.0452 ≫ 3-rep std (~0.001).
- The worst stream (sid21) takes Δ_GPU = **0.0465** (std 0.0005 → ~93σ, highly significant), while on
  the NPU the same log loses **0.0001**. So sid21's contended disadvantage is genuine, GPU-specific
  contention damage, not just difficulty.
- The largest Δ is sid3 (0.0654), not the worst stream — sid3 starts higher (0.1557) and ends at
  0.0903, still above sid21. So the worst-stream position is set by difficulty, but every GPU stream
  pays a real contention Δ.
- **Worst stream sid21 per-size Δ (large-biased staleness signature):**

  | size | GPU U→C | Δ_GPU | NPU U→C | Δ_NPU |
  |---|---|---|---|---|
  | small  | 0.0100→0.0076 | +0.0024 | 0.0014→0.0011 | +0.0003 |
  | medium | 0.0465→0.0130 | +0.0335 | 0.0245→0.0241 | +0.0004 |
  | large  | 0.2615→0.1822 | +0.0793 | 0.2124→0.2126 | -0.0001 |

  large ≫ medium ≫ small on GPU; ≈0 on NPU. The worst camera's contention loss is the same
  large-biased staleness the paper attributes to late delivery (consistent with Tables 2–3).

## 6. Pre-registered Outcome judgment (§0) → Outcome A
- Worst identity is the same across contended/uncontended (sid21).
- Worst's Δ_s = 0.0465 is significant vs 3-rep std (~0.0005), AND All-NPU same-log Δ = 0.0001 ≤ 0.005.
- Both clauses of the Outcome-A trigger are met → **Outcome A**: the worst-stream signal under
  co-tenancy reflects contention damage. (This subsumes Outcome C's reporting, provided in §5: Δ
  distribution + worst's Δ.) Outcome B is rejected — Δ is not uniform.

**Honest caveat (not overclaiming).** The worst-camera *rank* is content-driven (sid21 is hardest in
every condition); contention does not create the worst camera, it deepens it. What EXP-NORM
establishes is that this deepening is real, GPU-specific, large-biased, and absent on the NPU — so
moving the worst camera to the NPU recovers it (0.0682 → 0.0834, +0.0152), and the worst-stream
metric under co-tenancy is a valid contention signal rather than a pure difficulty artifact.

## 7. Proposed Table 5 addition (numbers ready)
Add a per-stream **Δ vs uncontended** view (or at least for the worst camera): on All-GPU the worst
camera loses 0.047 sAP (0.115→0.068) to contention while on All-NPU it loses ~0 (0.084→0.083),
with the loss concentrated on large objects (Δlarge +0.079 vs Δsmall +0.002). This shows the worst
camera is the hardest log *and* that its contended failure is device-specific contention damage the
NPU avoids.

## 8. Failures / skips
None. All 4 cells measured (12 runs), all sanity PASS. No value interpolated or back-derived.
Reused only as cited references (rev22) for the ±0.006 sanity; reported values are all freshly
measured (`data_source=measured`).

## Artifacts
- `exp_norm_results.csv` — 4 cells × 3 reps × 4 streams, per-stream + per-size.
- `env_snapshot_norm.md`, `exp_norm_stdout.log`.
- Script: `accv_experiments/scripts/phase_norm_v2.py` (new; core scripts unmodified).
