# FINDINGS — Contention-sweep refinement (table points, per-size data, GPU util)

Read-only data prep. No paper/code/config/existing-result modified. All new
artifacts live in `accv_experiments/results/sweep_refinement/`. Claims are
grounded in file paths + columns; anything not in logs is labeled **(inference)**.

Given (from prior code-verified work): "GPU frame skip" = foreground YOLOv11s
deadline-miss rate, mean over the N=4 streams; sweep = k co-located ResNet50
loops, k∈{0,1,2,3,4,6,8}; the 100% row is the VLM co-tenant, not ResNet.

---

## TASK 1 — Operating points and the ≈30/60/90 selection

### Full N=4 ResNet50 sweep (measured GPU frame skip, mean of 3 reps)
Source: `rev24_sweep_byN_points.csv` (mean) + `rev22_vlm_sweep.csv` (per-rep
`gpu_skip`, `strategy=All-GPU`, `N=4`).

| k | GPU skip % (mean) | per-rep skips | All-GPU worst sAP | worst std |
|---|---|---|---|---|
| 0 | 0.1  | 0.0 / 0.3 / 0.1   | 0.1149 | 0.0000 |
| 1 | 42.0 | 40.6 / 39.6 / 45.9 | 0.0879 | 0.0063 |
| 2 | 57.9 | 55.7 / 59.7 / 58.2 | 0.0765 | 0.0027 |
| 3 | 67.3 | 66.5 / 66.8 / 68.6 | 0.0739 | 0.0006 |
| 4 | 71.5 | 71.2 / 71.6 / 71.8 | 0.0719 | 0.0013 |
| 6 | 73.5 | 71.3 / 75.9 / 73.4 | 0.0700 | 0.0015 |
| 8 | 79.2 | 80.3 / 75.4 / 81.8 | 0.0663 | 0.0022 |

(`rev24_sweep_byN_points.csv:8-14`; per-rep from `rev22_vlm_sweep.csv`.)

**Key structural facts:**
- **k=0→k=1 is a cliff**: skip jumps 0.1% → 42%. There is **no k that lands near
  30%** in the rev22 sweep — k=1 is the first nonzero point and it sits at ~42%
  there.
- **k=1 is unstable** (it sits right at the 33 ms boundary — the instability the
  paper already flags at `main_vision.tex:513`). In the *per-size* run `rev25`,
  the **same k=1** measured **31.1%** (26.2 / 34.1 / 32.9), not 42%. So k=1's
  skip ranges ~26–46% across runs.
- **90% is UNREACHABLE by the ResNet50 sweep at N=4.** The maximum attained is
  **79.2% at k=8** (and even per-rep tops out at ~82%). Only the **VLM
  co-tenant** reaches literal saturation (100%). (At N=8 the ResNet sweep does
  reach 87.9%, but the table is N=4.)

### The three points behind the CURRENT table (reverse-engineered + checked)
The current `tab:persize-contention` rows are the three points that have
*per-size* data (`rev25_persize_under_contention.csv`), NOT an evenly-spaced set:

| table row | source point | k / co-tenant | measured skip (mean, spread) | stability |
|---|---|---|---|---|
| ≈31% | rev25 `skip~42` | k=1 ResNet | 31.1% (26.2–34.1) | **unstable (cliff)** |
| ≈59% | rev25 `skip~58` | k=2 ResNet | 58.8% (57.6–60.0) | stable |
| 100% | rev25 `skip~100` | VLM | 100.0% (literal) | stable |

**Per-size sAP-loss arithmetic** (All-GPU, mean of 3 reps, loss = uncontended −
contended; uncontended = 0.0078 / 0.1681 / 0.4587 for s/m/l):

- k=1 (≈31%): s 0.0078−0.0071=**0.001**, m 0.1681−0.1436=**0.025**,
  l 0.4587−0.3995=**0.059** → matches table `0.001 / 0.025 / 0.059` ✓
- k=2 (≈59%): s 0.0078−0.0063=**0.002**, m 0.1681−0.1285=**0.040**,
  l 0.4587−0.3635=**0.095** → matches table `0.002 / 0.040 / 0.095` ✓
- VLM (100%): s 0.0078−0.0015=**0.006**, m 0.1681−0.0527=**0.115**,
  l 0.4587−0.1328=**0.326** → matches table `0.006 / 0.115 / 0.326` ✓

The arithmetic fully reproduces the existing table, confirming the source rows.

### Verdict on an evenly-spaced ≈30/60/90 table
- **≈30%** → achievable at **k=1**, but only because k=1 happens to wander down
  to ~31% in the rev25 run; it is the unstable cliff point (26–46% across runs).
  A *stable* ~30% point does not exist in the ResNet sweep (the next stable rung
  is k=2 at ~59%).
- **≈60%** → **k=2, 58.8%, stable** — clean, already in the table.
- **≈90%** → **NOT available.** ResNet maxes at ~79% (k=8); the only ≥90% point
  is the VLM saturation (100%), a *different* co-tenant. Producing a genuine
  ResNet ≈90% per-size row would require **re-inference** (a heavier/extra
  co-tenant tuned to ~90%, or per-size extraction at k=8 ~79% — neither is
  stored; see Task 2).

So an evenly-spaced 30/60/90 set cannot be assembled from stored data without
re-inference. The defensible options without new runs are documented at the end.

---

## TASK 2 — Per-size sweep data

**Is per-size (s/m/l) sAP stored for every k? → NO.**
- `rev22_perstream_sap.csv` columns are `rep,N,resnet_k,strategy,stream_id,sid,
  device,sap,skip` — only `sap_5095`, **no per-size**. (covers all k, but no size split)
- `rev25_persize_under_contention.csv` has `sAP_small/medium/large` but **only at
  4 points**: `skip0` (k=0), `skip~42` (k=1), `skip~58` (k=2), `skip~100` (VLM)
  (`phase_rev25_persize.py:34`). k=3,4,6,8 have **no per-size data**.

**Are raw per-frame detections retained for re-extraction? → NO.**
- No `.npy`/`.pkl`/`*detections*` dumps exist (`find` returned none). The files
  named `*_raw.csv` (`geom_staleness_raw`, `L_multistream_raw`, `size_mass_raw`,
  …) are **per-stream aggregates**, not per-frame detections.
- `per_stream_sap` computes sAP on the fly from in-memory `result["results"]` and
  discards it (`_step_d_common.py:220-278`); nothing persists the boxes.

**Conclusion:** a *dense* per-size-vs-k figure (all 7 k) **requires re-inference**
(re-running the rev25-style streaming measurement = inference). Per the
constraints, **I did not run it.** Re-running existing *analysis* over raw
outputs is impossible because the raw outputs were never saved.

**What I produced without re-inference** — from the 4 existing rev25 points:
- `persize_sweep.csv` — k, co-tenant, strategy, measured skip + spread,
  absolute sAP (s/m/l) and loss-vs-uncontended, for **All-GPU and All-NPU**.
- `persize_sweep.pdf` — small/medium/large vs measured GPU frame skip (All-GPU),
  with All-NPU flat reference lines (dashed) and the ~48% crossover region
  marked; VLM saturation point annotated.

**Caveat (stated on the figure):** only **3 ResNet points (k=0,1,2) + 1 VLM
point** exist, so the curve is sparse and its right end (100%) is a *different*
co-tenant than the ResNet middle. It is honest as a 4-point illustration, not a
dense sweep. A dense version needs re-inference at k=3,4,6,8.

---

## TASK 3 — GPU utilization

**Logged in the ResNet sweep? → NO.** `grep` of `rev22_*`, `rev24_*`, `rev25_*`,
and their manifests for `gpu_util|utilization` returned nothing — the sweep
scripts never call NVML/`nvidia-smi`.

**Where GPU util DOES exist (but not for the k-sweep):**
- **Start-of-run idle snapshots** (single scalar, not per-point):
  `phase_rev10_measure.py:130-135` / `phase_rev12_measure.py:81-84` call
  `nvidia-smi --query-gpu=utilization.gpu` once at start → e.g. `rev13_smoke_report.md:10`
  "gpu_util 12%", `rev10_summary.md:8` "14%". These are idle-baseline only.
- **`cpuload_raw.csv` has a real `gpu_util_mean` column**, but for the *co-tenant
  axis* (a separate CPU-load run), not k:
  | condition | GPU util % (mean of 3) |
  |---|---|
  | baseline / L0 (fg only) | 2.8 |
  | +LM / L2_lm | 49.5 |
  | +VLM / L3_vlm | 82.3 |
  So the VLM saturation point (100% frame skip) corresponds to ~82% measured GPU
  util — useful as a reference, but it is **not** a ResNet-k point.

**`gpu_util_by_point.csv`** (produced): every ResNet k row is marked
`NOT_LOGGED`; the L0/LM/VLM reference rows carry the cpuload util values with the
source column flagging they are from a separate run.

**To get GPU util per k (read-only approach, NOT run):** sample in a *separate
process* during a re-execution of the sweep —
`nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv -lms 200`
(or a `pynvml` `nvmlDeviceGetUtilizationRates` poller) writing to a new CSV. This
does not touch any existing result file. **I did not run it**, because it
requires re-executing the inference sweep (not cheap, and the task forbids new
inference unless trivially cheap). Deferred for user approval. **(No utilization
values were fabricated.)**

---

## What paper changes the data would support (OPTIONS — not applied)

1. **`tab:persize-contention` rows.** The data supports the *current* three rows
   (k=1 ≈31% unstable, k=2 ≈59% stable, VLM 100%) exactly. It does **not**
   support a clean evenly-spaced 30/60/90 set without re-inference (no stable
   ~30%, no ResNet ~90%). Two no-new-run options:
   (a) keep the three current rows but relabel honestly — e.g. "≈31%, ≈59%,
   100% (saturated, VLM)"; or
   (b) if even spacing is wanted, re-inference is needed for a ~90% ResNet
   point (and a stable ~30% point).
2. **Per-size figure.** Feasible only as a **sparse 4-point** illustration from
   `persize_sweep.pdf` now; a dense 7-point sweep needs re-inference at
   k=3,4,6,8 (raw detections were not retained).
3. **GPU-util column/row.** Not feasible from sweep logs (absent). A reference
   triple (L0 2.8% / LM 49.5% / VLM 82.3%) exists from `cpuload_raw.csv` and
   could support a *co-tenant* util note, but a per-k util column requires the
   deferred sampler run.

---

## Audit trail / what was searched
- Skip per k: `rev24_sweep_byN_points.csv`, `rev22_vlm_sweep.csv` (N=4 All-GPU).
- Per-size: `rev25_persize_under_contention.csv` (4 points only);
  `rev22_perstream_sap.csv` (no size split). Source: `phase_rev25_persize.py:34`.
- Raw detections: `find ... -iname '*.npy' -o -iname '*detections*'` → none;
  `*_raw.csv` are aggregates. `_step_d_common.py:220-278` discards boxes.
- GPU util: `grep -riE 'pynvml|nvidia-smi|gpu_util|utilization|occupancy|nvml'`
  → only start-snapshots (`phase_rev10/12_measure.py`) and `cpuload_raw.csv`
  `gpu_util_mean`; **none in rev22/24/25**.

**Found vs inferred:** all skip/sAP/util numbers are read from logs. Inferences
(labeled): k=1's cross-run instability is read off the rev22-vs-rev25 spread (a
fact, not a guess); the "90% unreachable" claim is bounded by the logged max
(79.2% at k=8); no util value was invented.
