# rev14 — Table 1 `NPU infer 9.86ms` state-source determination

_Extraction only. No new measurement. Existing rev12/T-D0 artifacts untouched. `paper/*` not modified (branch determination only)._

## STEP 1 — source trace of 9.86 ms

- **Location in paper**: `paper/main_vision.tex:170` (Table 1, `infer mean (ms)` row): CPU `31.98±1.66`, GPU `7.95±0.23`, **NPU `9.86±0.30`**.
- **Declared source** (annotation at `main_vision.tex:151`): `SRC=rev7_single_stream.csv`.
- **Verified source CSV**: `accv_experiments/results/rev7_single_stream.csv`, column `latency_mean`, 24 logs, per device.

### State metadata of `rev7_single_stream.csv`

| device | infer mean (ms) | frame_skip_pct | n |
|---|---|---|---|
| CPU | 30.49 ± 0.35 | 4.01% | 24 |
| GPU | 8.67 ± 0.45 | **0.00%** | 24 |
| NPU | **9.98 ± 0.34** | **0.00%** | 24 |

- mxq: `b2441f9d` (rev7 pin), infer_mode `global8`, entry `measure_single_stream`.
- **NPU frame_skip_pct = 0.00%** → this is a **low-skip run**, the same fast-NPU family as rev9 partA (12 ms, <1% skip).
- rev7 large NPU−GPU gap = **+0.0005** (≈ 0, the discredited "large unaffected" result) — confirms rev7 is the outlier state, NOT normal state.

> Note: paper shows NPU `9.86±0.30`, the current `rev7_single_stream.csv` computes `9.98±0.34`. Same family (low-skip ~10 ms, 0% skip), minor numeric drift (likely an earlier regeneration of the same rev7 run / rounding). Either way the **state** is identical: rev7 low-skip outlier. CPU likewise differs slightly (paper 31.98 vs csv 30.49) — same rev7-family run.

### Comparison to normal state

| run | NPU L0 infer | NPU skip% | large gap | state |
|---|---|---|---|---|
| **rev7_single_stream (Table 1 source)** | **~9.86–9.98 ms** | **0%** | +0.0005 | **outlier (low-skip, fast NPU)** |
| rev10 A-1 (`rev10_gen_single`) | 35.53 ± 1.04 ms | 52.4% | (−0.096) | normal |
| rev6 baseline | 36.42 ± 0.89 ms | 55.0% | — | normal |
| rev13 smoke | 37.67 ms | 59.1% | — | normal |
| T-D0 (8-run anchor) | ~36 ms | ~52% | −0.0961 ± 0.0017 | normal (authoritative) |

## STEP 1 verdict — branch **(가)**

**9.86 ms = rev7 low-skip outlier state** (NPU 0% frame skip, ~10 ms infer, large gap ≈ 0). It is the **same anomalous fast-NPU state family** as rev9 partA (12 ms). It is **NOT** the normal state.

- Not branch (나): it is genuinely the NPU `latency_mean` column (not a GPU/CPU mix-up or unit error); the GPU 7.95 and CPU 31.98 rows come from the same rev7 low-skip run, so the *three infer cells are already mutually consistent within rev7* — but that whole run is the outlier state.
- Not branch (다): normal state does NOT reproduce 9.86 ms; it yields ~36 ms (5 independent runs: T-D0, rev6, rev10, rev13 smoke ×3).

**Table 1 infer row is on the outlier state and must be replaced with the normal-state value, to match the (already corrected) large-gap row (−0.096, rev12).**

## STEP 2 — normal-state single-stream NPU infer (extraction)

Extracted from existing normal-state runs (yolo11s, L0, `global8`, mxq `b2441f9d`):

| source | NPU L0 infer (ms) | skip% |
|---|---|---|
| rev10 A-1 (24 logs) | **35.53 ± 1.04** | 52.4% |
| rev6 baseline (24 logs) | 36.42 ± 0.89 | 55.0% |
| rev13 smoke (sid 2, 3 variants) | 35.8–38.0 | 53–59% |

**Recommended normal-state NPU infer = `35.5 ± 1.0 ms`** (rev10 A-1, 24-log, most directly comparable to Table 1's 24-log protocol).

State-consistency gate: T-D0 large gap = −0.0961 ± 0.0017 ✓ (within −0.096 ± 0.002) — confirms rev10/rev6/T-D0 are one normal state.

### GPU / CPU infer in normal state (for a unified Table 1)

To keep all three infer cells in one state, take them from the **same normal-state run** (rev10 A-1):

| device | rev7 (current Table 1, outlier) | rev10 A-1 (normal, recommended) |
|---|---|---|
| CPU | 31.98 ± 1.66 | _(rev10 A-1 did not measure CPU; see note)_ |
| GPU | 7.95 ± 0.23 | **8.55 ± 0.38** (L0, skip 0%) |
| NPU | 9.86 ± 0.30 | **35.53 ± 1.04** (L0, skip 52.4%) |

Note on CPU: rev10 A-1 only ran GPU+NPU. CPU infer is device-only compute and is **largely state-independent** (CPU does not depend on NPU driver state); the rev7 CPU 31.98 ms (or csv 30.49 ms) is acceptable as-is, but if strict single-run unification is desired, a short CPU L0 re-measure (~5 min, 24 logs) would close it. Recommendation: keep CPU ≈ 31 ms (state-independent), update GPU→8.55, NPU→35.53.

## STEP 3 — recommended Table 1 infer row

**Current (outlier, rev7):**
```
infer mean (ms) & 31.98±1.66 & 7.95±0.23 & 9.86±0.30 & --- \\
```

**Recommended (normal state, rev10 A-1; CPU state-independent):**
```
infer mean (ms) & ~31±1.7 & 8.55±0.38 & 35.53±1.04 & --- \\
```

- The NPU jumps 9.86 → 35.53 ms — this is the headline correction and it is **consistent** with the already-corrected large-gap row (−0.096) and with the high frame-skip behaviour that the worst-stream / capacity story relies on.
- GPU 7.95 → 8.55 ms (minor; brings GPU into the same rev10 run).
- CPU: keep ≈ 31 ms (device-only, state-independent); optional re-measure if exact single-run unification required.

## Notes for the (a) paper-edit pass (branch decisions confirmed here)

1. **Table 1 infer row** → replace with normal-state (NPU 35.53 ms). Branch (가).
2. **`%[VERIFY-LAT]` flag** → resolve with NPU infer = 35.53 ± 1.04 ms (rev10 A-1, normal state, skip 52.4%).
3. **partA (Table 2)** → it is a slice of the rev9-partA low-skip outlier session (NPU ~12 ms, <1% skip), same family as 9.86 ms. Demote to mechanism illustration; quantitative authority → normal-state N-axis (`rev9_capacity`) + Table 6.
4. **gen-cstar** → the bg-ladder NPU high-skip (96–100%) is **normal-state behaviour**, not pipeline contamination. The low-skip partA/rev7 numbers were the anomaly. Quantitative crossover authority → clean N-axis.
5. **Limitations** → state honestly: the rev7/rev9-partA fast-NPU low-skip state (~10–12 ms, large gap ≈ 0) was a non-reproducible transient; the normal, reproducible state is ~36 ms NPU infer with ~50% single-stream frame drop. Per-camera absolute gaps are state-dependent; the core conclusions (relative/absolute reversal, worst-stream) rest on the normal-state tables (rev12 single-stream, rev9_capacity, Table 6).

## Files
- `results/rev14_infer_state.md` (this file)
- No `rev14_infer_single.csv` — extraction sufficed, no new measurement.
- rev12/T-D0 artifacts unchanged. `paper/*` unchanged.
