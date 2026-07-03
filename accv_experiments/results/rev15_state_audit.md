# rev15 — state audit: which table/analysis is OUTLIER vs NORMAL

_Classification only. No new measurement. rev12/T-D0/rev14 artifacts unchanged. `paper/*` not modified._

## State classifier (anchors confirmed by rev14)

| state | single-stream NPU L0 latency | NPU skip% (low bg) | single-stream large gap |
|---|---|---|---|
| **OUTLIER** (fast, transient) | ~10–12 ms | < 5% | ≈ 0 (+0.000x) |
| **NORMAL** (reproducible) | ~36 ms | ~50% | ≈ −0.096 |

Anchors — NORMAL: T-D0, rev6, rev10 A-1 (35.5 ms/52%), rev12, rev13 smoke (37.7 ms/59%).
Anchors — OUTLIER: rev7 single-stream (10 ms/0%/gap +0.0005), rev9 partA (12 ms/0.4%).

**Key discriminator for multi-stream tables**: per-stream NPU `frame_skip_pct` at L1_light — OUTLIER family ≈ 1%, NORMAL family ≈ 40%. (Both use the same multistream `single` infer-mode, so the skip gap is a *state* difference, not a mode artifact: rev9 capacity 1.9% vs rev10 gen-gain 40.7% at the same N=4/L1_light.)

## Audit table

| table/figure | source CSV | rev | NPU lat / skip (low bg) | large gap | STATE | note |
|---|---|---|---|---|---|---|
| **Table 1** infer row (old) | rev7_single_stream.csv | rev7 | 10.0 ms / 0.0% | +0.0005 | **OUTLIER** | rev14 replaced w/ normal 35.5 ms |
| **Table 1** sAP/per-size rows | rev12_single_stream.csv | rev12 | (24-log, 52%) | −0.096 | **NORMAL** | already corrected |
| **Table 2 (partA)** | rev9_partA_npu.csv | rev9 | 12.2 ms / 0.4% | gaps −0.008…−0.041 (small) | **OUTLIER** | ★ fast-NPU session |
| **Table 3 (capacity / N-axis)** | rev9_capacity.csv | rev9 | per-stream skip 1.2% (N2:2.4 N4:1.9 N8:1.1) | — | **OUTLIER** | ★ NOT "clean" — low-skip |
| **Table 4 (main comparison)** | rev9_main_cmp.csv | rev9 | L1 skip 1.4% / L2_lm skip 66.5% | — | **MIXED→OUTLIER** | L1=outlier; L2_lm high skip from heavy bg |
| **Table 5 (gen-decomp)** | rev12_gen_decomp.csv | rev12 | (35.5 ms/52% L0) | −0.096 | **NORMAL** | ✓ |
| **Table 6 (gen-gain)** | rev10_gen_gain.csv | rev10 A-2 | L1 skip 40.7% / L2 88.3% | (normal) | **NORMAL** | ✓ ★ core reversal carrier |
| **N-axis crossover** | rev9_capacity.csv | rev9 | skip 1.2% | — | **OUTLIER** | ★ same as Table 3 |
| **Fig gen-cstar (bg-ladder)** | rev10_cstar.csv | rev10 A-3 | L1 66 ms/99%, L2 158 ms/100%, L3 54 ms/82% | — | **NORMAL** | high-skip = real normal bg behaviour |
| **Fig reversal / motiv** | rev9_partA_npu.csv + p1r6_ladder | rev9 | 12 ms / 0.4% | small gaps | **OUTLIER** | shares Table 2 source |
| **rev11 N=8 (3-rep)** | rev11_n8.csv | rev11 | per-stream skip 0.9% | gate single-stream −0.090 | **CONFLICTED** | see note below |

## Per-rev verdict

- **rev7** → OUTLIER (single-stream 10 ms/0%, large gap ≈ 0).
- **rev9** (partA, capacity, main_cmp L1) → **OUTLIER** (12 ms/0.4% single-stream; multistream skip ~1–2%). Decisive: rev9 partA single-stream global8 = 12.2 ms, but 5 independent NORMAL runs give ~36 ms.
  - *Flagged inconsistency*: `rev9_state_gate.csv` (6-sid L0) recorded large gap −0.1028 (normal-range) and passed rev9's own gate. This single number reads NORMAL while every latency/skip signal in the same rev9 run reads OUTLIER. Reported as-is, not forced. Likely the 6-sid large-object sAP saturates and is a weak discriminator; latency (12 ms) + skip (0.4%) are the strong signals → rev9 classified OUTLIER.
- **rev10** (gen_single, gen_gain, cstar) → **NORMAL** (35.5 ms/52% single-stream; multistream skip 40%).
- **rev11** (N=8) → **CONFLICTED**: its fresh 24-log single-stream GATE-Q read NORMAL (gap −0.090, npu_l 0.387), but its N=8 multistream per-stream skip is 0.9% (OUTLIER-like). Same `single` infer-mode as rev10 gen-gain which gave 40% skip → if rev11 were fully normal it should skip ~40%, not 0.9%. **Treat rev11 N=8 numbers as low-skip (outlier-family) for safety; do not rely on them as normal-state.**
- **rev12** → NORMAL (confirmed).

## Summary lists

### NORMAL (usable as-is)
- Table 1 sAP + per-size rows (rev12)
- Table 1 infer row after rev14 fix (rev10 normal)
- Table 5 gen-decomp (rev12)
- **Table 6 gen-gain (rev10 A-2)** ← carries the core reversal
- Fig gen-cstar bg-ladder (rev10 cstar) — high skip is genuine normal bg behaviour

### OUTLIER (demote / re-measure before quantitative use)
- Table 2 partA (rev9) — fast-NPU session slice
- Table 3 capacity / N-axis (rev9) — low-skip, NOT clean
- Table 4 main comparison L1 (rev9) — low-skip
- Fig reversal/motiv (rev9 partA source)
- rev11 N=8 (conflicted → treat as outlier-family)

### MIXED
- Table 4 main comparison: L1_light = OUTLIER (skip 1.4%), L2_lm = high-skip (66.5%) driven by heavy LM background even in the fast session. The L2_lm column behaves normal-like only because the bg load itself forces skips; the underlying NPU state is still the fast session.

## N-axis (rev9_capacity) — final verdict

**OUTLIER.** NPU per-stream skip = 1.2% (N=2: 2.4%, N=4: 1.9%, N=8: 1.1%) — the fast-NPU transient, not the normal ~40% regime. My earlier rev12 call of "clean N-axis" was **wrong** (it was based on internal consistency + same mxq, not on skip/latency state).

**Consequence**: the generalization authority for the crossover **cannot** rest on the N-axis (rev9_capacity). It must rest on **Table 6 (gen-gain, rev10 A-2, NORMAL)**, which already shows the worst-stream inversion at N=4 / L1_light for all four detectors in normal state.

## Does the core claim survive on NORMAL data alone?

**Yes.** The reversal (Contention-aware beats Isolated on worst-stream) is shown by **Table 6 (gen-gain, NORMAL)** — rev12 T3 confirmed all four detectors have positive worst-gain in normal state (yolo11s +0.0168, m +0.0176, l +0.0153, x +0.0280; all 3/3 reps positive). The single-stream quantization story (Table 1 per-size, Table 5) is NORMAL (rev12). So the **core (reversal + worst-stream + size-stratified quantization) stands on NORMAL tables**.

What is OUTLIER-tainted and must be demoted or re-measured:
- partA per-camera absolute gaps (Table 2 / Fig reversal) — mechanism illustration only.
- capacity N-sweep absolute worst-sAP (Table 3) and the N-axis crossover.
- main-comparison L1 absolute numbers (Table 4) — re-source from rev10 gen-gain if a normal headline table is wanted.
- rev11 N=8 reproducibility numbers (conflicted).

## Files
- `results/rev15_state_audit.md` (this file)
- No measurement performed (classification only).
- paper/*, rev12/T-D0/rev14 artifacts unchanged.
