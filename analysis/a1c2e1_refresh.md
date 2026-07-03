# A1 / C2 / E1 refresh — data re-collection consistent with current paper

Re-analysis only (no new GPU experiment). `paper/main_vision.tex` and `a3_*.py`
were **not modified**. Reproduce with `.venv/bin/python analysis/a1c2e1_refresh.py`,
`analysis/a3_k1_boundary_check.py`, and `paper/results_data/regen_table3_main.py`.

Deliverables: `analysis/c2_per_size_losses.csv`, `analysis/c2_q_vs_l_bar.pdf`
(→ `paper/figures/`), `analysis/e1_oracle_selection.csv`,
`paper/results_data/table3_main/` (regenerated) + `regen_table3_main.py`.

---

## Task 0 — Paper-version check ✅ LATEST

`main_vision.tex` (736 lines, mtime 2026-07-03T17:44): `\label{tab:policy-comparison}`
(L510), `\paragraph{\textbf{Placement-selection policies.}}` (L193),
"miss the reversal together" (L642); k=1 boundary finding merged (L505). **Current.**

**Table 3 (`tab:decomp`) as printed** — Quantization −0.008/−0.036/+0.001
(−48.4/−19.7/+0.1%); Staleness ≈0/**−0.030**/−0.098 (≈0/−16.3/−20.5%). ΔsAP 3dp.

## Task 1 — C2 ✅ (see `c2_per_size_losses.csv`, `c2_q_vs_l_bar.pdf`)

Denominator for BOTH rows = FP32 GPU reference $A_b$ (small 0.0159, medium 0.1839,
large 0.4768).

| mechanism | size | ΔsAP raw | paper-round | rel% (raw/$A_b$) | std | source |
|---|---|---|---|---|---|---|
| Quantization | small | −0.0077 | −0.008 | −48.4 | 0.0000 | rev19_table1_threads4.csv |
| Quantization | medium | −0.0362 | −0.036 | −19.7 | 0.0000 | rev19 |
| Quantization | large | +0.0005 | +0.001 | +0.1 | 0.0000 | rev19 |
| Staleness | small | −0.0013 | −0.001 | **−8.0** | 0.0000 | rev18_postproc_levers.csv |
| Staleness | medium | **−0.0295** | −0.030 | **−16.0** | 0.0007 | rev18 |
| Staleness | large | −0.0979 | −0.098 | −20.5 | 0.0010 | rev18 |

- ΔsAP: **all 6 cells match Table 3.** Quantization rel% reproduce exactly.
- Figure `c2_q_vs_l_bar.pdf`: absolute ΔsAP labels only, medium staleness −0.030,
  error bars = 3-rep std. **Quantization std = 0 in every bin → error bars omitted**
  (GPU/NPU sAP identical across reps); staleness std small 0.0000 / medium 0.0007 /
  large 0.0010. Copied to `paper/figures/`.
- ⚠ **Staleness rel% mismatches (flagged):** medium recomputes **−16.0%** (raw
  −0.0295/0.1839), paper prints −16.3% (only from rounded −0.030/0.1839); large
  −20.5% matches; small strict rel% −8.0% vs paper "≈0%".

## Task 2 — E1 ✅ (see `e1_oracle_selection.csv`)

`rev20_5strat_heavybg.csv` (L3_vlm, N=4): rep0 NNNN 0.0834/0.1262; **rep1 NGNN
0.0835/0.1072** (worst 0.0835 > All-NPU 0.0832); rep2 NNNN 0.0834/0.1258. **NNNN
2/3.** Run-avg **Oracle 0.083/0.120, All-NPU 0.083/0.126** = tab:main \Lvlm{}.
Cross-check rev21_mean_worst_extract.csv identical. RES1 pick_dist `{"0":1,"1":9}`
= "nine of ten". **Explanation of Oracle-mean 0.120 < All-NPU 0.126 at equal worst:**
in rep1 the Oracle takes a much lower-mean mixed placement (NGNN mean 0.107) to gain
a hair of worst-stream; averaged, this drops Oracle mean below All-NPU — expected,
since the Oracle maximizes worst, not mean.

## Task 3 — one-stream split wording → **판정 ㄴ (accepted)**

MOVE_ORDER `[3,21,22,2]` = hardcoded large-first 2-tier order (no runtime sort);
one-stream split moves **sid 3**. Within PANEL4 [2,22,3,21]:

| metric | top-1 | sid 3 |
|---|---|---|
| pct_large_count | **sid 21** (0.269) | 2nd (0.256) |
| pct_large_area | **sid 22** (0.848) | 2nd (0.845) |

sid 3 is top-1 under **neither** metric → "the single stream with the **highest**
large-object fraction" is strictly inaccurate; "large-object-aware" name stays
valid (sid 3 is large-dominant). GNGN (sid21-only) split sAP unmeasured → whether
strict top-1 changes 0.103 is UNKNOWN.

## Task 4 — A1 support ✅

**(a) Staleness-small confirmed value** (denom = GPU small 0.0159):
per-rep rel% = **−8.18 / −8.18 / −7.55**, mean raw −0.00127 → printed abs −0.001.
→ **Confirmed single replacement for Table 3 "≈0 (≈0%)": `−0.001 (−8.0%)`**
(raw numerator / $A_b$, consistent with the Quantization row's convention; the
rounded-numerator path would give −6.3%, not recommended).

**(b) Wording locations to co-edit (grep on main_vision.tex):**

| line | text (excerpt) | category |
|---|---|---|
| L272 | `$-0.030$ ($-16.3\%$)` | **only** occurrence of "16.3"; rel% fix |
| ~L268 | Staleness small `$\approx 0$ ($\approx 0\%$)` | make precise → −0.001 (−8.0%) |
| L189 | Oracle "fixed offline order given by each stream's large-object fraction" | order wording (판정 ㄴ) |
| L191 | one-stream split "the single stream with the **highest** large-object fraction" (×2) | **primary ㄴ fix** |
| L194 | P3 "large-object-first in the same order as the Oracle" | consistent — OK |
| L500 | "moving the most affected large-object stream" | loose but defensible |
| L629 | "large-object-aware one-stream split" | name — keep |
| L149 | "sort logs by their fraction of large objects" | panel composition (different context) — OK |
| L231 | "$48.4\%$ and $19.7\%$" quantization rel% | correct — no change |

**F1 fact-check** (`a3_policy_comparison.csv`): P1 = P2 = P3(θ80) regret across
ResNet sweep = **+0.005, +0.000, +0.002, +0.005, +0.007, +0.012** (k1..k8);
diverge only at L3VLM (P1/P2 **+0.068**, P3 +0.000). ✓ supports "selection
criteria used in practice all miss this reversal".

**C1–C4 references:** appear **only** in the bullet list (L70/73/76/79) — no
in-body back-references. `F1`/`F2` labels do not yet exist. An A1 restructuring
edits only those 4 bullets.

## Task 5 — reproduction-package rev30 alignment ✅ (table3 regenerated)

### 5.1 tab:main per-cell source (it is a **mix** of rev30 and rev20/rev21)

| row | cells | value | source |
|---|---|---|---|
| **L1CNN** | AllGPU/AllNPU/Oracle worst+mean, DM | 0.098/0.137, 0.083/0.126, 0.103/0.137, 24/1 | **rev30** — `rev30_tabmain_L1CNN.tex` (rev30_raw N4 k1, 10 reps) / `rev30_oracle_by_contention.csv` RES1 |
| **L2LM** | worst+mean ×3 | 0.062/0.100, 0.042/0.088, 0.063/0.099 | `rev21_mean_worst_extract.csv` (N4 L2_lm) |
| L2LM | DM | 83/76 | `cpuload_raw.csv` (npu_skip 76) / rev20 |
| **L3VLM** | worst+mean ×3 | 0.016/0.047, 0.083/0.126, 0.083/0.120 | `rev21_mean_worst_extract.csv` + `rev20_5strat_heavybg.csv` |
| L3VLM | DM | 100/1 | `rev20_5strat_heavybg.csv` (gpu_skip 100, npu_skip 1) |

Only the **L1CNN row** had drifted (old package: 0.092/0.131, Oracle 0.109, skip 35).

### 5.2 table3_main regenerated ✅

- Backup: `main_worst_mean_rev20_backup.csv`, `table3_rev20_backup.tex`.
- `regen_table3_main.py` rewrites `main_worst_mean.csv` + `table3.tex`: L1CNN from
  rev30 (0.098/0.137, 0.083/0.126, 0.103/0.137, DM 24/1), L2/L3 unchanged; format
  matches paper tab:main body (no bold, `\Lxxx{}` macros, `$g / n$` DM).
- **Cross-check vs main_vision.tex: PASS** (all 18 sAP values present).

### 5.3 Other packages — mismatch report only (NOT regenerated)

| package | status | mismatch vs current paper |
|---|---|---|
| `table1_single_stream` | ⚠ stale | infer/e2e **8.5** vs paper **8.3**; **missing sAP$_{overall}$ row** (0.196/0.186, −5.0%, p<0.01); NPU large **0.477** vs paper **0.478**; row label "infer mean" vs paper "end-to-end mean" |
| `table2_decomp` | ⚠ format | quant row shows **rel%** (−48.4/−19.7/+0.1), not paper's ΔsAP+rel two-value cells; **no staleness rel%**; staleness abs (≈0/−0.030/−0.098) match |
| `fig2_sweep` | ⚠ **major** | **rev24-based**; All-GPU N=4 coords (0.1,0.1149)…(79.2,0.0663) vs paper **rev30** (0.0,0.1149)(24.3,0.0979)…(67.3,0.0713) — entirely different sweep |
| `table4_persize` | ⚠ stale | skip points **31/59/100** & large loss **+0.059/+0.095/+0.326** vs paper rev30 persize **24/45/56/67** & large **0.042/0.078/0.096/0.120** (rev25 vs rev30) |

---

## 원고 반영 시 주의 — 문구 수정 대상 + 확정 수치 (consolidated)

**Confirmed numbers to insert:**
- C2 absolute ΔsAP (all match Table 3): Q −0.008/−0.036/+0.001; L −0.001/−0.030/−0.098.
- C2 relative %: Q −48.4/−19.7/+0.1 (exact). L **−8.0 / −16.0 / −20.5** — use **−16.0**
  (not −16.3) and **−8.0** for the small cell if `≈0` is replaced.
- E1: Oracle 0.083/0.120, All-NPU 0.083/0.126; rep1 NGNN 0.0835 vs All-NPU 0.0832; NNNN 2/3.
- tab:main L1CNN authoritative (rev30): 0.098/0.137, 0.083/0.126, 0.103/0.137, 24/1.

**Residual risks:**
1. **Staleness medium rel% −16.3% (L272) does not reproduce** from rev18 with the
   GPU-$A_b$ denominator (raw = −16.0%). Recommend citing −16.0% or dropping the rel%.
2. **Staleness small (L268) "≈0 (≈0%)"** → precise value **−0.001 (−8.0%)** available
   if desired (rel% is on the small-object $A_b$; keep "≈0" only if framed on absolute).
3. **one-stream split "highest large-object fraction" (L191, and order phrasing L189)
   is 판정 ㄴ** — soften to "the first stream in the large-object-first order"; keep
   "large-object-aware".
4. **Reproduction packages beyond table3 are stale** (table1 e2e 8.3 + missing overall
   row; table2 format; fig2 rev24; table4 rev25) — regeneration out of this task's scope
   (report-only); flag before A1 camera-ready if these packages ship.
5. **C2 bar figure is not yet referenced** in the tex; whether it augments Table 3 /
   `persize_sweep.pdf` is an A1/C2 layout decision.
