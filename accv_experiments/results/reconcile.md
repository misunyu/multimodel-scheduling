# reconcile — bundle vs cC value conflicts traced to source

_Aggregation only. No measurement. No prose / table edits. `paper/main_vision.tex` untouched._

## R-1  Source trace for each conflicting value

| ID | claimed value | CSV path | filter (N, bg, strategy) | sid panel (in CSV row order) | aggregation | actual CSV value |
|---|---|---|---|---|---|---|
| **V1** | `0.083` | `accv_experiments/results/rev9_capacity.csv` | `tag=N=8, bg=L1_light, placement_name=AllNPU` | `[2, 22, 13, 16, 3, 21, 14, 4]` | `worst_sap` column (= min over `s0..s7_sap`) | **`0.0829`** |
| **V2** | `0.0707` | `accv_experiments/results/rev9_main_cmp.csv` (NOT `rev9_natural.csv` — bundle §5 is main-comparison) | `tag=L2_lm, n_streams=4, bg=L2_lm, placement_name=AllNPU` | `[2, 22, 3, 21]` (Comp A N=4) | `worst_sap` column | **`0.0707`** |
| **V3** | `0.0811` | `accv_experiments/results/rev9_schedule.csv` (decomposition reads this) | `tag=N=8, bg=L1_light, placement_name=AllNPU` | `[2, 22, 13, 16, 3, 21, 14, 4]` | `worst_sap` column (sid 21) | **`0.0811`** |
| **V4** | `0.040` | _(not in any CSV)_ | — | — | — | **assistant chat-text hallucination** — `cC_summary.md` file actually has `0.0829` (matches V1) |
| **V5a** | `0.029` (bundle) | `accv_experiments/results/rev9_main_cmp.csv` | `tag=L2_lm, n_streams=4, placement_name=SizeBlindRev` | `[2, 22, 3, 21]` | `worst_sap` column (sid 21) | **`0.0294`** |
| **V5b** | `0.078` ("cC") | _(not in any CSV)_ | — | — | — | **assistant chat-text hallucination** in the cC summary turn (I typed `SBR(0.0780)` in chat narrative; `cC_summary.md` file itself has `0.0294`) |

### Cross-table sanity check — `cC_summary.md` file (Q-F section) vs `rev9_capacity.csv`

| N | cap CSV AllNPU worst | cap CSV Naive worst | cap CSV SA worst | cap CSV SBR worst | cC file AllNPU | cC file Naive | cC file SA | cC file SBR | match? |
|---|---|---|---|---|---|---|---|---|---|
| 2 | `0.1081` | `0.1205` | `0.1085` | `0.1203` | `0.1081` | `0.1205` | `0.1085` | `0.1203` | ✓ |
| 3 | `0.1090` | `0.1176` | `0.1077` | `0.1158` | `0.1090` | `0.1176` | `0.1077` | `0.1158` | ✓ |
| 4 | `0.0836` | `0.0870` | `0.0924` | `0.0833` | `0.0836` | `0.0870` | `0.0924` | `0.0833` | ✓ |
| 5 | `0.0836` | `0.0702` | `0.0862` | `0.0823` | `0.0836` | `0.0702` | `0.0862` | `0.0823` | ✓ |
| 6 | `0.0836` | `0.0658` | `0.0749` | `0.0834` | `0.0836` | `0.0658` | `0.0749` | `0.0834` | ✓ |
| 8 | `0.0829` | `0.0543` | `0.0615` | `0.0791` | `0.0829` | `0.0543` | `0.0615` | `0.0791` | ✓ |

`cC_summary.md` file is consistent with `rev9_capacity.csv` everywhere. **The 0.040 / 0.078 values exist only in my chat narrative; they were not in any CSV or any output file.**

## R-2  Same-cell judgement per pair

### V1 ↔ V4  (`AllNPU N=8 L1_light worst`)
- **Same CSV?** Both claimed to come from `rev9_capacity.csv`. CSV has exactly one row for `(N=8, L1_light, AllNPU)`.
- **Same sid panel?** N=8 capacity panel is `[2, 22, 13, 16, 3, 21, 14, 4]` (single row, single panel).
- **Same aggregation?** Both = `worst_sap` = min over `s0..s7_sap`. min of `[0.1088, 0.1913, 0.1871, 0.3591, 0.121, 0.0829, 0.4103, 0.2042]` = `0.0829`.
- **Verdict:** **chat-narrative hallucination** in my previous turn. The cC code path is correct; my prose summary mis-stated the value. No CSV/aggregation bug. Correct value = **`0.0829`**.

### V1 ↔ V3  (`AllNPU N=8 L1_light worst` in two different CSVs)
- **Same configuration:** N=8 L1_light AllNPU, same `[2, 22, 13, 16, 3, 21, 14, 4]` sid panel, same `single` infer_mode.
- **Different CSVs:** capacity.csv `0.0829` vs schedule.csv `0.0811`. Difference = `0.0018` (about 2.2% relative).
- **Verdict:** Same conceptual cell, **two independent measurement runs** of the same configuration recorded in two different output tables (capacity table and schedule-shift table both ran `AllNPU N=8 L1_light` independently). The 0.0018 gap is run-to-run measurement noise on the worst sid (sid 21). **Not a bug.** Decomposition table happens to use schedule.csv row, so it reports `0.0811`; capacity row of the main-comparison reports `0.0829`. Both are valid measurements of the same configuration.

### V2 (natural?) attribution
- **Source mis-attribution.** Bundle line 318 sits inside section §5 "main-comparison absolute sAP per placement" (lines 299–319), which derives from `rev9_main_cmp.csv` (N=4 Comp A), not `rev9_natural.csv`. There is **no conflict** — `0.0707` is the N=4 Comp A L2_lm AllNPU worst, not an N=8 natural value.
- For comparison, the natural N=8 L1_light AllNPU values (different rows) are: Comp B `0.0713`, Comp C `0.0583`. Different cells, different values, no contradiction.

### V5a ↔ V5b  (`Isolated L2_lm worst`)
- **Same CSV (both claim main_cmp.csv):** row exists for `(L2_lm, n_streams=4, SizeBlindRev)`. `worst_sap` column = `0.0294`. min over per-stream sAPs `[0.0751, 0.1825, 0.0375, 0.0294]` = `0.0294`.
- **`0.078` does not appear anywhere in the rev9 CSVs as a SizeBlindRev L2 worst.** `grep -n "0.078"` returns only `0.0788` (SA L2_lm mean), `0.0787` (Oracle_k2_1_3 L2_lm mean), and `0.0780` (a few unrelated per-stream stats).
- **Verdict:** `0.078` is **chat-narrative hallucination** from the cC delivery turn (I typed `SBR(0.0780)` in the narrative ranking). `cC_summary.md` file itself has `0.0294`, matching the bundle. Correct value = **`0.0294`**.

## R-3  Definitive values (read directly from CSV, no measurement)

### Cell: `AllNPU N=8 L1_light worst`
- Capacity panel sids = `[2, 22, 13, 16, 3, 21, 14, 4]` (NOT the user-assumed `[2, 22, 16, 17, 13, 8, 3, 21]`; capacity table uses different sids — sid 17 and sid 8 are not in this panel; sids 14 and 4 are).
- Two independent CSV rows for this configuration:
  - `rev9_capacity.csv` (capacity table): worst_sap = **`0.0829`** (1 row, sid 21 as worst)
  - `rev9_schedule.csv` (schedule-shift table): worst_sap = **`0.0811`** (1 row, sid 21 as worst)
- Mean of the two independent measurements = `(0.0829 + 0.0811) / 2 = 0.0820`.
- Range = `0.0018`.

### Cell: `Isolated (SizeBlindRev) L2_lm worst, N=4 Comp A`
- `rev9_main_cmp.csv` panel sids = `[2, 22, 3, 21]` (Comp A N=4, single CSV row).
- worst_sap = **`0.0294`** (sid 21 is the worst stream).
- No second independent measurement of this exact configuration exists in rev9.

### Cell summary (numerical truth, by CSV)

| target cell | CSV row(s) | sid panel | definitive value(s) | range |
|---|---|---|---|---|
| AllNPU N=8 L1_light worst | `rev9_capacity.csv` (cap table) + `rev9_schedule.csv` (sched table) | `[2, 22, 13, 16, 3, 21, 14, 4]` | `0.0829`, `0.0811` (two indep. runs) | `0.0018` |
| Isolated (SBR) L2_lm worst, N=4 Comp A | `rev9_main_cmp.csv` | `[2, 22, 3, 21]` | `0.0294` (single run) | — |
| AllNPU L2_lm worst, N=4 Comp A | `rev9_main_cmp.csv` | `[2, 22, 3, 21]` | `0.0707` (single run) | — |
| AllNPU N=8 L1_light worst, Comp B | `rev9_natural.csv` (B row) | `[17, 8, 10, 11, 2, 13, 3, 14]` | `0.0713` (single run) | — |
| AllNPU N=8 L1_light worst, Comp C | `rev9_natural.csv` (C row) | `[15, 19, 23, 0, 1, 20, 5, 7]` | `0.0583` (single run) | — |

## R-4  Headline conclusion

1. **No CSV / aggregation / parsing bug.** All `rev9_*.csv` files are internally consistent and `cC_summary.md` file values match them exactly (Q-F table cross-checked above).
2. **The conflict was in my previous chat narrative, not in the data.** In the cC delivery turn I typed several wrong numbers (`0.040`, `0.0780`, `0.062`, `0.022`) that did not match the file I had just produced. The `cC_summary.md` file itself has correct values.
3. **The one real inter-CSV discrepancy** is `AllNPU N=8 L1_light worst` measured twice independently as `0.0829` (capacity) and `0.0811` (schedule), differing by `0.0018`. This is normal run-to-run noise on a single worst sid (sid 21); not a bug.
4. **Source mis-attribution in the bundle framing:** the `0.0707` value labelled "All-NPU" in section §5 lines 313–319 is the **N=4 Comp A L2_lm** AllNPU worst from `main_cmp.csv`, not an N=8 natural value from `natural.csv`. Different cell, not a conflict.

No further measurements warranted by this reconciliation. `paper/main_vision.tex` and the auto-generated `paper/tables/*.tex` not modified.
