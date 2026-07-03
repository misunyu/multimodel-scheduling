# EXP-AUDIT-SIZEGROUP — hardcoded SIZE_GROUP vs tertile classification (read-only)

Goal (measurement integrity): verify the hardcoded `SIZE_GROUP` map used by the partial-placement
assignment (`phase_rev20_heavybg.py`) gives the SAME small-rich/large-rich label as step_e's
tertile classification on large-object fraction — so the reported split numbers reflect the intended
device assignment. No re-inference; stored stats only.

## Verdict: **MATCH** (all 8 entries, incl. all 4 N=4-panel sids)

## 1. Hardcoded map (`phase_rev20_heavybg.py:32`)
`SIZE_GROUP={2:"small",22:"small",13:"small",16:"small",3:"large",21:"large",14:"large",4:"large"}`
N=4 partial-placement panel (`PANELS[4]` rev20:33) = **[2,22,3,21]**.

## 2. Tertile recomputation (from stored stats — no re-inference)
Source: `accv_experiments/results/step_e_size_classification.csv` (24 logs, column `pct_large_count`).
sid→log_id via `Argoverse-HD/annotations/val.json` `sequences[sid]`.
Tertile cuts on `pct_large_count` (q1,q2 = quantile[1/3,2/3], step_e's rule): **q1=0.1771, q2=0.2475**
→ label: large if ≥q2, small if <q1, else medium.

## 3. Comparison (hardcoded vs recomputed tertile vs step_e's own size_label)
| sid | log_id[:8] | pct_large_count | step_e size_label | tertile (recalc) | hardcoded | match |
|---|---|---|---|---|---|---|
| 2 * | 1d676737 | 0.0571 | small-dominant | small | small | OK |
| 22 * | f1008c18 | 0.1066 | small-dominant | small | small | OK |
| 13 | aeb73d7a | 0.1202 | small-dominant | small | small | OK |
| 16 | cb0cba51 | 0.1648 | small-dominant | small | small | OK |
| 3 * | 2d12da1d | 0.2559 | large-dominant | large | large | OK |
| 21 * | e9a96218 | 0.2689 | large-dominant | large | large | OK |
| 14 | b1ca08f1 | 0.3030 | large-dominant | large | large | OK |
| 4 | 33737504 | 0.3298 | large-dominant | large | large | OK |

(* = sid used in the N=4 partial-placement table [2,22,3,21].)
All 8 hardcoded labels == recomputed tertile labels == step_e `size_label`. No panel sid lies in the
medium band [q1,q2)=[0.1771,0.2475): all are clearly < q1 (small) or ≥ q2 (large) — no borderline
ambiguity. Robust MATCH.

## 4. Split-number cross-check (auxiliary)
Since labels match, the assignment is as intended, and the reported §6.3 values follow from it
(already confirmed in EXP-AUDIT-SPLIT): N=4 VLM large-object-aware (Isolated, large→NPU, GGNN)
worst ≈ 0.024; contention-aware (Cont-aware, small→NPU, NNGG) worst ≈ 0.015; L1_cnn contention-aware
≈ 0.099 vs All-GPU ≈ 0.092 (`rev20_5strat_heavybg.csv`).

## Conclusion
The hardcoded `SIZE_GROUP` labels are consistent with the large-object-fraction tertile classification
(Setup definition). Measurement integrity confirmed: the partial-placement numbers come from the
intended GPU/NPU assignment. No further action needed.

## Compliance
Read-only; stored step_e stats + val.json only; no re-inference/measurement; per-sid evidence with
values; MATCH not loosened (exact label equality on all entries); paper .tex untouched. Output = this
file only.
