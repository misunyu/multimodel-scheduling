# EXP-AUDIT-SPLIT — partial-placement split rules (read-only)

Question: how do the two §6.3 partial placements ("large-object-aware split", "contention-aware
split") assign streams to GPU/NPU? Confirmed from the assignment code that produced the reported
numbers. Source: `accv_experiments/scripts/phase_rev20_heavybg.py` → `rev20_5strat_heavybg.csv`
(the N=4 subset extracted into the main table via `rev21_mean_worst_extract.csv`). No measurement run.

## Assignment code (verbatim)
`phase_rev20_heavybg.py:45-49`:
```
def named(sids,kind):
    if kind=="All-GPU": return ["GPU"]*len(sids)
    if kind=="All-NPU": return ["NPU"]*len(sids)
    if kind=="Isolated":   return ["NPU" if SIZE_GROUP[s]=="large" else "GPU" for s in sids]
    if kind=="Cont-aware": return ["NPU" if SIZE_GROUP[s]=="small" else "GPU" for s in sids]
```
`phase_rev20_heavybg.py:32`:
```
SIZE_GROUP={2:"small",22:"small",13:"small",16:"small",3:"large",21:"large",14:"large",4:"large"}
```

## Code-name ↔ paper-name mapping (confirmed by reported values)
| paper §6.3 name | code `kind` | N=4 placement | VLM worst | L1_cnn worst |
|---|---|---|---|---|
| large-object-aware split | `Isolated` | GGNN | 0.024 | — |
| contention-aware split | `Cont-aware` | NNGG | 0.015 | 0.099 |
From `rev20_5strat_heavybg.csv` (N=4): L3_vlm Isolated worst 0.0239/0.0226/… ≈ paper **0.024**;
L3_vlm Cont-aware worst 0.0153/0.0149/… ≈ paper **0.015**; L1_light Cont-aware 0.0998/0.0999/0.0959
≈ paper **0.099** vs All-GPU 0.091-0.093 ≈ **0.092**. Values match the §6.3 text → these are the
functions that generated the reported numbers.

## Split-rule table
| policy | sort key / criterion | cutoff → NPU | GPU/NPU @ N=4 | evidence | 확정/미상 |
|---|---|---|---|---|---|
| large-object-aware (`Isolated`) | object-size composition: large-object-rich tertile (by large-object fraction) | streams in the **large**-object-rich group → NPU; others → GPU | 2 GPU / 2 NPU (GGNN for sids [2,22,3,21]) | rev20:48 + :32 | 확정 |
| contention-aware (`Cont-aware`) | object-size composition: **small**-object-rich group | streams in the **small**-object-rich group → NPU; large-object-rich → GPU | 2 NPU / 2 GPU (NNGG) | rev20:49 + :32 | 확정 |

## Q1 — large-object-aware split (fact statement)
The large-object-aware split routes the **large-object-rich** streams to the NPU and the rest to the
GPU (2 NPU / 2 GPU at N=4; placement GGNN for the panel [2,22,3,21]).

## Q2 — contention-aware split (fact statement) — answer = candidate (iii)
The contention-aware split routes the **small-object-rich** streams to the NPU and the
large-object-rich streams to the GPU (2 NPU / 2 GPU at N=4; placement NNGG). It is therefore the
**inverse object-size selection** of the large-object-aware split. It is **not** based on per-stream
sAP/difficulty (candidate i) and **not** on measured frame-skip/staleness (candidate ii); the split
key is the same object-size composition group used in Setup. (The code only encodes the partition;
no rationale comment is present — reported as code fact, not interpreted.)

## "large-object-rich" definition vs Setup
Consistent. Setup (paper) sorts logs by large-object fraction into three equal groups. The size
grouping is computed in `step_e_size_classification.py` by **tertile on `pct_large_count`**
(`q1,q2 = np.quantile(pct_large_count,[1/3,2/3])`, lines ~123/141) → labels small-dominant /
medium-mixed / large-dominant (`step_e_size_classification.csv` `size_label`). `SIZE_GROUP`
(rev20:32) is a **hardcoded per-sid map** using the extreme tertiles only ("small" = small-rich,
"large" = large-rich) for the N=4/N=8 panels. Note (미상 한정): the hardcoded per-sid labels were not
line-verified back to each sid's step_e `size_label`, but the criterion (large-object-fraction
tertile) and terminology match Setup; the assignment *rule* itself is 확정.

## Compliance
Read-only; no measurement/re-run; mappings backed by code lines + matching reported values; no
guessing reported as confirmed; paper .tex untouched. Output = this file only.
