# Reproduction: Table `tab:policy-comparison`

## Paper reference
- **Label:** `tab:policy-comparison`
- **Caption (paper):** worst-stream sAP per placement-selection policy (P1/P2 accuracy/latency,
  P3 utilization-threshold, P4 deadline-miss-aware) vs. the post-hoc Oracle, at N=4 devices,
  reported for five representative contention points. Each cell shows worst-stream sAP with the
  regret (gap to Oracle) in parentheses.
- **Paper location:** Section 4.2 (placement-selection policies / policy comparison).

## Input
- `data/a3_policy_comparison.csv` — copied verbatim from `analysis/a3_policy_comparison.csv`.
  One row per (contention point, policy) with columns including `point`, `policy`, `worst_sap`
  (worst-stream sAP over the 4 streams), and `regret` (sAP gap of that policy vs. the Oracle at
  the same contention point). Policies present:
  - `P1_isolated_accuracy` — isolated-accuracy ranking (P1)
  - `P2_latency_profiling` — isolated-latency profiling (P2; ties with P1 here)
  - `P3_util_theta{70,80,90}` — GPU-utilization-threshold migration (P3; the table uses
    theta80)
  - `P4_deadline_miss_aware` — deadline-miss-aware migration (P4)
  - `Oracle` — post-hoc best placement over the family
  Five representative points are selected: `ResNet k=1` (\Lcnn, 24%), `ResNet k=3` (ResNet, 52%),
  `ResNet k=8` (ResNet, 67%), `L2LM` (\Llm, 83%), `L3VLM` (\Lvlm, 100%).

## Run
```
/home/msyu/PycharmProjects/multimodel-scheduling-video/.venv/bin/python generate.py
```

## Outputs
- `policy_comparison.tex` — LaTeX `tabular` for the paper table body.
- `policy_comparison.csv` — the five rows with per-policy worst-sAP and regret.
- `footnote_values.txt` — provenance for the P3 utilization proxy (L2LM) and k=1 boundary note.
- `*_stale_backup.*` — one-time backup of any pre-existing output (created by `backup_stale`).

## Transform (raw values -> paper display)
- The generator emits raw sAP fractions with 3 decimals (e.g. `0.098`).
- The paper table reports **sAP x 100** (e.g. `9.8`) and the regret gap `= policy sAP - Oracle sAP`
  also x 100 (e.g. `+0.5`). The dagger on L2LM/P3 flags the co-tenant-only utilization proxy.

## Verification vs. paper `tab:policy-comparison`  ->  MATCH

| point           | P1/P2         | P3 (theta80)      | P4            | Oracle | paper P1/P2 | paper P3        | paper P4      | paper Oracle |
|-----------------|---------------|-------------------|---------------|--------|-------------|-----------------|---------------|--------------|
| \Lcnn (24%)     | 9.8 (+0.5)    | 9.8 (+0.5)        | 9.8 (+0.5)    | 10.3   | 9.8 (+0.5)  | 9.8 (+0.5)      | 9.8 (+0.5)    | 10.3         |
| ResNet (52%)    | 8.2 (+0.2)    | 8.2 (+0.2)        | 8.3 (+0.0)    | 8.4    | 8.2 (+0.2)  | 8.2 (+0.2)      | 8.3 (+0.0)    | 8.4          |
| ResNet (67%)    | 7.1 (+1.2)    | 7.1 (+1.2)        | 8.4 (+0.0)    | 8.4    | 7.1 (+1.2)  | 7.1 (+1.2)      | 8.4 (+0.0)    | 8.4          |
| \Llm (83%)      | 6.2 (+0.1)    | 6.2 (+0.1) dagger | 4.2 (+2.0)    | 6.3    | 6.2 (+0.1)  | 6.2 (+0.1)dag   | 4.2 (+2.0)    | 6.3          |
| \Lvlm (100%)    | 1.6 (+6.8)    | 8.3 (+0.0)        | 8.3 (+0.0)    | 8.3    | 1.6 (+6.8)  | 8.3 (+0.0)      | 8.3 (+0.0)    | 8.3          |

All 25 reproduced cells (5 points x {P1/P2, P3, P4, Oracle} worst-sAP and regret) equal the paper
values after the x100 display transform. **Verdict: MATCH.**
