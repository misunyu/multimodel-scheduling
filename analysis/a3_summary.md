# A3 Summary — Prior-Work Placement Policy Comparison (Phase 3)

**Question (reviewer):** the isolated→All-GPU baseline looks like a strawman;
compare against the placement-selection criteria that prior work / practice
actually use.

**Answer (one line):** all three prior-work criteria (isolated accuracy,
latency profiling, utilization threshold) select **All-GPU throughout the
GPU-localized contention regime and miss the reversal**; only the
deadline-miss signal tracks it. This is not a strawman — the standard criteria
genuinely fail here, and they fail *together*.

All numbers below are **re-analysis of existing logs** (no new experiment).
Sources: `rev30_clean_resnet/rev30_canonical.csv` (ResNet N=4 sweep, per-cell
util), `rev21_mean_worst_extract.csv` (L2LM/L3VLM endpoints), `cpuload_raw.csv`
(endpoint co-tenant GPU util). Full grid: `a3_policy_comparison.csv`.

---

## 1. What each policy selects, and its regret

Regret = worst-stream sAP(Oracle) − worst-stream sAP(policy selection), $N{=}4$.

| Contention point | GPU DM | GPU util (mean) | P1 acc. | P2 lat. | P3 util θ70/80/90 | P4 DM-aware | Oracle worst |
|---|---|---|---|---|---|---|---|
| ResNet k=1 (\Lcnn) | 24.3% | 69.7% | All-GPU **+.005** | All-GPU **+.005** | All-GPU **+.005** | All-GPU **+.005** | 0.103 |
| ResNet k=2 | 45.0% | 66.4% | All-GPU +.000 | +.000 | All-GPU +.000 | All-GPU +.000 | 0.087 |
| ResNet k=3 | 51.5% | 67.8% | All-GPU +.002 | +.002 | All-GPU +.002 | All-NPU **+.000** | 0.084 |
| ResNet k=4 | 56.1% | 68.3% | All-GPU +.005 | +.005 | All-GPU +.005 | All-NPU **+.000** | 0.084 |
| ResNet k=6 | 62.4% | 68.4% | All-GPU +.007 | +.007 | All-GPU +.007 | All-NPU **+.000** | 0.084 |
| ResNet k=8 | 67.3% | 69.4% | All-GPU **+.012** | **+.012** | All-GPU **+.012** | All-NPU **+.000** | 0.084 |
| \Llm (L2LM) | 83% | 49.5%* | All-GPU +.001 | +.001 | All-GPU +.001 | All-NPU **+.020** | 0.063 |
| \Lvlm (L3VLM) | 100% | ~100%* | All-GPU **+.068** | **+.068** | All-NPU **+.000** | All-NPU +.000 | 0.083 |

`*` endpoint GPU util notes: L2LM = co-tenant-only proxy (49.5%, `cpuload_raw`)
because the All-GPU+L2LM utilization was not directly logged; L3VLM = GPU is
**saturated** (measured GPU DM = 100%, co-tenant-only proxy 82.3%), so All-GPU
utilization ≈ 100%.

**Key readings:**
- **P1 (isolated accuracy) ≡ P2 (latency profiling):** GPU wins isolated sAP
  (0.196 > 0.186) *and* isolated latency (8.3 < 10.0 ms, both < 33.3 ms), so both
  criteria always pick **All-GPU**. Two independent prior-work criteria make the
  *same* wrong call — this is the point, not a coincidence.
- **P3 (utilization threshold)** is **identical to All-GPU across the entire
  ResNet sweep and at L2LM**: measured mean GPU utilization never exceeds 70%
  (max 69.7% at k=1), so θ∈{70,80,90} never triggers migration. Utilization is
  flat while deadline misses climb 24%→67% (`a3_util_vs_dm.pdf`), so a
  utilization threshold cannot see the reversal.
- **P4 (deadline-miss-aware, ours)** is the only policy that tracks the
  reversal on the ResNet sweep, reaching **zero regret at k≥3** and at L3VLM.

Peak regret of the standard criteria: **+0.068 worst-stream sAP at \Lvlm**
(All-GPU 0.016 vs Oracle 0.083 — the paper's 5.2× gap), and a monotonically
growing **+0.012 at the ResNet k=8 point** before any saturation.

## 2. Results that differ from the naive expectation (flagged explicitly)

1. **P3 is not uniformly wrong — it is right for exactly the wrong reason.**
   The A3 brief predicted P3 would "never trigger." That holds across the whole
   ResNet crossover regime and at L2LM (regret up to +0.012 / +0.001), **but at
   \Lvlm the GPU saturates (DM=100%) so utilization ≈100% > θ and P3 migrates to
   All-NPU (regret +0.000).** Interpretation: a utilization threshold only reacts
   once the GPU is *fully* saturated; it is blind to the deadline-driven reversal
   that occurs at moderate contention (flat ~68% utilization). This *strengthens*
   the paper's argument — utilization is a lagging, coarse signal, useful only
   after collapse. Evidence: `rev30_canonical.csv` util_mean per k
   (52/70/66/68/68/68/69, reproducing appendix `tab:gpu-util`).

2. **The recommended signal (P4) has its own failure mode at \Llm.**
   P4 keys only on *GPU* deadline misses. At L2LM, GPU DM=83% > 48% triggers
   All-NPU, but the LLM co-tenant also saturates the host CPU (17%→93%,
   `cpuload_raw`), stalling NPU post-processing (NPU DM=76%), so All-NPU (0.042)
   is *worse* than All-GPU/Oracle (0.062/0.063) → regret **+0.020**. This is the
   direct quantitative motivation for the paper's recommendation to report
   **both** GPU and NPU deadline-miss rates, not GPU DM alone.

3. **P3's L2LM decision rests on a proxy, not a direct measurement** (see §3).

## 3. Additional experiments needed

**None are required for the core P1–P4 comparison** — the entire grid above is
from existing logs. Two optional items would only tighten endpoint precision:

| # | Measurement | Why | Est. runs |
|---|---|---|---|
| 1 | All-GPU + \Llm and All-GPU + \Lvlm **GPU utilization** time series (currently only co-tenant-only util via `cpuload_raw`, foreground on NPU) | make P3's endpoint decision measured rather than proxied/inferred; confirm All-GPU+L2LM util (proxy 49.5%) really stays below θ=70 | 2 configs × 3 reps = **6** short (~14 s) runs |
| 2 | P3 at θ<52% (e.g. p95-utilization variant) where it *would* migrate mid-sweep | the migrated placements (split1–3) are **already** measured in the split family, so this needs **0** new runs — pure re-analysis | 0 |

## 4. Draft text for Sec 4 (English, ready to insert)

> To rule out a strawman comparison, we evaluate the placement that four
> prior-work selection criteria would choose at each contention level and
> measure their regret against a post-hoc Oracle over the same restricted
> placement family. Isolated-accuracy selection (vendor-style FP32 preference)
> and latency-profiling selection (TOD-style~\cite{tod}) both pick All-GPU at
> every operating point, since the GPU wins isolated sAP (0.196 vs.\ 0.186) and
> isolated latency (8.3 vs.\ 10.0~ms, both within the 33.3~ms budget); their
> worst-stream regret therefore grows from $+0.005$ at low contention to $+0.012$
> at the heaviest ResNet50 point and reaches $+0.068$ under the vision-language
> co-tenant. A utilization-threshold migration policy
> ($\theta\in\{70,80,90\}\%$) is indistinguishable from All-GPU across the entire
> GPU-localized sweep, because measured GPU utilization stays near $68\%$ (never
> exceeding $70\%$) while foreground deadline misses climb from $24\%$ to $67\%$,
> so the threshold only fires once the GPU is fully saturated by the VLM
> co-tenant. Only the deadline-miss-aware policy tracks the reversal, achieving
> near-zero regret once GPU deadline misses exceed the $48\%$ crossover, although
> it in turn mis-selects All-NPU under the LLM co-tenant ($+0.020$ regret), where
> host-CPU contention also stalls the NPU path—motivating our recommendation to
> report both GPU and NPU deadline-miss rates. These results show the reversal is
> not an artifact of a weak baseline: the accelerator-selection criteria used in
> practice fail together in the moderate-contention regime that isolated
> profiling cannot see.

---

### Deliverables
- `analysis/a3_data_inventory.md` — Phase 0 data map
- `analysis/a3_policies.py` — P1–P4 pure functions
- `analysis/a3_evaluate.py` — re-analysis driver (regenerates everything below)
- `analysis/a3_policy_comparison.csv` — full (point × policy) grid
- `analysis/a3_policy_table.tex` — compact booktabs table (`tab:policy-comparison`)
- `analysis/a3_policy_figure.pdf` — worst-stream sAP vs GPU deadline-miss
- `analysis/a3_util_vs_dm.pdf` — utilization (flat) vs deadline-miss (rising)
