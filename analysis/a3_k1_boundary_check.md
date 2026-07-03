# A3 follow-up — k=1 GPU utilization vs theta=70 boundary check

Re-analysis only (no new GPU experiment). Reproduce with
`.venv/bin/python analysis/a3_k1_boundary_check.py`. `a3_evaluate.py` unchanged.

**Verdict: CASE B** — theta=70 fires in some reps at k=1; "theta=70 never
fires" is run-to-run unstable and must be re-worded (options in §4).

## §1 Pre-rounding aggregate value

- `a3_evaluate.cell()` averages the per-rep `util_mean` values stored in
  `rev30_canonical.csv` (each already rounded to 1 dp by `attach_util()`), then
  rounds to 1 dp for display.
- Aggregate over 10 reps (from stored per-rep 1dp values):
  **69.7300%** -> displayed 69.7%.
- Aggregate recomputed from raw samples (full-precision per-rep means):
  **69.7219%**.
- Both are in **69.65–69.749** -> the displayed 69.7% is a genuine round-to-
  nearest of a value just **below** 70, not an artifact. But the aggregate hides
  rep-level exceedances (see §2).

## §2 Rep-level variability (k=1 All-GPU, 10 reps)

- stored per-rep util_mean: mean 69.730, min 63.6,
  max 74.6, std 2.900.
- **Reps with util_mean > 70.0 (stored): 4/10** -> reps [2, 3, 5, 9].
- **Reps with util_mean > 70.0 (recomputed): 4/10** -> reps [2, 3, 5, 9].

Since at least one rep exceeds 70.0%, the criterion in the brief's decision rule
("a single rep with util_mean > 70.0") selects **Case B**.

## §3 Raw-sample cross-check (aggregation pipeline validation)

Window per rep = `[t_start + 1.0s, t_end]` from `rev30_raw.csv`
(matching `attach_util()` warm-up drop), recomputed directly on
`util_samples.csv`.

- Max |recomputed − stored| across all reps: **0.049 %p** (util_mean and
  util_p95), well within the ±0.5 %p tolerance -> **pipeline consistent**, no
  Case-C discrepancy.
- k=1 p95: stored mean 86.3%, recomputed mean
  86.3% (min 82,
  max 92). A p95-based threshold would have fired
  theta=80 at k=1 (p95 approx 86% > 80%).

### Per-rep table

| rep | GPU DM% | stored mean | recomp mean | >70? | stored p95 | recomp p95 | n | Δmean | Δp95 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 28.56 | 69.7 | 69.699 | False | 85.0 | 85.0 | 73 | 0.001 | 0.0 |
| 1 | 23.57 | 67.5 | 67.465 | False | 83.0 | 83.0 | 71 | 0.035 | 0.0 |
| 2 | 23.17 | 73.4 | 73.352 | True | 90.0 | 90.0 | 71 | 0.048 | 0.0 |
| 3 | 22.67 | 70.9 | 70.899 | True | 86.0 | 86.0 | 69 | 0.001 | 0.0 |
| 4 | 21.75 | 68.2 | 68.2 | False | 92.0 | 92.0 | 70 | 0.0 | 0.0 |
| 5 | 22.07 | 70.3 | 70.319 | True | 86.0 | 86.0 | 69 | 0.019 | 0.0 |
| 6 | 27.51 | 69.2 | 69.151 | False | 83.0 | 83.0 | 73 | 0.049 | 0.0 |
| 7 | 25.68 | 69.9 | 69.901 | False | 86.0 | 86.0 | 71 | 0.001 | 0.0 |
| 8 | 24.95 | 63.6 | 63.62 | False | 82.0 | 82.0 | 71 | 0.02 | 0.0 |
| 9 | 23.03 | 74.6 | 74.614 | True | 90.0 | 90.0 | 70 | 0.014 | 0.0 |

## §4 Decision and paper-wording recommendation — CASE B

theta=70 is a **run-to-run borderline** at k=1: it fires in 4/10
reps (recomputed) / 4/10 (stored). Consequences:

1. **Regret when theta=70 fires at k=1** (P3 -> All-NPU):
   - All-NPU worst-stream sAP (k=1) = 0.0830
   - Oracle worst-stream sAP (k=1) = 0.1026
   - **regret = 0.0196** (approx +0.020).
   (When it does not fire, P3 -> All-GPU, regret = 0.0046.)
   So at k=1 the P3 outcome swings between +0.005 (stay All-GPU)
   and +0.020 (migrate All-NPU) depending on the rep — utilization
   is not just a weak signal, it is an *unstable* one at this boundary.

2. **Paper table is unaffected**: `tab:policy-comparison` uses the theta=80
   column, and theta=80 does **not** fire at k=1 (p95 approx 86% but mean approx 69.7%,
   and 80 > every rep's mean). Table numbers are unchanged.

3. **Recommended wording changes** (do not silently claim identical behavior):

   Replace a blanket "theta in {70,90} behave identically" with:

   ```latex
   % caption / body, honest about the k=1 boundary
   The utilization-threshold policy with $\theta{=}90\%$ never migrates across
   the ResNet50 sweep; $\theta{=}70\%$ is borderline at $k{=}1$, where mean GPU
   utilization is $69.7\pm2.9\%$ and exceeds $70\%$ in 4 of 10
   repetitions, so its migration decision is run-to-run unstable at that single
   point while remaining below threshold for all $k\ge2$.
   ```

   And, where the peak is stated:

   ```latex
   Mean GPU utilization peaks at $69.7\%$ (mean over 10 reps at $k{=}1$;
   per-rep range $63.6$--$74.6\%$) and its aggregate stays at or below $70\%$ for
   every sweep point, even as the foreground GPU deadline-miss rate rises from
   $24\%$ to $67\%$.
   ```

   This keeps the headline (utilization is a poor contention signal) while
   pre-empting the reviewer objection that $69.7\%$ hides threshold crossings.

## Reproducibility

- Script: `analysis/a3_k1_boundary_check.py` (this file's generator).
- No modification to `a3_evaluate.py`; all figures/tables there use theta=80 and
  are unaffected by this boundary finding.
