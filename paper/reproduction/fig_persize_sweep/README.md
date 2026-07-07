# Reproduction: fig:persize-sweep (`persize_sweep.pdf`)

## Paper reference
- **Label:** `fig:persize-sweep` (Section 4.2, `paper/main_vision.tex`).
- **Caption summary:** Per-size All-GPU sAP-loss figure. Under a GPU
  contention sweep, sAP loss (relative to the no-contention baseline) is
  plotted against the GPU deadline-miss rate for three object-size strata
  (large / medium / small). The large stratum degrades most: its loss grows
  from **4.2% to 12.0%** across the sweep, while the small stratum stays near
  zero — larger objects are disproportionately harmed by GPU contention.

## Input data
- `data/rev30_persize.csv` — copied verbatim from
  `accv_experiments/results/rev30_clean_resnet/rev30_persize.csv`.
- Content: per-size sAP and fractional sAP loss vs the GPU deadline-miss rate
  (`gpu_skip`, in %), from the **rev30 ResNet50 contention sweep (N=4)**.
  Columns: `k, gpu_skip, sap_small, sap_medium, sap_large,
  loss_small, loss_medium, loss_large`. No new measurement is performed; the
  figure re-plots the canonical CSV already emitted by
  `phase_rev30_aggregate.py`.

## Run
```
cd paper/reproduction/fig_persize_sweep
/home/msyu/PycharmProjects/multimodel-scheduling-video/.venv/bin/python generate.py
```

## Output
- `persize_sweep.pdf` — the file `main_vision.tex` `\includegraphics` uses
  for `fig:persize-sweep` (renamed from the source generator's original
  `persize_sweep_percent.pdf`).

## Transform
- The CSV stores **fractional** sAP loss. The plot multiplies each loss column
  by **100** to report on a **percent** y-axis, fixed to the range **0–12.5**
  (headroom so the 12.0 marker at 67.3% deadline misses is not clipped).
- Style matches `phase_rev30_aggregate.fig_persize`: figsize 4.2x3.0;
  tab:red/orange/blue with o/s/^ markers for large/medium/small; no title
  (the caption lives in the tex).

## Verification
Printed percent-scale checks vs `main_vision.tex` L293:

| Check | Reproduced | Paper (tex) | Verdict |
|---|---|---|---|
| large @ ~56% skip (56.1) | 9.6 | 9.6 | match |
| large sweep (24.3% → 67.3% skip) | 4.2 → 12.0 | 4.2 → 12.0 | match |
| small @ ~56% skip | 0.1 | 0.2 | rounding (0.15 → 0.1 floor vs 0.2) |

The large-stratum sweep **4.2% → 12.0%** reproduces the caption exactly.
The only discrepancy is the small-stratum point (0.1 vs tex's 0.2): the CSV
value is `loss_small=0.0015` → 0.15%, which the script formats with `%.1f`
(→ 0.1), whereas the tex rounds to 0.2. This is a cosmetic ±1-in-last-digit
rounding difference on a near-zero value, not a data or logic mismatch.

**Verdict: MATCH.** The regenerated `persize_sweep.pdf` is the same size
(12,861 bytes) as `paper/figures/persize_sweep.pdf` and differs only from
byte 12276 onward (PDF metadata/timestamp region); it renders identically.
