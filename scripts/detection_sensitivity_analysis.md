# Detection Sensitivity Analysis — Reproducibility Notes

This document describes how `results/detection_sensitivity.pdf` is
produced so it can be regenerated exactly.

| Component                       | Path                                                       |
|--------------------------------|------------------------------------------------------------|
| Generator script               | `scripts/detection_sensitivity_analysis.py`               |
| Source measurements (per-tick) | `results/bounded_sweep/run_s*_rep*.csv` (12 CSVs)         |
| Output PDF                     | `results/detection_sensitivity.pdf`                       |
| Output JSON (per-cell numbers) | `results/detection_sensitivity.json`                      |

---

## 1. End-to-end reproduction

```bash
# Re-window the existing bounded-recovery sweep CSVs across the
# (T, eps) grid and render the figure. The bounded sweep itself is
# the prerequisite -- if you do not yet have run_s*_rep*.csv, run:
#   python scripts/run_bounded_recovery_sweep.py --reps 3 --epsilon 50
python scripts/detection_sensitivity_analysis.py
```

The script does **not** spin up the executor; it only postprocesses
the per-tick `v_score` column already saved by the bounded-recovery
sweep. Re-windowing offline keeps the comparison apples-to-apples
(every (T, eps) cell sees the exact same workload trace, so any
difference comes purely from the detector knobs).

---

## 2. What the figure shows

Four 3 × 3 heatmaps, one per metric. Rows are the sliding-window
length `T ∈ {1, 3, 5}` seconds, columns are the threshold
`ε ∈ {25, 50, 75}`.

| Panel       | Metric                                                                 |
|-------------|------------------------------------------------------------------------|
| top-left    | False triggers in healthy phase 1 (count of windowed `V(t) > ε`)       |
| top-right   | Detection latency (ticks from failure to first windowed `V(t) > ε`)    |
| bottom-left | Recovery latency (ticks from failure to first windowed `V(t) ≤ ε` after rollback) |
| bottom-right| Cumulative violation (`Σ_t max(0, V(t) − ε)` over the entire run)      |

Each cell value is the mean across all 12 source CSVs.

The block of explanatory text below the heatmaps is auto-rendered by
`matplotlib`'s `figtext` and embedded in the same PDF page.

---

## 3. Source data

The source CSVs are produced by
`scripts/run_bounded_recovery_sweep.py` and live under
`results/bounded_sweep/`. They contain the canonical 1 Hz per-tick
metrics from the bounded-recovery experiment:

- 4 background-load scenarios (no bg / +squeezenet / +squeeze+shuf /
  +squeeze+shuf+vgg19)
- 3 repetitions per scenario
- = 12 CSVs total

Each CSV has the columns `timestamp, combination, total_fps,
view{1..4}_fps, view{1..4}_infer_ms, drop_rate_fps, v_score`. The
detection-sensitivity script only uses `v_score` (the per-tick
`v(τ)`) and `combination` (to detect phase boundaries).

The view models, schedule YAML, and detection parameters used to
collect those CSVs are documented in
`scripts/bounded_recovery_validation.md`.

---

## 4. Metric definitions (matching the script)

Let `v_per_tick[i]` be the per-tick `v(τ)` written by the executor
(1 Hz) and let `phase_indices(combos)` return the index ranges of
phase 1 (initial), phase 2 (overload), phase 3 (offload).

**Windowed V(t):**
```
V_T(t) = (1/T) · Σ_{τ=t−T+1..t} v(τ)
```

For a given `(T, ε)` cell:

- **False triggers** = `count{ i ∈ phase 1 : V_T(i) > ε }`
- **Detection latency** = `min{ i ∈ phase 2 : V_T(i) > ε  AND  i ≥ p2_start + T − 1 } − p2_start`
  (`None` if `V_T` never crosses `ε` inside phase 2)
- **Recovery latency** = `min{ i ∈ phase 3 : V_T(i) ≤ ε } − p2_start`
- **Cumulative violation** = `Σ_t max(0, V_T(t) − ε)`

The detection-latency rule mirrors the executor's own detector: it
only fires once the sliding window is fully populated with
post-failure samples, so the algorithm cannot trigger on a phase-1
tail leaking into phase 2.

---

## 5. Latest measurements (2026-04-11)

Output of `python scripts/detection_sensitivity_analysis.py`:

```
False triggers (count, all 0):
  T=1   eps={25,50,75} : 0.00 0.00 0.00
  T=3   eps={25,50,75} : 0.00 0.00 0.00
  T=5   eps={25,50,75} : 0.00 0.00 0.00

Detection latency (ticks):
  T=1   eps={25,50,75} : 0.00 0.50 3.08
  T=3   eps={25,50,75} : 2.00 2.17 4.33
  T=5   eps={25,50,75} : 4.00 4.00 5.75

Recovery latency (ticks):
  T=1   eps={25,50,75} : 14.42 14.33 14.17
  T=3   eps={25,50,75} : 16.17 16.08 16.00
  T=5   eps={25,50,75} : 18.08 18.00 17.58

Cumulative violation (sum of windowed V(t) excess over eps):
  T=1   eps={25,50,75} : 1362.85 1008.13  694.95
  T=3   eps={25,50,75} : 1330.55  962.64  634.21
  T=5   eps={25,50,75} : 1296.48  905.99  565.29
```

These are the exact numbers also written to
`results/detection_sensitivity.json`.

---

## 6. How to extend

- **Different `T` / `ε` grids.** Edit `T_VALUES` and `EPS_VALUES` at
  the top of `scripts/detection_sensitivity_analysis.py`. The grid
  is rendered automatically; cell text/font size scales fine for
  3-5 entries per axis.
- **Different source workloads.** Pass `--sweep-glob` pointing at a
  different set of `*.csv` files (any per-tick metrics CSV with a
  `v_score` and `combination` column will work).
- **Add a new metric.** Add it to
  `compute_metrics_for_run()`, register it in `aggregate()`, and add
  another `render_heatmap()` call in `main()`.
