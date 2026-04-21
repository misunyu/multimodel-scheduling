# Runtime Overhead Analysis — Reproducibility Notes

This document describes how `results/runtime_overhead.pdf` is produced
so it can be regenerated exactly.

| Component                       | Path                                                       |
|--------------------------------|------------------------------------------------------------|
| Generator script               | `scripts/runtime_overhead_analysis.py`                    |
| Schedule (single combo)        | `tests/steady_state_views_schedule.yaml`                  |
| Per-run CSVs (1 Hz metrics)    | `results/runtime_overhead/mode{1,2,3}_rep{1..N}.csv`      |
| Aggregated JSON                | `results/runtime_overhead/overhead_sweep.json`            |
| Output PDF                     | `results/runtime_overhead.pdf`                            |

---

## 1. End-to-end reproduction

```bash
# Run the steady-state experiment for each mode and render the PDF.
python scripts/runtime_overhead_analysis.py --duration 30 --reps 3

# Skip the experiments and just regenerate the PDF from the existing
# results/runtime_overhead/overhead_sweep.json.
python scripts/runtime_overhead_analysis.py --replot
```

The script wraps `schedule_executor_main.py`, captures both the
executor's metrics CSV and the process tree's CPU usage via psutil
(sampled every 0.5 s), and aggregates across repetitions.

---

## 2. What the figure shows

Four bar charts, one per metric, with three bars each — one per
execution mode:

| Mode label        | `--adaptive-mode` | What it does in steady state                              |
|-------------------|------------------:|------------------------------------------------------------|
| Static            | 3                 | Placement frozen, no monitoring threads                    |
| Adaptive hot-swap | 1                 | `AdaptiveDeployManager` -- monitors for transitions       |
| BoundGuard        | 2                 | `ReactiveDeployManager` -- mode 1 + post-transition rollback |

| Panel       | Metric                                                                 |
|-------------|------------------------------------------------------------------------|
| top-left    | Mean total throughput across the four views (`total_fps` column)       |
| top-right   | Mean per-view inference latency (avg of `view{1..4}_infer_ms`)         |
| bottom-left | Mean drop rate (`drop_rate_fps` column)                                |
| bottom-right| Mean process-tree CPU (psutil, summed over the executor and its children) |

Error bars are 1 σ across the repetitions.

The text block below the panels is auto-rendered by matplotlib's
`figtext` and embedded in the same PDF page. Its numeric content is
templated from the run -- if you re-run the experiment, the deltas
quoted in the text update automatically.

---

## 3. Schedule

`tests/steady_state_views_schedule.yaml` is a single-combination
schedule:

| view  | model     | execution | infps |
|-------|-----------|-----------|------:|
| view1 | mnasnet   | cpu       | 5     |
| view2 | resnet50  | cpu       | 4     |
| view3 | resnext50 | cpu       | 4     |
| view4 | yolov4    | cpu       | 1     |

The single combo is named `combination_stable`. The driver passes
`--combo-duration combination_stable=30` so the executor sticks with
this single placement for the entire run -- no transitions, no
cold-start gaps, V(t) ≈ 0 throughout.

This is intentional: with no transitions, the
`AdaptiveDeployManager` and `ReactiveDeployManager` monitor threads
do not get a reason to fire. Any throughput / latency / CPU delta
observed across modes is therefore attributable to whatever
background work the safety layer does *between* transitions (in
practice, very little).

---

## 4. Measurement details

For each `(mode, rep)` pair the runner:

1. Removes any leftover CSV at the target path.
2. Spawns `schedule_executor_main.py` as a subprocess with
   `QT_QPA_PLATFORM=offscreen` and `--auto_start_all`.
3. Polls the process tree CPU and RSS via psutil every 0.5 s.
   The first 3 s of samples are dropped as warm-up.
4. Waits for the executor to finish (`subprocess.communicate`).
5. Reads the metrics CSV and computes
   - `total_fps` = mean of the `total_fps` column over rows after warmup,
   - `avg_latency_ms` = mean across rows of `mean(view{1..4}_infer_ms)`,
   - `drop_fps` = mean of `drop_rate_fps`,
   - `cpu_mean_pct` = mean of the post-warmup psutil samples.
6. Sleeps `cooldown-sec` (default 4 s) before the next run.

Mean and standard deviation across the `--reps` repetitions are
computed in `stats()` and written to
`results/runtime_overhead/overhead_sweep.json`.

---

## 5. Latest measurements (2026-04-11)

Output of `python scripts/runtime_overhead_analysis.py --duration 30 --reps 3`:

```
mode                      fps     lat (ms)       drop      cpu %
Static                  12.25       480.39      2.261     2269.6
Adaptive hot-swap       13.25       473.76      2.063     2286.6
BoundGuard              12.95       473.88      2.536     2341.6
```

Derived deltas vs. Static:

| Metric       | Adaptive hot-swap | BoundGuard          |
|-------------|------------------:|--------------------:|
| Throughput   | +8.2 %            | +5.7 %              |
| Latency      | -1.4 %            | -1.4 %              |
| Drop rate    | -8.9 %            | +12.2 %             |
| Process CPU  | +17.0 pp (+0.75 %)| +72.0 pp (+3.17 %) |

(`pp` = percentage points; the `%` after it is the relative change.)

Throughput and latency move in both directions across runs and
overlap their 1 σ error bars, so we cannot distinguish the safety
modes from Static at the view level. The CPU footprint is
consistent: BoundGuard sits ~3 % above Static, Adaptive hot-swap
~0.75 % above. Both numbers are dominated by the inference workers
themselves, not the control plane.

---

## 6. How to tweak

- **Longer / shorter runs.** Pass `--duration 60` (or anything you
  want). The first 3 s are always dropped as warm-up.
- **More repetitions.** `--reps 5` if you want tighter error bars
  (each repetition is one full run of `--duration` seconds).
- **Different sample period.** Edit `sample_period=0.5` in
  `run_one()` if you want finer or coarser psutil sampling. Note
  that very fine sampling (e.g. < 0.1 s) starts to add measurable
  observer overhead of its own.
- **Different mode mapping.** The `MODES` list at the top of the
  script is the source of truth. Reorder it or add modes 0 / 4 if
  you want to compare against the stop-and-restart variants too.
- **Different stable workload.** Edit
  `tests/steady_state_views_schedule.yaml` to use a different model
  set or input rates. Make sure the resulting V(t) stays below the
  detection threshold so the system never enters reaction mode.

---

## 7. Software environment

| Component        | Notes                                                    |
|-----------------|----------------------------------------------------------|
| Python          | 3.10 (`.venv/bin/python3`)                               |
| psutil          | 7.2.2                                                    |
| ONNX Runtime    | onnxruntime-gpu (CPU providers used in this experiment) |
| Qt platform     | `QT_QPA_PLATFORM=offscreen` (set by the runner)          |
| Project root    | `/home/msyu/PycharmProjects/fsrr-multimodel-scheduling`  |
| Schedule        | `tests/steady_state_views_schedule.yaml`                 |

---

## 8. Known limitations

- **`cpu_mean_pct` is the *sum* across all worker threads.** On a
  multi-core machine the absolute number can easily run into the
  thousands of percent. The figure quotes the absolute baseline as
  well as the relative delta in the explanation, but reviewers
  unfamiliar with multi-core CPU accounting may need that
  caveat spelled out.
- **3 repetitions × 30 s is small.** It is enough to estimate the
  *direction* of the effect (positive / negative) but not enough to
  give tight 95 % confidence intervals. If a tighter bound is
  needed, raise `--reps`.
- **No inter-process synchronisation.** psutil samples and the
  executor's CSV writer are independent clocks; the alignment of
  the warm-up cut-off is approximate (within ±0.5 s).
- **Steady-state only.** This figure deliberately excludes the cost
  of an actual transition; the cost of the validation/rollback path
  itself is documented in the bounded-recovery and dynamic-load
  experiments.
