# Bounded Recovery Validation — Reproducibility Notes

This document records every input, parameter, and assumption that goes into
`results/bounded_recovery_analysis.pdf` so the figure can be regenerated
exactly from real measurements.

| Component                       | Path                                                      |
|--------------------------------|------------------------------------------------------------|
| Sweep driver                   | `scripts/run_bounded_recovery_sweep.py`                   |
| Headless background worker     | `scripts/headless_inference_worker.py`                    |
| 4-view recovery schedule (YAML)| `tests/bounded_recovery_views_schedule.yaml`              |
| Per-tick metrics extractor     | `scripts/qos_recovery_validation.py` (helpers reused)     |
| Plot generator                 | `scripts/bounded_recovery_validation.py`                  |
| Sweep output (raw + meta)      | `results/bounded_sweep/sweep.json`                        |
| Per-run CSVs                   | `results/bounded_sweep/run_s{N}_rep{R}.csv`               |
| Final figure                   | `results/bounded_recovery_analysis.pdf`                   |

---

## 1. End-to-end reproduction

```bash
# 1. Run the sweep (4 scenarios x 3 reps; takes ~10 minutes on this machine).
python scripts/run_bounded_recovery_sweep.py --reps 3 --epsilon 50

# 2. Render the figure from the sweep JSON.
python scripts/bounded_recovery_validation.py --from-json results/bounded_sweep/sweep.json
```

The sweep driver automatically:
- launches background ONNX workers per scenario,
- runs `schedule_executor_main.py` against the 4-view schedule with
  mode 0 (stop-and-restart),
- stops background workers,
- parses each per-run CSV through the same windowed-V(t) helpers as
  `scripts/qos_recovery_validation.py`,
- writes everything (configs, command lines, raw markers, aggregates)
  into `results/bounded_sweep/sweep.json`.

The plot script consumes that JSON: per scenario it draws an
"observed" bar (mean of `T_total` = `T_detect + T_post`) and a "bound"
bar (`T_window + T_post`). Error bars are 1σ over the three repetitions.

---

## 2. Figure semantics

Grouped bar chart, one pair per workload scenario:

| Bar             | Quantity                                 |
|-----------------|------------------------------------------|
| dark navy       | Observed `T_recovery` = `T_detect + T_post` (mean ±1σ over 3 reps) |
| light blue      | Upper bound `T_window + T_post` (mean ±1σ) |

The bound uses `T_window` (= sliding window length used for V(t)) as the
worst-case detection delay, since a windowed threshold detector can take
at most one full window to declare a violation. `T_post` is measured per
run: it covers reconfiguration, worker (re)init, and the V(t)
stabilisation tail.

The script raises `ValueError` if any observed value exceeds its bound.

---

## 3. View models (constant across all scenarios)

Schedule file: `tests/bounded_recovery_views_schedule.yaml`

| view  | model      | initial (CPU) | overload (CPU) | offload (GPU) |
|-------|------------|--------------:|---------------:|--------------:|
| view1 | mnasnet    | infps 3       | infps 400      | infps 400     |
| view2 | resnet50   | infps 3       | infps 60       | infps 60      |
| view3 | resnext50  | infps 3       | infps 60       | infps 60      |
| view4 | yolov4     | infps 1       | infps 10       | infps 10      |

Three phases (combo names are referenced by all scripts):

1. `combination_initial`  — all four view models on CPU at low input
   rates. V(t) ≈ 0.07.
2. `combination_overload` — same models, **same placement**. The
   schedule_executor_main same-placement detection performs an in-place
   `infps` swap (no worker restart). V(t) climbs above ε.
3. `combination_offload`  — all four view models moved to GPU. The
   stop-and-restart between phase 2 and phase 3 produces the cold-start
   gap whose duration is the dominant component of `T_post`.

Note: although the YAML uses `model: yolov4`, the
`run_yolo_{cpu,gpu}_process` handlers in `model_processors.py` resolve
the actual ONNX file via `yolov3_small` (same handler is reused for any
"yolov4"-keyed view in the codebase). This is an existing convention,
not specific to this experiment.

---

## 4. Background load (varies per scenario)

Scenario set in `scripts/run_bounded_recovery_sweep.py`'s `SCENARIOS` list:

| # | Label                              | Background workers (model × device × rate)                                              |
|---|------------------------------------|------------------------------------------------------------------------------------------|
| 1 | Baseline (no bg)                   | none                                                                                     |
| 2 | +1 light CPU bg                    | squeezenet1.0-12 / cpu / 30 fps                                                          |
| 3 | +2 CPU bg (squeeze+shuf)           | squeezenet1.0-12 / cpu / 30 fps; shufflenet-v2-12 / cpu / 30 fps                         |
| 4 | +heavy CPU (squeeze+shuf+vgg)      | squeezenet1.0-12 / cpu / 30; shufflenet-v2-12 / cpu / 30; vgg19 / cpu / 5                |

Each background worker is a `subprocess.Popen` running
`scripts/headless_inference_worker.py` with `--quiet`, started 2 seconds
before the executor and SIGTERM'd after the executor exits. They have no
view, no display, no contribution to V(t) — they only consume CPU.

The inference worker:
- loads the ONNX model with `intra_op_num_threads=1, inter_op_num_threads=1`,
- pads dynamic / symbolic input dimensions to {batch=1, channels=3, spatial=224},
- runs warmup × 2 then loops `sess.run` at the configured rate
  (or unbounded if `--rate 0`),
- exits cleanly on SIGTERM/SIGINT.

---

## 5. Detection / recovery parameters

| Parameter                  | Value | Notes                                          |
|---------------------------|-------|------------------------------------------------|
| Sliding-window length T    | 3 s   | from `qos_recovery_validation.WINDOW_T`        |
| Detection threshold ε      | 50    | `--epsilon 50`                                  |
| Tick rate                  | 1 Hz  | one CSV row per second from the executor      |
| QoS metric ℓ_i             | `avg_infer_time + avg_wait_ms`                 |
| L_SLO,i                    | `1000 / infps_i` ms                             |
| Mode                       | 0     | stop-and-restart (cold-start gap visible)      |
| Repetitions per scenario   | 3     | configurable via `--reps`                       |
| Phase budgets              | initial 8s / overload 14s / offload 16s         |

The marker extraction logic is reused from
`scripts/qos_recovery_validation.py`:

- `t_detect` = first tick in the overload phase where the windowed V(t)
  exceeds ε **and** the window is fully populated with post-failure
  samples (`i >= p2_start + T - 1`).
- `t_recover` = first tick in the offload phase where windowed V(t)
  drops back to ≤ ε.
- `T_detect = t_detect_sec - p2_start_sec`
- `T_post   = t_recover_sec - t_detect_sec`
- `T_total  = t_recover_sec - p2_start_sec`

The figure's analytical bound replaces the *measured* `T_detect` with
its theoretical worst case (= `T = 3 s`). The bound's per-scenario
value is therefore `3 + T_post_mean` and the slack
(`bound − observed = 3 − T_detect_mean`) is bounded above by
`T − 1 = 2 s`.

---

## 6. Latest sweep results (2026-04-11)

`results/bounded_sweep/sweep.json` records the canonical measurements.
Summary:

| # | Scenario                          | T_detect (s) | T_post (s)   | Observed T_recovery (s) | Bound (s)    |
|---|-----------------------------------|--------------|--------------|--------------------------|--------------|
| 1 | Baseline (no bg)                  | 2.33 ± 0.47  | 15.00 ± 0.82 | 17.33 ± 0.47             | 18.00 ± 0.82 |
| 2 | +1 light CPU bg                   | 2.33 ± 0.47  | 15.33 ± 0.47 | 17.67 ± 0.47             | 18.33 ± 0.47 |
| 3 | +2 CPU bg (squeeze+shuf)          | 2.00 ± 0.00  | 15.33 ± 0.47 | 17.33 ± 0.47             | 18.33 ± 0.47 |
| 4 | +heavy CPU (squeeze+shuf+vgg)     | 2.00 ± 0.00  | 16.67 ± 0.47 | 18.67 ± 0.47             | 19.67 ± 0.47 |

Total wall-clock for this sweep run: 14:52:34 → 15:02:34 (10 minutes).

Observations:

- Detection delay sits at 2–3 s, bounded above by the sliding-window
  length T = 3 s as expected.
- `T_post` (post-detection recovery) climbs monotonically as the CPU
  background load gets heavier: 15.0 s (no bg) → 16.7 s (heavy bg).
  This is the contribution of the cold-start cycle plus the V(t)
  stabilisation tail; both lengthen under contention.
- Observed total recovery stays strictly below the analytical bound in
  every scenario, with a margin of 0.67–1.00 s (= worst-case detection
  delay − measured detection delay).
- Variability across reps is small (σ ≤ 0.82 s) thanks to the 1 Hz tick
  resolution and the deterministic phase structure.

---

## 7. How to add a new scenario

1. Edit `SCENARIOS` in `scripts/run_bounded_recovery_sweep.py`. Each
   entry needs a `label` and a `background` list of
   `{model, device, rate}` dicts (paths relative to the project root).
2. Re-run the sweep:
   ```bash
   python scripts/run_bounded_recovery_sweep.py --reps 3 --epsilon 50
   ```
3. Re-render the figure:
   ```bash
   python scripts/bounded_recovery_validation.py --from-json results/bounded_sweep/sweep.json
   ```

The plot script raises `ValueError` if any observed value exceeds its
bound, so an unexpectedly broken scenario fails loudly instead of
silently producing a misleading figure.

---

## 8. Software environment

| Component        | Notes                                                    |
|-----------------|----------------------------------------------------------|
| Python          | 3.10 (`.venv/bin/python3`)                               |
| ONNX Runtime    | onnxruntime-gpu (CUDAExecutionProvider available)        |
| Qt platform     | `QT_QPA_PLATFORM=offscreen` (set by sweep driver)        |
| Project root    | `/home/msyu/PycharmProjects/fsrr-multimodel-scheduling`  |
| Schedule file   | `tests/bounded_recovery_views_schedule.yaml`             |

All other dependencies are pinned in `requirements.txt`.

---

## 9. Known limitations

- **`T_detect` is granular at 1 second** because the executor writes
  CSV rows at 1 Hz. The "true" detection time can fall anywhere inside
  a 1-second window; the figure underestimates it by up to 0.5 s on
  average.
- **Background load never crosses to GPU.** All scenarios add CPU
  contention only. The 4 view models go from CPU (initial/overload) to
  GPU (offload), so a GPU-side background worker would mainly stress
  phase 3 — interesting but a separate experiment.
- **`T_post` is dominated by the V(t) stabilisation tail**, not the raw
  reconfiguration time. The cold-start gap itself is ~3 s on this
  machine; the rest of `T_post` is the time for the post-spike V(t)
  values to fall out of the 3-second sliding window.
- **N = 3 repetitions** is enough to expose the qualitative trend
  (heavier background → longer `T_post`), but the σ for `T_detect` /
  `T_post` would shrink with more reps.
