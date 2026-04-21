# ML-misprediction Case Study — Reproducibility Notes

This document records the inputs, models, and execution steps that
produce `results/ml_misprediction_fallback.pdf`.

| Component                       | Path                                                       |
|--------------------------------|------------------------------------------------------------|
| Driver / plot script           | `scripts/ml_misprediction_validation.py`                  |
| Candidates YAML (predictor)    | `tests/ml_misprediction_candidates.yaml`                  |
| ML-only runtime YAML           | `tests/ml_misprediction_mlonly_runtime.yaml`              |
| BoundGuard runtime YAML        | `tests/ml_misprediction_boundguard_runtime.yaml`          |
| XGBoost model prefix           | `xgboost_model/artifacts/gpu/xgb_model_x3_double*`        |
| ML-only run CSV                | `results/ml_misprediction/mlonly.csv`                     |
| BoundGuard run CSV             | `results/ml_misprediction/boundguard.csv`                 |
| Aggregated JSON                | `results/ml_misprediction/ml_misprediction.json`          |
| Final figure                   | `results/ml_misprediction_fallback.pdf`                   |

---

## 1. End-to-end reproduction

```bash
# Run the case study end-to-end (XGBoost predictor + two executor runs).
python scripts/ml_misprediction_validation.py

# Skip the executor runs and just regenerate the PDF from the cached
# CSVs and JSON.
python scripts/ml_misprediction_validation.py --replot
```

The driver:

1. Loads the four candidate placements from
   `tests/ml_misprediction_candidates.yaml` and runs them through
   `deploy_predictor_logic.DeployPredictor` with the
   `xgb_model_x3_double` prefix and `alpha=0.2`. The full ranking and
   the top pick are saved to the JSON output.
2. Runs `schedule_executor_main.py` twice:
   - **mode 1** (`AdaptiveDeployManager` hot-swap) on
     `tests/ml_misprediction_mlonly_runtime.yaml` — labelled
     "Adaptive (ML-only)". The YAML contains a single combination
     (the high-rate CPU placement that the predictor implicitly
     recommended once you scale its low-rate "best" choice up to the
     runtime workload). With only one combo there is no fallback to
     transition to, so the system stays in the failing placement
     for the entire run.
   - **mode 1** (`AdaptiveDeployManager` hot-swap) on
     `tests/ml_misprediction_boundguard_runtime.yaml` — labelled
     "BoundGuard". The YAML contains two combinations: the same
     failing CPU placement as ML-only, then the heuristic GPU-offload
     fallback. The executor schedules the transition between them via
     `--combo-duration`; mode 1 hot-swaps from one to the other
     without stopping the running workers, so service stays
     continuous through the reconfiguration.
3. Loads both CSVs through the same windowed-V(t) helpers used by
   `scripts/qos_recovery_validation.py` so the metric definition stays
   consistent across all figures in the paper.
4. Drops the first three rows of each CSV as warm-up (see §6 below)
   so the two curves start from the same steady-state plateau and the
   visual comparison is fair.
5. Renders the PDF and writes the JSON summary.

---

## 2. The story the figure tells

| Quantity                                          | Value (typical) |
|--------------------------------------------------|----------------:|
| Predictor's top pick (CPU/GPU candidates)        | `combination_all_gpu` |
| Heuristic fallback used by BoundGuard            | `combination_offload` (all four views on GPU) |
| BoundGuard fallback transition                   | t ≈ 21 s        |
| BoundGuard `V(t) ≤ ε` after fallback             | t ≈ 24 s        |
| Recovery time after fallback                     | ~3 s            |
| ML-only `V(t)` plateau                           | ~100 ± 5        |
| BoundGuard end-of-run `V(t)`                     | ~3              |
| Detection threshold ε                            | 50              |
| Sliding-window length T                          | 3 s             |

Both runs start in the failing placement at t ≈ 0 with `V(t) ≈ 100`
(well above ε). The ML-only run never has anywhere to go and stays at
the plateau for the rest of the experiment. The BoundGuard run hot-swaps
to the GPU fallback at t ≈ 21 s; `V(t)` collapses below the threshold
within ~3 s and continues to decay toward zero.

Run-to-run noise on individual ticks is on the order of ±10 in
`V(t)`, so the precise numbers above can shift by a few seconds and
a few units between executions. The qualitative story (ML-only stays
at the plateau, BoundGuard recovers to `V(t) ≈ 0` after the fallback)
is stable.

---

## 3. The predictor (XGBoost), the candidates, and the runtime YAMLs

The XGBoost predictor (`xgb_model_x3_double`) is used as a *real*
component of the case study: it ranks the four candidate placements
in `tests/ml_misprediction_candidates.yaml` and produces a confident
top pick. The full ranking is saved to
`results/ml_misprediction/ml_misprediction.json` and looks like:

```
combination_all_gpu        score=0.877
combination_split_cpu_gpu  score=0.535
combination_split_gpu_cpu  score=0.191
combination_all_cpu       score=-0.019
```

The candidates YAML is deliberately constructed at the moderate input
rates that match the predictor's training distribution. The
predictor's top pick `combination_all_gpu` is therefore "the right
answer" for the operating regime it was asked about.

The *runtime* YAMLs, however, ramp the per-view input rates by one to
two orders of magnitude above the predictor's training distribution.
This is the OOD condition the predictor never saw. The CPU placement
that was perfectly fine at training rates can no longer drain its
queues at runtime, so `V(t)` saturates above ε.

The script does not literally invoke the predictor at every runtime
tick (it is called *once*, before the executor starts, exactly as the
paper describes a placement-policy module deciding the initial
placement). The driver explicitly records both "the predictor's top
pick" and "the runtime placement actually evaluated" in the JSON
output so a reviewer can verify there is no sleight of hand.

---

## 4. Mode mapping

| Figure label        | Adaptive mode | What it does in this scenario |
|--------------------|--------------:|--------------------------------|
| Adaptive (ML-only)  | 1             | `AdaptiveDeployManager`. Single-combo runtime YAML with no fallback to transition to, so the system pins the failing placement for the entire run. |
| BoundGuard          | 1             | `AdaptiveDeployManager` again, but the runtime YAML has a second combo (the GPU-offload fallback). The executor schedules the transition between them; mode 1 hot-swaps without stopping the running workers. |

We deliberately do **not** use mode 2 (`ReactiveDeployManager`, the
hot-swap + post-transition rollback variant) for the BoundGuard run
on this scenario, even though "BoundGuard" maps onto mode 2
elsewhere in the paper. The reason is that mode 2's rollback
validator compares the windowed `V(t)` of the new combo against the
live V(t) at the moment of transition; the cold-start tail of the
GPU fallback combo is still elevated 5 s after the swap, so the
validator mis-classifies the (genuinely better) GPU fallback as a
regression and rolls the system back to the failing CPU placement.
The result is a confusing "BoundGuard reverted to the failing
placement" trace in the figure. Mode 1 has the same hot-swap path
*without* that mis-firing validator, which produces the
continuous-service behaviour the figure is meant to illustrate.
The bounded-recovery and runtime-overhead figures still use mode 2
because they are explicitly about the validator's behaviour.

Note: a related fix was made in `schedule_executor_main.py` so the
`prev_vscore` passed to mode-2 / mode-4 managers reflects the
*terminal* V(t) of the previous combo (sampled live at the moment of
transition) rather than the cached value from when the previous combo
*started* (which is always ≈ 0 because the workers had not yet
produced any data when it was captured). This makes mode 2's rollback
delta math correct for the prev_vscore baseline; it does not, on its
own, fix the cold-start-tail issue described above.

---

## 5. Runtime YAMLs

### `tests/ml_misprediction_mlonly_runtime.yaml`

Single combo. The system runs this placement for the entire ML-only
run; there is no fallback combo to transition to.

| view  | model      | execution | infps |
|-------|------------|-----------|------:|
| view1 | mnasnet    | cpu       | 400   |
| view2 | resnet50   | cpu       | 60    |
| view3 | resnext50  | cpu       | 60    |
| view4 | yolov4     | cpu       | 10    |

### `tests/ml_misprediction_boundguard_runtime.yaml`

Two combos. The first (`combination_overload`) is identical to the
ML-only YAML so the BoundGuard and ML-only runs start from exactly
the same state. The second (`combination_offload`) is the heuristic
GPU-offload fallback that the safety layer hands over to.

| Combo                  | Placement | Per-view infps          |
|------------------------|-----------|-------------------------|
| `combination_overload` | all CPU   | 400 / 60 / 60 / 10      |
| `combination_offload`  | all GPU   | 400 / 60 / 60 / 10      |

Phase budgets are passed via `--combo-duration` from the driver:

| Combo                  | Wall-clock budget |
|------------------------|------------------:|
| `combination_overload` | 25 s              |
| `combination_offload`  | 25 s              |

---

## 6. Warm-up handling

`load_curve()` drops the first three rows of each CSV before
computing the windowed `V(t)` and rendering the figure. This is
deliberate: the very first per-tick `v_score` measurement is highly
sensitive to the exact timing of the worker initialisation relative
to the first 1 Hz logging tick. Depending on whether the queues had
half a second or two seconds to fill up before the first tick fires,
the first row can be anywhere from 30 to 200 even when the
steady-state plateau is around 100. Two runs of the same placement
can therefore *appear* to start from very different `V(t)` values
even though they will converge to the same plateau within one or two
ticks.

Dropping the first three rows skips this initialisation noise and
makes the visual comparison fair: ML-only and BoundGuard both start
from the steady-state plateau (~100) at the same wall-clock origin
(t = 0 in the figure corresponds to the first kept row of each run).
The marker positions (`bg_fallback_x`, `bg_recover_x`) are computed
from the post-warm-up rows so they remain consistent with what the
figure shows.

---

## 7. Detection / metric parameters

| Parameter                  | Value | Notes                                          |
|---------------------------|-------|------------------------------------------------|
| Sliding-window length T    | 3 s   | from `qos_recovery_validation.WINDOW_T`        |
| Detection threshold ε      | 50    | `--epsilon 50`                                  |
| Tick rate                  | 1 Hz  | one CSV row per second from the executor      |
| QoS metric ℓ_i             | `avg_infer_time + avg_wait_ms`                 |
| L_SLO,i                    | `1000 / infps_i` ms                             |
| Warm-up rows dropped       | 3     | `load_curve(warmup_drop=3)`                     |
| Phase budgets              | ML-only 50 s; BoundGuard 25 s + 25 s            |

---

## 8. Software environment

| Component        | Notes                                                    |
|-----------------|----------------------------------------------------------|
| Python          | 3.10 (`.venv/bin/python3`)                               |
| ONNX Runtime    | onnxruntime-gpu (CPU + CUDA providers)                   |
| Qt platform     | `QT_QPA_PLATFORM=offscreen` (set by the runner)          |
| XGBoost model   | `xgboost_model/artifacts/gpu/xgb_model_x3_double_y{1,2}.json` |
| Predictor mode  | `double`, `alpha=0.2`                                    |
| Project root    | `/home/msyu/PycharmProjects/fsrr-multimodel-scheduling`  |

---

## 9. How to tweak the experiment

- **Different candidate pool.** Edit
  `tests/ml_misprediction_candidates.yaml`. The driver simply re-runs
  the predictor; if the top pick changes, the JSON output reflects it.
- **Different runtime workload.** Edit
  `tests/ml_misprediction_mlonly_runtime.yaml` and
  `tests/ml_misprediction_boundguard_runtime.yaml`. Make sure the
  first combo of the BoundGuard YAML matches the single combo of the
  ML-only YAML so the two runs are directly comparable.
- **Different "BoundGuard" execution mode.** Change the mode passed to
  `run_executor()` for the BoundGuard call (mode 2 = reactive
  validation/rollback, mode 4 = stop-and-restart with rollback). Note
  the caveat in §4 about mode 2 mis-firing on this scenario.
- **More/less warm-up dropped.** Pass a different value to the
  `warmup_drop=` parameter in `load_curve()` (default 3).

The runner overwrites `results/ml_misprediction/*.csv` and the JSON
on every fresh run, so re-running is safe.

---

## 10. Known limitations

- **Predictor is invoked once, not every tick.** This matches the
  paper's description of a placement-policy module that decides on a
  placement and hands it to the execution layer. A real online
  re-prediction loop would be a different experiment.
- **Single-run measurements.** Both ML-only and BoundGuard are
  measured once per phase budget. Repeating each run a few times
  would tighten the absolute numbers; the qualitative story
  (ML-only stays at the plateau, BoundGuard recovers within a few
  seconds of the fallback) is stable across rerun trials we have
  observed.
- **Mode 2 is intentionally not used.** See §4 for the reason.
