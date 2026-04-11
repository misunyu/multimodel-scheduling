# Dynamic Load Adaptation Validation — Reproducibility Notes

This document records every input, parameter, and assumption that goes
into `results/dynamic_load_adaptation.pdf` so the figure can be
regenerated exactly.

| Component                        | Path                                                       |
|---------------------------------|------------------------------------------------------------|
| Schedule (3 phases, 4 views)    | `tests/dynamic_load_views_schedule.yaml`                  |
| Runner / plot script            | `scripts/dynamic_load_validation.py`                      |
| Per-tick / windowed V(t) helpers| `scripts/qos_recovery_validation.py` (reused)             |
| BoundGuard (mode 0) CSV         | `results/dynamic_load_boundguard.csv`                     |
| Static (mode 3) CSV             | `results/dynamic_load_static.csv`                         |
| Final figure                    | `results/dynamic_load_adaptation.pdf`                     |

---

## 1. End-to-end reproduction

```bash
# Run BoundGuard (mode 0) and Static (mode 3) and render the figure.
python scripts/dynamic_load_validation.py --epsilon 50

# Replot from the existing CSVs without re-running the workload.
python scripts/dynamic_load_validation.py --no-run --epsilon 50
```

The script:
- runs `schedule_executor_main.py --adaptive-mode 0` to produce
  the BoundGuard trace,
- runs `schedule_executor_main.py --adaptive-mode 3` to produce
  the Static trace,
- loads both CSVs, computes the windowed `V(t)` (T = 3 s) from the
  per-tick `v_score` column,
- inserts NaNs at cold-start gaps so the line breaks visibly,
- marks the load-change event (start of phase B in the BoundGuard CSV)
  with a vertical dotted line,
- marks the BoundGuard recovery point (first phase-C tick with
  windowed `V(t) ≤ ε`) with another dotted line,
- saves the PDF and prints a textual summary.

Cold-start gap detection is filtered to only count gaps that *straddle
a phase boundary* — pure missing ticks inside a single phase (GC
pauses, scheduler jitter) are ignored so the figure does not sprout
fake "rollback in progress" hatching in the recovered region.

---

## 2. Scenario description

Schedule file: `tests/dynamic_load_views_schedule.yaml`

| Phase | Combination key       | Placement | Input rates                       | Wall-clock budget |
|------:|----------------------|----------|----------------------------------|-------------------|
|   A   | `combination_p1_low`  | all CPU  | mnasnet 3 / resnet50 3 / resnext50 3 / yolov4 1 | 22 s |
|   B   | `combination_p1_high` | all CPU (same as A) | mnasnet 400 / resnet50 60 / resnext50 60 / yolov4 10 | 6 s |
|   C   | `combination_p2_high` | all GPU  | mnasnet 400 / resnet50 60 / resnext50 60 / yolov4 10 | 22 s |

Phase A → Phase B is processed by `schedule_executor_main.py`'s
same-placement detection: the placement signature is identical, so
no workers are stopped — only the per-view `infps` is swapped in
place. This is what makes the load-change event appear as a clean,
instantaneous infps jump rather than a cold-start spike.

Phase B → Phase C is a real placement change (CPU → GPU) and goes
through the regular stop-and-restart path. The cold-start window
between the two phases is the visible reconfiguration interval shown
on the figure.

The actual wall-clock budgets above leave a small slack at the start
of phase A so the executor's worker-init time does not eat into the
visible portion of phase A. In practice the load change shows up at
about t = 18 s on the figure (not the nominal 20 s) because phase A
loses ~4 s to the initial worker construction.

---

## 3. View models

Same four view models as the bounded-recovery experiment so the two
figures share a consistent workload baseline:

| view  | model      | initial (CPU) | overload (CPU) | offload (GPU) |
|-------|------------|--------------:|---------------:|--------------:|
| view1 | mnasnet    | infps 3       | infps 400      | infps 400     |
| view2 | resnet50   | infps 3       | infps 60       | infps 60      |
| view3 | resnext50  | infps 3       | infps 60       | infps 60      |
| view4 | yolov4     | infps 1       | infps 10       | infps 10      |

(Note: as in the bounded-recovery experiment, the YAML uses
`model: yolov4` but the underlying `run_yolo_*_process` handlers in
`model_processors.py` resolve the actual ONNX file via `yolov3_small`.
This is an existing convention in the codebase.)

---

## 4. Mode mapping

| Figure label | Adaptive mode | What it does in this scenario |
|-------------|--------------:|--------------------------------|
| Static      | 3             | Keeps the first combination's placement (P1 = all CPU) for the entire run; phase transitions only swap `infps`. The system therefore *cannot* react to the load change. |
| BoundGuard  | 0             | Stop-and-restart between phases. Same-placement detection collapses A → B into an in-place infps swap; B → C is a real reconfiguration that brings up the GPU workers. |

The "BoundGuard" label in the figure is the conceptual name used in
the paper for the mechanism that detects instability and reconfigures
the placement. Mode 0 is the underlying execution path used to
measure the resulting V(t) trajectory in this experiment.

---

## 5. Detection / metric parameters

| Parameter                   | Value | Notes                                          |
|----------------------------|-------|------------------------------------------------|
| Sliding-window length T    | 3 s   | from `qos_recovery_validation.WINDOW_T`        |
| Detection threshold ε      | 50    | `--epsilon 50`                                  |
| Tick rate                  | 1 Hz  | one CSV row per second from the executor      |
| QoS metric ℓ_i             | `avg_infer_time + avg_wait_ms`                 |
| L_SLO,i                    | `1000 / infps_i` ms                             |
| Cold-start gap threshold   | 1.5 s (gap must also straddle a phase boundary) |
| Phase budgets              | A 22 s, B 6 s, C 22 s                          |

The per-tick v(τ) and the windowed V(t) are computed exactly as in
the bounded-recovery experiment:

```
v(τ)  = (1/N_active) · Σ_i max(0, ℓ_i(τ) / L_SLO,i − 1)
V(t)  = (1/T) · Σ_{τ=t−T+1..t} v(τ)
```

---

## 6. Latest measurements (2026-04-11)

Output of `python scripts/dynamic_load_validation.py --no-run --epsilon 50`:

```
epsilon                    = 50.0
Load-change wall-clock     = 18.00 s   (start of phase B)
BoundGuard cold-start gaps = 2.0 s @ 23.0–25.0 s   (phase B → phase C)
BoundGuard V(t) max        = 214.94    (instant cold-start spike)
BoundGuard V(t) end        = 0.16
BoundGuard t_stable        = 27.00 s   (first V(t) ≤ ε in phase C)
Static V(t) max            = 83.03
Static V(t) end            = 83.03     (does not recover)
Static rows                = 44
BoundGuard rows            = 47
```

Derived numbers used in the paper text:

| Quantity                                              | Value     |
|------------------------------------------------------|----------:|
| Time from load change to BoundGuard recovery         | ~9 s      |
| Reconfiguration window (cold-start gap, phase B→C)   | 2 s       |
| Static plateau V(t) (~ end of phase C)               | ~83       |
| BoundGuard end-of-run V(t)                           | ~0.2      |
| Ratio Static/BoundGuard at end of run                | ≈ 500×    |

The static curve climbs from V(t) ≈ 0 in phase A to a plateau around
V(t) ≈ 83 in phase C and stays there. The plateau is bounded by the
fixed input queue depth (`maxsize=2`), which caps how much wait time
can accumulate per request — the curve is therefore *flat-topped*
rather than diverging linearly. This is an architectural artefact of
the executor and not a property of the adaptation algorithm.

---

## 7. How to tweak the experiment

- **Change the load-change instant.** Edit `PHASE_A_DURATION` in
  `scripts/dynamic_load_validation.py`. The actual visible load change
  on the figure will land at roughly `PHASE_A_DURATION − 4 s` due to
  initial worker construction.
- **Vary the load step size.** Change the per-view `infps` in
  `combination_p1_high` / `combination_p2_high` inside the YAML.
- **Use a different recovery placement.** Change
  `combination_p2_high` to put a different subset of models on the
  GPU.
- **Compare a different mode as "BoundGuard".** Pass a different
  `--adaptive-mode` value when calling `schedule_executor_main.py`
  inside `run_scenario()` (e.g. mode 2 for the reactive
  hot-swap+rollback path or mode 4 for stop-and-restart with rollback).

The runner overwrites `results/dynamic_load_*.csv` on every run, so
re-running is safe.

---

## 8. Software environment

| Component        | Notes                                                    |
|-----------------|----------------------------------------------------------|
| Python          | 3.10 (`.venv/bin/python3`)                               |
| ONNX Runtime    | onnxruntime-gpu (CUDAExecutionProvider available)        |
| Qt platform     | `QT_QPA_PLATFORM=offscreen` (set by the runner)          |
| Project root    | `/home/msyu/PycharmProjects/fsrr-multimodel-scheduling`  |
| Schedule file   | `tests/dynamic_load_views_schedule.yaml`                 |

---

## 9. Known limitations

- **Static V(t) plateaus instead of diverging** because the executor
  uses a bounded request queue (`maxsize=2`). The figure still tells
  the right story — Static stays *above* ε for the rest of the run —
  but the visual "monotonic climb" stops once queues saturate.
- **The load change does not land exactly at t = 20 s** because phase
  A loses ~4 s to initial worker construction. The runner extracts
  the actual phase-B start time from the CSV and marks it on the
  figure.
- **`t_stable` is granular at 1 second** (1 Hz CSV tick rate). The
  true recovery moment can fall anywhere inside that 1-second
  interval.
- **Mode 0 is used as the "BoundGuard" execution path.** This was
  chosen because the conceptual story (detect instability, reconfigure)
  maps cleanly onto a stop-and-restart transition. The reactive mode
  (mode 2) and the new restart-with-rollback mode (mode 4) implement
  more elaborate validation/rollback policies; reproducing them in
  this figure is a one-line change in `run_scenario()`.
