# FINDINGS — Co-tenant behind Figure `fig:sweep` (read-only verification)

Scope: identify, from code + logs, exactly what background workload generates the
GPU-contention sweep in **Fig. 2a (`fig:sweep`)** of `paper/main_vision.tex`
(All-GPU N=2,4,8 vs All-NPU worst-stream sAP vs GPU frame skip). No paper, code,
config, or result file was modified.

## Verdict (one line)

The `fig:sweep` co-tenant is **ResNet50** (ORT-CUDA, GPU-bound), and the sweep
knob is the **number of independent ResNet50 inference loops co-located on the
GPU**, `k ∈ {0,1,2,3,4,6,8}`. It is the **same** co-tenant and mechanism behind
`fig:regime`. The NPU path stays unsaturated across the sweep (NPU frame skip
≤0.1% for N=2,4; ≤4.7% at N=8).

---

## 1. Which model is the GPU co-tenant — ResNet50 (verified)

- Producer script header states it verbatim:
  `accv_experiments/scripts/phase_rev22_sweep.py:3-6`
  > "Contention lever: number of ResNet50 GPU-loops (0..k) registered at runtime
  > into BG_VARIANTS … ResNet50 is GPU-bound (ORT-CUDA), minimal host-CPU →
  > isolates the GPU-contention axis, keeping the NPU host postprocess fast
  > (avoids the L2_lm CPU-confound)."
- The loop body is `_bg_resnet50_loop`, imported at
  `phase_rev22_sweep.py:30` (`from step_h2_robustness import _bg_resnet50_loop`),
  which re-exports it from `_step_d_common.py`.
- Definition `accv_experiments/scripts/_step_d_common.py:342-345`:
  ```python
  def _bg_resnet50_loop(stop_event):
      sess, feed = _BG_PRELOADED["resnet"]
      while not stop_event.is_set():
          sess.run(None, feed)
  ```
  The session is `ort.InferenceSession("models/onnx/resnet50.onnx",
  providers=["CUDAExecutionProvider", ...])` (`_step_d_common.py:298`), fed a
  `(1,3,224,224)` tensor — a pure GPU-resident ResNet50 inference loop.
- The run preloads **only** ResNet50: `phase_rev22_sweep.py:109`
  `preload_background_models(max_level="L1")  # resnet50 only`. (`L1` in
  `_step_d_common.py:293` = `["resnet"]`; no TinyLLaMA/Qwen-VL loaded.)

Not assumed — verified in code. No other model participates in this sweep.

## 2. How contention is swept — number of ResNet50 instances `k`

- Sweep grid: `accv_experiments/scripts/phase_rev22_sweep.py:38`
  `RESNET_COUNTS = [0,1,2,3,4,6,8]   # GPU-pressure lever`.
- The lever registers `k` copies of the ResNet50 loop as one background level:
  `phase_rev22_sweep.py:47-50`
  ```python
  def reg_level(k):
      name = f"SWEEP{k}"
      h2.BG_VARIANTS[name] = [_bg_resnet50_loop]*k   # k concurrent ResNet50 loops
      return name
  ```
  Loop over `k` at `:128` (`for k in RESNET_COUNTS:`), measured via
  `measure_multistream(..., lvl)` at `:137`.
- **The plotted x-axis is the *measured* GPU frame skip%, not `k` directly.**
  Each cell records `gpu_skip` / `npu_skip` (`dev_skips`, `:52-55`; written at
  `:139-142`). Increasing `k` from 0→8 moves measured GPU frame skip from ≈0% to
  ≈80%.
- **Within one curve, foreground stream count `N` is fixed; only `k` is varied.**
  Curves are `(N, sids, strats)` tuples `N∈{2,4,8}` (`:117-119`); `k` is swept
  inside each (`:128`). So `N` selects the curve, `k` (ResNet50 instances) moves
  you along it. The foreground stream count is **not** the sweep axis within a
  curve.

This contradicts a "single ResNet50, varied batch size / synthetic generator"
reading: it is literally *more ResNet50 processes-loops*, batch size fixed at 1.

### Provenance of the exact `fig:sweep` coordinates

The tikz points in `main_vision.tex:331-333` come from
`accv_experiments/results/rev24_sweep_byN_points.csv`, an extraction (0 new
measurements) of the rev22 sweep — confirmed by `results/rev24_report.md`:
> "All points from `rev22_vlm_sweep.csv` … ResNet50 GPU-pressure lever, 3
> reps/point. x-axis = measured GPU skip%."

Exact match (N=4, orange curve `main_vision.tex:333`):

| `resnet_k` | GPU skip% | All-GPU worst sAP (CSV) | tikz point |
|---|---|---|---|
| 0 | 0.1 | 0.1149 | (0.1, 0.1149) |
| 1 | 42.0 | 0.0879 | (42, 0.0879) |
| 8 | 79.2 | 0.0663 | (79.2, 0.0663) |

(CSV rows: `rev24_sweep_byN_points.csv:8,9,14`.) N=2 red curve
(`main_vision.tex:331`) matches `rev24_sweep_byN_points.csv:2-7` identically.

## 3. Same co-tenant as `fig:regime`? — Yes (verified)

- `fig:regime` text (`main_vision.tex:413`) says it sweeps GPU contention with
  "a **stable ResNet50 co-tenant**."
- Its data is `results/rev28_oracle_gap_sweep.csv`, produced by
  `accv_experiments/scripts/phase_rev28_analysis.py`, which **reads the same
  rev22-derived file**: `phase_rev28_analysis.py:70`
  `df = pd.read_csv(RES / "rev24_sweep_byN_points.csv")`. The oracle gap rows
  share the identical GPU-skip grid (e.g. `rev28_oracle_gap_sweep.csv` has
  `0.1,0.1149` / `42.0,0.0879` / `79.2,0.0663`).
- Therefore `fig:sweep` and `fig:regime` are driven by the **same ResNet50
  co-tenant and the same `k`-instance GPU-pressure lever**. (The `fig:regime`
  moderate point at "GPU frame-skip ≈38%" is a point on this same ResNet50 grid;
  the N=4 `k=1` cell measures 42% GPU skip.)

## 4. Does the sweep leave the NPU/CPU path unsaturated? — Yes

NPU frame skip across the entire sweep (from `rev24_sweep_byN_points.csv`,
`npu_skip` column):

| Curve | NPU frame-skip range across sweep |
|---|---|
| N=2 | 0.0% at every point |
| N=4 | 0.0%–0.1% |
| N=8 | 0.2%–4.7% |

So NPU frame skip stays **≤0.1% for N=2 and N=4**, and **≤4.7% even at N=8**,
while GPU frame skip ranges ≈0→88%. This supports the "contention localized to
the GPU path" claim. Mechanistically consistent with `phase_rev22_sweep.py:5-6`
(ResNet50 is GPU-bound, minimal host-CPU, so the NPU host post-process stays
fast) and the rev22 sanity check (`results/rev24_report.md`: "NPU infer ~14 ms,
skip 0%, threads=4 held").

**CPU-util caveat:** host CPU utilization is **not numerically logged** in the
rev22/rev24 result files (no CPU% column). The 17%→93% CPU-util figures in
`main_vision.tex:358` belong to the **LM co-tenant** (Table 1 / `tab:main`),
*not* to this ResNet50 sweep. So I report NPU frame skip as the (logged)
GPU-localization evidence; CPU% for this specific sweep was not recorded.

---

## Recommended sentence (exact-as-implemented wording)

The `fig:sweep` discussion paragraph is at **`main_vision.tex:360`** (note: the
task referenced ~line 408, which is the `fig:regime` paragraph; the wording
below is intended for the `fig:sweep` paragraph at ~360 — adjust placement as
desired). Suggested insertion, matching the code exactly:

> GPU contention is generated by co-locating $k$ independent ResNet50 inference
> loops (ONNX Runtime, CUDA) on the GPU, with $k\in\{0,1,2,3,4,6,8\}$; each curve
> fixes the foreground stream count $N$, and increasing $k$ sweeps the measured
> GPU frame skip from ${\approx}0\%$ to ${\approx}80\%$ while the NPU path stays
> unsaturated (NPU frame skip ${\le}0.1\%$ for $N{=}2,4$ and ${\le}4.7\%$ at
> $N{=}8$).

If a shorter form is preferred:

> Contention is swept by increasing the number of co-located ResNet50 GPU loops
> ($k=0\ldots8$) at fixed foreground stream count $N$; the $x$-axis is the
> resulting measured GPU frame skip.

(Do **not** write "by increasing batch size" or "synthetic load" — the
implementation increases the *count of ResNet50 inference loops*, batch size
fixed at 1.)

---

## What was searched (audit trail)

- `grep` for tikz values `0.1149 / 0.0879 / 0.0663` across `*.py/*.csv/*.json/*.dat/*.tex`
  → matched `rev24_sweep_byN_points.csv` (exact `fig:sweep` points) and
  `rev28_oracle_gap_sweep.csv`.
- Traced producer via `grep resnet_k` → `phase_rev22_sweep.py` (the sweep
  driver); read it in full (`:1-160`).
- Confirmed `_bg_resnet50_loop` source in `_step_d_common.py:298,342-345`.
- Confirmed `fig:regime` linkage via `phase_rev28_analysis.py:70` reading the
  same file, and `main_vision.tex:413` text.
- Confirmed NPU-skip range from the `npu_skip` column of
  `rev24_sweep_byN_points.csv` (21 points) and `rev24_report.md`.

**Inference vs. found:** all four answers (co-tenant=ResNet50, lever=ResNet50
instance count, same as fig:regime, NPU unsaturated) are **found in code/CSV**,
not inferred. The only *inference* is that the absence of a CPU% column means
CPU-util for this sweep was not logged (stated as a caveat, not a claim that CPU
was unsaturated — though NPU-skip≈0 is consistent with it).
