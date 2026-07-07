# Reproduction: `fig:sweeps` (Fig. 3 — `fig:sweep` + `fig:split`)

Regenerates and cross-checks the two subfigures of `fig:sweeps` in
`paper/main_vision.tex`:

- **`fig:sweep`** (left): homogeneous placement, worst-stream sAP vs. GPU
  deadline-miss rate for co-tenancy **N = 2, 4, 8** (All-GPU), with the flat
  **All-NPU** baseline overlaid.
- **`fig:split`** (right): at **N = 4**, the three placement policies
  **All-GPU / All-NPU / Oracle** vs. GPU deadline-miss rate.

Caption summary: worst-stream sAP degrades as ResNet50 co-tenant contention
raises the GPU deadline-miss (skip) rate; the All-GPU and All-NPU curves
**cross over at ~48% GPU skip**, and the Oracle policy tracks the better of the
two on each side of the crossover.

## IMPORTANT — this figure has NO external PDF

`fig:sweeps` is drawn **inline** in `main_vision.tex` as pgfplots/TikZ with
hardcoded `\addplot ... coordinates {...}` blocks. There is no external image to
overwrite. Therefore "reproduction" here means:

1. **Regenerate** the `\addplot` coordinate blocks from the source data, and
2. **Cross-check** every regenerated `(x, y)` coordinate against the paper's
   inline TikZ coordinates (coordinate-by-coordinate).

The verification result is the cross-check verdict printed by `generate.py`.

## Input data (`./data/`)

Copied verbatim from `accv_experiments/results/rev30_clean_resnet/`:

- **`rev30_fig3a_points.csv`** — per-N points for `fig:sweep`. Columns:
  `N, k, gpu_skip_mean, allgpu_worst(+std), allnpu_worst(+std)`.
  Each row is a ResNet50 contention level `k` giving a measured GPU skip
  (deadline-miss) rate `gpu_skip_mean` and the resulting All-GPU (and, where
  present, All-NPU) worst-stream sAP.
- **`rev30_oracle_by_contention.csv`** — N=4 points for `fig:split`. Columns
  include `gpu_skip_allgpu, allgpu_worst, allnpu_worst, oracle_worst`
  (plus per-split diagnostics). One row per contention level `RES{k}`.

## Run

```bash
/home/msyu/PycharmProjects/multimodel-scheduling-video/.venv/bin/python generate.py
```

## Outputs (written in this folder)

- **`fig2_sweep.tex`** — the regenerated `\addplot` coordinate blocks for both
  subfigures (the artifact that mirrors the inline paper TikZ).
- **`sweep_byN_points.csv`** — copy of the `fig:sweep` input points.
- **`oracle_by_contention.csv`** — copy of the `fig:split` input points.
- `*_stale_backup.*` — one-time backup of any pre-existing output.

## Transform

- **`fig:sweep`**: for each N ∈ {2, 4, 8}, filter rows with a defined
  `allgpu_worst`, sort by `gpu_skip_mean`, and emit `(gpu_skip_mean,
  allgpu_worst)` coordinates. **All-NPU** is drawn as a single flat dashed line
  at the **mean** of all defined `allnpu_worst` values (from `(0, mean)` to
  `(84, mean)`).
- **`fig:split`** (N=4): emit `(gpu_skip_allgpu, y)` coordinates for `y ∈
  {allgpu_worst, allnpu_worst, oracle_worst}`, in file order.
- No computation logic is changed from the source generator
  `paper/results_data/regen_fig2_sweep.py`; only path constants were adjusted.

## Verification

`generate.py` prints the N=4 All-GPU and Oracle coordinate lists, then compares
every regenerated coordinate (both subfigures) against the paper's inline TikZ
coordinates (tolerance ±0.15 in x, ±5e-4 in y) and prints:

```
coordinate cross-check vs paper tikz: PASS
```

**PASS** = the regenerated coordinates match the paper's hardcoded inline TikZ
coordinates.
