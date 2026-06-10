# Paper figure/table reproduction package — MANIFEST

Reproduces every figure and table in `paper/main_vision.tex` from existing
measurement CSVs. **No new measurement** — all values extracted from
`accv_experiments/results/rev*.csv`. Core experiment scripts, `results/rev*`,
and the paper body (`main_vision.tex`) are **read-only / unchanged**.

> Note on location: the spec named `accv_experiments/paper/`, but the LaTeX paper
> actually lives at the repo-root `paper/`, so this package is at `paper/results_data/`.

## Operating point (all artifacts)
RTX 5090 GPU + Mobilint MLA100 NPU · YOLOv11s · Argoverse-HD (24 logs) ·
NPU binary mxq `b2441f9d`, infer_mode `global8` · both devices `torch.set_num_threads(4)`.

## Reproduce
```
cd paper/results_data
python regen_all.py        # 6 generators + cross-check vs main_vision.tex + refresh index.html
```
Last run: **6/6 artifacts reproduced and matched paper values.**

## Map: paper label → folder → source → generator → output

| paper label | folder | source data (rev) | generator | output |
|---|---|---|---|---|
| `fig:failures` (Fig.1) | `fig1_failures/` | `phase_vis_failure_examples.py` frames (sids 2,13,17,22) | `generate_fig1.py` | `vis_failure_examples.pdf` |
| `fig:sweep` (Fig.2) | `fig2_sweep/` | `rev24_sweep_byN_points.csv` (21 pts, 3 reps) | `generate_fig2.py` | `fig2_sweep.tex`, `.png` |
| `tab:single-stream` (Table 1) | `table1_single_stream/` | `rev19_table1_threads4.csv` (24 logs, 3 reps, skip≈0) | `generate_table1.py` | `table1.tex` |
| `tab:decomp` (Table 2) | `table2_decomp/` | `rev18_postproc_levers.csv` + `rev22_bitident.csv` | `generate_table2.py` | `table2.tex` |
| `tab:main` (Table 3) | `table3_main/` | `rev21_mean_worst_extract.csv` + `rev20_5strat_heavybg.csv` (N=4, 3 reps) | `generate_table3.py` | `table3.tex` |
| `tab:persize-contention` (Table 4) | `table4_persize/` | `rev25_persize_under_contention.csv` (3 reps) | `generate_table4.py` | `table4.tex` |

## Per-artifact conditions
- **Table 1**: threads=4 single-stream L0, frame skip ≈0. Per-size sAP exact match. (Infer GPU measured 8.3 ms; paper rounds rev18's 8.55→8.5 — a provenance detail; per-size sAP is exact.)
- **Table 2**: quantization = threads=4 NPU−GPU per-size gap; staleness = (threads=24 gap − threads=4 gap). Bit-identical detections: 20 frames, box/score diff 0.00e+00. *offline mAP is NOT thread-invariant (penalizes skip as zero recall) — causal separation rests on bit-identical detections + skip 0 vs 52% contrast, not mAP invariance.*
- **Table 3**: N=4, worst/mean sAP, threads=4. Bold = L2 All-GPU worst, L3 All-NPU worst (matches paper). Partial placements (Isolated/Cont-aware) kept in source CSVs but not in the table (paper prose only).
- **Table 4**: All-GPU per-size Δ vs uncontended (N=4). All-NPU per-size invariance ≤0.001 over the range.
- **Fig.2**: x = measured GPU frame-skip%; All-GPU N=2/4/8 curves + All-NPU flat 0.0836; crossover ≈48%. NPU skip ≤0.1% (N=2,4).
- **Fig.1**: qualitative; needs GPU+NPU device to re-render (not pure extraction). PDF copied byte-for-byte; selected frames in `fig1_failures/data/selected_frames.csv`.

## Rounding
3-decimal sAP, 1-decimal %, **round-half-up** (paper convention) — matters only at the
0.0005 boundary (Table 1 large diff +0.0005→+0.001; Table 4 medium Δ 0.0245→0.025).

## Cross-validation
Each generator reads `paper/main_vision.tex` and asserts its generated numbers appear in the
corresponding table/figure block; mismatches are reported (the paper is never modified).
`regen_all.py` aggregates a PASS/FAIL matrix and refreshes `index.html`.

## Files not measured here
Everything is extraction. Source rev files: rev18, rev19, rev20, rev21, rev22, rev24, rev25
(+ `phase_vis_failure_examples.py` for Fig.1). See each generator header for exact columns.
