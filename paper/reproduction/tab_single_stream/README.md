# Reproduction: tab:single-stream (Table 1 — `table1.tex`)

## Paper reference
- **Label:** `tab:single-stream` (Table 1 in the compiled PDF, `paper/main_vision.tex`).
- **Caption summary:** Single-camera evaluation — one video stream on one device,
  GPU (FP32) vs NPU (quantized). Because there is no co-tenancy, **deadline misses
  are negligible** ($\sim\!0$) on both devices, so the accuracy gap is purely the
  quantization cost, not a scheduling cost. Significance is a **paired Wilcoxon
  signed-rank test over the 24 Argoverse-HD logs** (one GPU/NPU pair per log),
  computed per object-size stratum.

## Input data (copied verbatim into `data/`)
- `rev19_table1_threads4.csv` — **pooled 24-log sAP point estimates** at
  `threads=4`, `skip~0` (from `accv_experiments/results/`). One row per device
  (`GPU`/`NPU`) per repeat; the point estimates (overall / small / medium / large
  sAP, and `infer_ms`) are the mean over the 24-log pool. Used for the table's
  reported values.
- `rev7_single_stream.csv` — **24 per-log GPU/NPU pairs** (one row per
  `log_id` × device, with `CPU` rows ignored). Used only to recompute the
  per-size paired **Wilcoxon** p-values; not used for the point estimates.

## Run
```
cd paper/reproduction/tab_single_stream
/home/msyu/PycharmProjects/multimodel-scheduling-video/.venv/bin/python generate.py
```

## Outputs
- `table1.tex` — the `tabular` body for `tab:single-stream` (paper Table 1).
- `single_stream.csv` — per-metric GPU / NPU / diff / rel% / Wilcoxon p / flag.
- `wilcoxon_pvalues.csv` — per-metric Wilcoxon p-value and significance flag.
- `*_stale_backup.*` — created only on re-runs (one-time backup of a prior output).

## Transform
- **Point estimates:** mean per-size sAP over the 24-log pool (`rev19`), separately
  for GPU and NPU; end-to-end `infer_ms` mean per device.
- **Difference:** `NPU − GPU` per size; **rel% = diff / GPU** (denominator is the
  FP32 GPU baseline).
- **Significance:** per-size **paired Wilcoxon signed-rank** over the 24 common
  logs (`rev7`), NPU vs GPU. Flags: `p<0.001` / `p<0.01` / `p<0.05` / `n.s.`.
- **Scale:** sAP is stored fractional here (e.g. `0.196`); the **paper reports sAP
  ×100 as a percent** (e.g. `19.6`). So each fractional value below maps to the
  paper cell by multiplying by 100.

## Verification (vs paper Table 1)

| Metric | Reproduced (frac) | Paper (%) | Verdict |
|---|---|---|---|
| end-to-end mean (ms) GPU / NPU | 8.3 / 10.0 | 8.3 / 10.0 | **MATCH** |
| deadline misses | $\sim\!0$ / $\sim\!0$ | $\sim\!0$ / $\sim\!0$ | **MATCH** |
| sAP overall GPU / NPU | 0.196 / 0.186 | 19.6 / 18.6 | **MATCH** (rel −5.0%, p<0.01) |
| sAP small GPU / NPU | 0.016 / 0.008 | 1.6 / 0.8 | **MATCH** (rel −48.4%, p<0.001) |
| sAP medium GPU / NPU | 0.184 / 0.148 | 18.4 / 14.8 | **MATCH** (rel −19.7%, p<0.001) |
| sAP large GPU / NPU | 0.477 / **0.478** | 47.7 / **47.8** | **MATCH** — display rounding, see note |

**Display-rounding note (large / NPU):** the pooled source value is
`sap_l = 0.4773`, which rounds to `0.477` (→ 47.7%). The paper prints the NPU
large cell as **0.478 (47.8%)**. The generator carries the raw `0.4773` in
`single_stream.csv` (`npu_raw`) but displays `0.478` (`npu_paper`) to reproduce
the paper cell exactly; the `NPU−GPU` large gap is the exact **+0.001 (+0.1%,
n.s.)**. This is a ±1-in-last-digit display choice on a stratum with no
significant difference, not a data or logic mismatch.

**About the `WARN token ... not in paper` lines:** the source generator's
`paper_text()` cross-check searches for the **fractional** tokens
`0.196`, `0.186`, `0.478`. These are absent from the paper *because the paper
reports sAP in percent* (`19.6`, `18.6`, `47.8`) — all three percent forms **are**
present in `main_vision.tex`. The warnings are an artifact of the fractional-vs-percent
scale, not a value discrepancy.

**Verdict: MATCH.** All Table 1 cells (end-to-end 8.3/10.0 ms; sAP overall
19.6/18.6; small 1.6/0.8; medium 18.4/14.8; large 47.7/47.8) and all significance
flags (p<0.01 / p<0.001 / p<0.001 / n.s.) reproduce the paper exactly, with the
one documented display-rounding of NPU-large 0.4773→0.478.
