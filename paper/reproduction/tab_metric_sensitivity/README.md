# Reproduction: Table `tab:metric-sensitivity`

Paper label: **`tab:metric-sensitivity`**
Paper location: **Section 4.3** (Evaluation).

Caption (paraphrased): Metric sensitivity at a high-contention ResNet50 sweep point.
Central-tendency metrics (mean, median) mask the degraded camera, while the
lower tail (p10) and the worst-case per-stream value expose it. Reading only
mean/median "hides" or only "partially" reveals the failed stream; the worst /
lower-tail statistics "expose" it.

## Input

`data/rev28_metric_sensitivity.csv` — per-stream sAP summarized at N=4 streams for a
single high-contention operating point (ResNet50 co-tenant), for both placement
policies (All-GPU and All-NPU). One row per statistic:

| column | meaning |
|---|---|
| `metric` | which summary statistic over the 4 per-stream sAP values (`mean`, `median`, `p10`, `worst`) |
| `allgpu_value` | that statistic under the All-GPU placement (sAP as a fraction) |
| `allnpu_value` | that statistic under the All-NPU placement (sAP as a fraction) |
| `exposes_failed_camera` | qualitative verdict: `hides` / `partial` / `exposes` |

## Run

```
/home/msyu/PycharmProjects/multimodel-scheduling-video/.venv/bin/python generate.py
```

## Outputs

- `metric_sensitivity.tex` — the LaTeX `tabular` for `tab:metric-sensitivity`.
- `metric_sensitivity.csv` — the numbers echoed back (copy of the input rows).

## Transform

The generator reads the four summary rows verbatim. Each row's `mean`, `median`,
`p10`, and `worst` is the corresponding statistic over the four per-stream sAP
values (p10 = 10th percentile via linear interpolation; `worst` = minimum over
the 4 streams). Values are stored as sAP fractions; the paper table displays
them **×100** (sAP × 100). No other computation is applied.

## Verification vs paper `tab:metric-sensitivity`

Generator output is in sAP fractions; multiply by 100 to compare with the paper.

| statistic | All-GPU (gen → ×100) | All-NPU (gen → ×100) | paper (GPU/NPU) | exposes | result |
|---|---|---|---|---|---|
| mean    | 0.107 → 10.7  | 0.1263 → 12.6 | 10.7 / 12.6 (hides)   | hides   | MATCH |
| median  | 0.0866 → 8.7  | 0.1152 → 11.5 | 8.7 / 11.5 (partial)  | partial | MATCH |
| p10     | 0.072 → 7.2   | 0.0912 → 9.1  | 7.2 / 9.1 (exposes)   | exposes | MATCH |
| worst   | 0.066 → 6.6   | 0.0836 → 8.4  | 6.6 / 8.4 (exposes)   | exposes | MATCH |

**Verdict: MATCH** — all mean/median/p10/worst values and the exposes-degraded-camera
verdicts reproduce the paper table.
