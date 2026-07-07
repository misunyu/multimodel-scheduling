# Reproduction: `fig:quant_conf` — per-size confidence shift under INT8 quantization

Reproduces **Figure `fig:quant_conf`** of `paper/main_vision.tex`:
`\includegraphics[width=\linewidth]{b1_quant_conf_shift.pdf}`.

**Paper caption:** "Per-size confidence distributions for matched FP32 and INT8
detections. INT8 shifts scores downward, especially for small and medium
objects. Detections pushed below threshold (51% for small objects) are excluded,
so the plotted shift understates the small-object effect."

**Paper location:** Section 4 (*Deployment-Faithful Evaluation of Placement
Decisions*), Sec. 4.2 discussion — paragraph "Quantization and staleness reduce
accuracy through different size-dependent mechanisms" (`main_vision.tex` line
~257). The figure supports the claim that quantization loss is concentrated in
small objects via a score/recall path, not a localization path.

## Two-stage pipeline (dataset-decoupled)

The original single-file generator read the Argoverse-HD `val.json` directly, so
the figure could not be reproduced without the dataset. It is now split into two
stages so **the figure is reproducible with no dataset**:

| stage | script | needs `val.json`? | reads | writes |
|---|---|---|---|---|
| 1 · extract | `extract_data.py` | **yes** | `val.json` + `data/b1_dets/*.npz` | `data/quant_conf_matched_scores.csv` (+ diagnostics) |
| 2 · figure  | `generate.py`     | **no**  | `data/quant_conf_matched_scores.csv` | `b1_quant_conf_shift.pdf` |

- **To reproduce the figure only, run `generate.py`.** It reads the bundled
  intermediate `data/quant_conf_matched_scores.csv` and needs no dataset.
- **`extract_data.py` only needs to be re-run to rebuild that intermediate** from
  the raw dumps + `val.json` (e.g. if you change the matching logic or the dumps).

The data-processing / plotting logic was copied **verbatim** from the original
`analysis/b1_quant_decompose.py`; the only change is the I/O split (extract writes
the per-bin matched scores; the figure loads them instead of recomputing). Every
adjusted line is marked `# [reproduction]`.

### The bundled intermediate — what it contains (and why it is license-safe)
`data/quant_conf_matched_scores.csv` has three columns: `size, fp32_score,
int8_score`, one row per GT object matched by **both** FP32 and INT8 (the exact
arrays the figure histograms). These are **model-output confidence scores only**
(`%.10g`, so the doubles the figure consumes round-trip exactly). It contains
**no Argoverse-HD annotation content** — no bounding boxes, no image ids, no
category labels — so redistributing it does not redistribute the dataset.

## Inputs and their meaning

### Stage 1 (`extract_data.py`) — dataset-dependent
- **`accv_experiments/data/argoverse_hd/Argoverse-HD/annotations/val.json`**
  — Argoverse-HD validation ground-truth annotations (COCO format; `bbox` in
  xywh on native 1920x1200, `area` used for the small/medium/large size bins).
  **EXTERNAL DATASET DEPENDENCY — NOT bundled** (~39 MB; licensed dataset).
  `extract_data.py` references it at its original repo path
  `accv_experiments/data/argoverse_hd/Argoverse-HD/annotations/val.json`.
  - *To obtain it:* download **Argoverse-HD** from the streaming-perception
    project page — https://www.cs.cmu.edu/~mengtial/proj/streaming/ (the
    "Argoverse-HD" tarball; annotations also mirrored via the Argoverse site
    https://www.argoverse.org/). Unpack so that the validation annotation file
    lands at exactly the path above (`.../Argoverse-HD/annotations/val.json`).
    Only `val.json` is required by this script — the images are not needed.
  - *If missing at run time:* `extract_data.py` exits with a message pointing
    here; the figure can still be rebuilt via `generate.py` from the bundled
    intermediate.
- **`./data/b1_dets/gpu_fp32.npz`, `./data/b1_dets/npu_int8.npz`** (+ their
  `*_meta.json`) — pre-dumped per-detection arrays (`image_id`, `box` xyxy,
  `score`, `cls`) for the FP32 GPU path and the INT8 NPU path, run over the 24
  Argoverse-HD val logs at img_size 640, conf 0.25, iou 0.45 (see meta json).
  ~3.9 MB total; **bundled** in `./data/b1_dets/`.

### Stage 2 (`generate.py`) — dataset-free
- **`./data/quant_conf_matched_scores.csv`** — the bundled aggregated
  intermediate described above (the ONLY input the figure needs).

## Run
```
# Figure only (no dataset needed):
python generate.py

# Rebuild the intermediate from val.json + dumps (needs Argoverse-HD), then figure:
python extract_data.py && python generate.py
```
(Use the repo `.venv` interpreter, e.g.
`/home/msyu/PycharmProjects/multimodel-scheduling-video/.venv/bin/python`.)

## Outputs
- `b1_quant_conf_shift.pdf` — the paper figure (per-bin FP32 vs INT8 matched-score histograms). *(generate.py)*
- `data/quant_conf_matched_scores.csv` — bundled figure intermediate. *(extract_data.py)*
- `b1_quant_conf_shift.csv` — per-bin matched-score means, mean/median shift. *(extract_data.py)*
- `b1_quant_decomposition.csv` — per-bin x error-category counts and fractions. *(extract_data.py)*

## Transform
For each image and class, FP32 and INT8 detections are independently matched to
GT with COCO-style greedy assignment (score-desc, IoU >= 0.5, one det per GT).
Taking the FP32-matched GT objects as denominator, each is classified by what
INT8 did: score kept (i), score dropped (ii), missed / below-threshold (iii,
recall/score path), or localization-degraded (iv). The figure overlays the
per-size density histograms of matched detection scores for FP32 vs INT8 and
reports the mean score shift per size bin.

## Verification result — MATCH (and split is figure-identical)
Reproduced values match the paper text exactly:

| size   | matched-score shift | paper |
|--------|--------------------:|------:|
| small  | -0.088              | -0.088 |
| medium | -0.086              | -0.086 |
| large  | -0.043              | -0.043 |

Small-object "missed / below threshold" fraction (category iii) = **51.3%**,
matching the paper's "51% of small-object detections fall below threshold".

**Split equivalence check (before vs after the two-stage split):**
- `b1_quant_conf_shift.csv` and `b1_quant_decomposition.csv` from `extract_data.py`
  are **byte-for-byte identical** to the pre-split single-file outputs.
- Per-bin matched-pair counts and FP32/INT8/shift means from the bundled
  intermediate are identical to the reference to 4 decimals
  (n = 3907 / 37395 / 41498).
- The rendered `b1_quant_conf_shift.pdf` before vs after is **pixel-identical**
  (rasterized at 150 dpi, max pixel difference = 0).
