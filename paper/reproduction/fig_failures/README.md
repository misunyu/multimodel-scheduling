# fig:failures — Qualitative INT8-NPU quantization failures

Reproduction item for the paper figure labelled **`fig:failures`**.

## What the figure shows

A panel of Argoverse-HD camera frames overlaid with two sets of detections from a
COCO-pretrained **YOLOv11s** model run on two devices:

- **GREEN boxes** — detections from the **FP32 GPU** (the reference).
- **RED boxes** — safety-critical small objects that the **FP32 GPU detected but the
  INT8 NPU missed** (labelled `... (NPU miss)`).

The missed objects are predominantly **small / distant safety-critical classes**
(pedestrians, traffic lights, traffic/stop signs), illustrating that INT8
quantization on the NPU concentrates its accuracy loss on exactly those classes.

## Where it appears in the paper

- **Section 4.1** (evaluation), referenced near **Table 1**.
- LaTeX: `\includegraphics[width=0.72\linewidth]{vis_failure_examples.pdf}`,
  `\label{fig:failures}` in `paper/main_vision.tex`.
- Caption gist: small-object sAP drops sharply under INT8 (pedestrians, traffic
  lights, and signs missed by the INT8 NPU but detected by the FP32 GPU), while
  large-object accuracy is essentially unchanged.

## Files in this folder

| File | Description |
|---|---|
| `vis_failure_examples.pdf` | The exact rendered figure used by the paper (byte-for-byte copy — see note below). |
| `generate.py` | **Verbatim** copy of the original generator `paper/results_data/fig1_failures/generate_fig1.py` (logic unchanged; a header block was prepended). |
| `data/selected_frames.csv` | The chosen Argoverse-HD frames per panel (see below). |

## Input data — `data/selected_frames.csv`

The frame-selection record (the panels chosen from Argoverse-HD, **sids 2, 13, 17, 22**):

| panel | sid | frame | NPU-missed classes |
|---|---|---|---|
| 1 | 2  | 5   | person, person, traffic_light ×3 |
| 2 | 13 | 41  | traffic_light ×3 |
| 3 | 17 | 163 | person ×3 |
| 4 | 22 | 1   | person |

## How the figure is produced

`generate.py` (= `generate_fig1.py`):

1. Loads **YOLOv11s** on the **GPU in FP32** (`ultralytics` YOLO, `device="cuda"`).
2. Loads the same model on the **Mobilint NPU in INT8** (`global8` mode, pinned mxq
   from `results/rev7_pin.json`).
3. Scans the first frames of several Argoverse-HD sequences, matches GPU vs NPU
   detections by class + IoU, and selects frames where the NPU drops a
   safety-critical small object the GPU keeps.
4. Overlays the boxes (green = GPU FP32, red = object missed by INT8 NPU) and
   writes the multi-panel PDF to `paper/figures/vis_failure_examples.pdf`
   (the generator's hard-coded `OUT` path).

## ⚠️ Reproducibility status — BLOCKED (hardware-dependent)

**재현 스크립트 있음 (a reproduction script is provided) — but it requires GPU + NPU
hardware and CANNOT be regenerated in a CPU-only reproduction environment.** The
exact rendered PDF is therefore provided here as a **byte-for-byte copy** of the
figure the paper uses (`paper/figures/vis_failure_examples.pdf`).

The generator performs **live device inference**: it loads a CUDA YOLO model *and*
the Mobilint NPU, and reads the local Argoverse-HD dataset. It also imports
repo-internal modules (`_step_d_common`, `step0_compare_devices`) via `sys.path`
manipulation relative to its **original location**. It cannot run standalone from
this folder and must not be attempted in a CPU-only environment.

### Command (NOT runnable here — requires GPU + NPU hardware)

```bash
# Must be run from the ORIGINAL location, on the original platform
# (CUDA GPU + Mobilint NPU present, Argoverse-HD dataset available):
cd paper/results_data/fig1_failures
python generate_fig1.py        # writes paper/figures/vis_failure_examples.pdf
```

### Hardware / software requirement (from the paper setup and the scripts)

| Component | Requirement | Source |
|---|---|---|
| GPU | **NVIDIA RTX 5090**, CUDA — runs YOLOv11s in **FP32** (`device="cuda"`) | paper §4.1 "Platform and runtimes"; `generate_fig1.py` L61-68 |
| NPU | **Mobilint MLA100**, INT8, `infer_mode="global8"`, legacy pinned mxq binary | paper §4.1; `generate_fig1.py` L106-108 (`preload_npu_instances(1, infer_mode="global8")`) |
| CPU/host | Intel Core Ultra 9 285K (24 cores) class host | paper §4.1 |
| Model / input | COCO-pretrained **YOLOv11s**, input **640×640**, `conf=0.25`, `iou=0.45` | `step0_compare_devices.py` (`IMG_SIZE=640, CONF=0.25, IOU=0.45`) |
| Software | `ultralytics` (YOLO predict), `opencv-python` (`cv2`), `matplotlib`, `numpy`, the **Mobilint NPU SDK**, and repo modules `_step_d_common` / `step0_compare_devices` | `generate_fig1.py` imports |
| Dataset | local Argoverse-HD val images + `val.json` | `generate_fig1.py` (`load_val`, `load_split_for_sid`) |

Without **both** accelerators (RTX 5090 GPU **and** Mobilint MLA100 NPU) the
generator cannot run and the figure cannot be reproduced — hence this item ships
the exact rendered PDF instead.

## Note on the provided PDF

The `vis_failure_examples.pdf` here is copied from `paper/figures/vis_failure_examples.pdf`,
which is (a) the file `main_vision.tex` renders for `fig:failures`, and (b) exactly
the path the generator's `OUT` writes to. It is confirmed **byte-for-byte identical**
to `paper/figures/vis_failure_examples.pdf` (`cmp` → no differences; matching md5).

### ⚠️ Stale copy in `results_data` (do not use it)

An **older, stale render** exists at
`paper/results_data/fig1_failures/vis_failure_examples.pdf`
(**1.5 MB**, md5 `104eb74e…`, earlier dimensions). It does **not** match the
figure the paper actually renders — the canonical figure is the **9.2 MB**
`paper/figures/vis_failure_examples.pdf` (md5 `447a3722…`), which is exactly what
the generator's `OUT` path writes and what this folder copies. Anyone rebuilding
the figure should overwrite `paper/figures/…`, not trust the `results_data` copy.

**Why the stale file was NOT renamed to `*_STALE.pdf`:** although this task
permitted a one-off rename in `results_data`, the stale file is **referenced by
several scripts and pages** and renaming it would break them, so it was left in
place (reported instead):

- `paper/results_data/regen_all_stale_backup.py` (L24 reads it)
- `paper/results_data/generate_index_html.py` (L80/L115/L117 link to it)
- `paper/results_data/index.html` (L26/L32/L34 link to it)
- `paper/results_data/fig1_failures/README.md` (L3-4 describe it)
