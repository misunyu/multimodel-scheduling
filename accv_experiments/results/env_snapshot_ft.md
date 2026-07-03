# EXP-FT v2 — Prerequisite check (STOP before training)

Recorded after EXP-NORM v2. GPU 0 compute procs, no MPS daemon, disk 1.1T free. No env drift
(driver 580.159.03/CUDA 13.0, torch 2.12.0+cu130, ort 1.20.1, ultralytics 8.4.56).

EXP-FT was NOT started. Two pre-registered stop-and-report conditions are hit:

## Blocker 1 — Argoverse-HD train IMAGES absent (Step 1)
- `Argoverse-HD/annotations/train.json` IS present (39,384 images, 771,774 anns, 65 sequences),
  but it is the Li et al. pseudo-label train set (date 2021-03-01; image keys id/sid/fid/name).
- The actual train IMAGES are NOT on disk: `Argoverse-1.1/argoverse-tracking/` contains only `val/`
  (24 sequences); no `train/` dir, no `ring_front_center` train frames anywhere on the system.
- Fine-tuning (Step 2) needs the Argoverse 1.1 tracking train images (~39k front-center frames,
  tens of GB) which must be downloaded. → Step 1 prerequisite not satisfiable locally.

## Blocker 2 — Mobilint INT8 toolchain: broken in this env + provenance mismatch (Step 3)
- A quantization compiler is installed: `qbcompiler 1.1.2` (+ wheel in ~/PycharmProjects/MobilintTest).
- BUT it **cannot run in the measurement venv**: importing fails to init the native MMC module —
  `qbcompiler/mmc.cpython-310-...so: undefined symbol: _ZN3c104impl12PyObjectSlotD1Ev`
  i.e. an ABI mismatch between qbcompiler's compiled extension and torch 2.12.0+cu130. No INT8
  compilation is possible here without rebuilding the toolchain against this torch.
- Provenance mismatch: the paper's NPU model (`yolo11s.mxq`, sha b2441f9d, HF rev 8e62b1a9) is a
  **vendor-provided HuggingFace blob**, not locally compiled. So qbcompiler 1.1.2 is a DIFFERENT
  export path than the one that produced the main-experiment INT8 model. Step 3 requires "same
  Mobilint toolchain version as main; if different, stop and report" — it is different (and
  unverifiable, since the vendor compiler version for b2441f9d is unknown). This is also the export
  path that previously corrupted YOLOv10s (STATUS H2).

## Decision required (human) — see chat report
