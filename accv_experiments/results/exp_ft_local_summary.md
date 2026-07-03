# EXP-FT-LOCAL — Summary (W3 defence, self-contained paired local-compile design)

**Question.** With absolute accuracy raised by fine-tuning, does the small-biased INT8
quantization structure (small loss >> large, large negligible) survive — under the **same local
compile recipe** for COCO and FT, so "fine-tuning effect" and "recipe effect" are separated?

**Verdict: Outcome A (supports the claim, within this export recipe).** Gate G passes (COCO-local is
small-biased). The fine-tuned model's INT8−FP32 per-size gap keeps the same direction: small
relative loss −34.9% (p<1e-7) far exceeds large −4.3% (single-digit, **n.s.** p=0.079). The
small-biased structure is therefore not an artifact of the low-accuracy COCO-pretrained regime.

> Scope: these are local-recipe numbers (qbcompiler 1.1.2, train-calib), a DIFFERENT quantization
> from the vendor blob; compiler-check showed they are NOT equivalent to vendor (FAIL). They are
> **not comparable to main Table 1** and are for the internal COCO-vs-FT paired comparison only.

---

## 1. Environment + gates
- Prereq gate: EXP-AUDIT-RUNTIME artifact present; `.venv_qbc` (torch 2.8.0+cu128 cxx11, qbcompiler
  1.1.2 MMC) verified; disk OK. `yolo11s.pt` sha 85a76fe86dd8afe3.
- FP32 gate (Step 5, offline mAP, CONF/IOU = project harness, val 15,062 imgs): see §3 — PASS.

## 2. Data + training
- **Download (detached, session-independent)**: Argoverse 1.1 tracking train tars (`argoverse`
  S3 bucket; old `argoai-argoverse` is dead; compact 29 GB Argoverse-HD-Full.zip is AccessDenied).
  Extracted ring_front_center only. **Integrity: 39,384/39,384 images present == train.json** (PASS).
- train.json labels are Li et al. **pseudo-labels** (recorded). val never used for train/calibration.
- **Fine-tuning**: yolo11s.pt → 50 epochs, ultralytics 8.4.56 defaults (auto-opt MuSGD), imgsz 640,
  seed 0, deterministic. Labels mapped AHD→COCO ([0,1,2,3,5,7,9,11]) to keep the 80-class head so the
  rev19 eval harness (coco_mapping) is reused unchanged. Dataset: `accv_experiments/data/ahd_yolo/`
  (train 39,384 / val 15,062), yaml `ahd_coco80.yaml`.
- **best checkpoint = epoch 1** (Ultralytics default val-mAP criterion; mAP50-95 peaked 0.236 at ep1,
  declined to 0.195 by ep50 — pseudo-label overfit). Used as-is (no cherry-pick). best.pt sha 4c4b342a6d09de67.
  Curves: `accv_experiments/results/ft_runs/ft_yolo11s/results.csv`.

## 3. FP32 validation gate (Step 5) — PASS
Offline per-size mAP on val (15,062 imgs), COCO-pretrained vs FT (both FP32 GPU PyTorch):

| size | COCO | FT | Δ(FT−COCO) |
|---|---|---|---|
| all | 0.1844 | 0.2131 | +0.0287 |
| small | 0.0125 | 0.0576 | **+0.0451** |
| medium | 0.1734 | 0.2421 | **+0.0688** |
| large | 0.5029 | 0.4419 | −0.0611 |

FT improves small/medium by ≥+0.01 (small ~4.6×) → not Outcome C → PROCEED. (Large decreased; flagged.)

## 4. Compile + calibration (Step 6)
- Recipe (same for both): qbcompiler 1.1.2, onnx 640/opset13, preset `yolo_640`, scheme **global8**.
- **Calibration: 200 train-split frames, uniform, single shared list for BOTH compiles**
  (`qbc_calib200_train.list`). compiler-check's val-calib .mxq NOT reused.
- Outputs: COCO-local `qbc_coco_traincalib_global8.mxq` (sha f987aca5), FT-local
  `qbc_ft_traincalib_global8.mxq` (sha ea091a9d). Log: `ft_dual_compile.log`.
- **Bit-identical (Step 7)**: both mxq, 20 frames ×2 → max box/score diff = 0 (deterministic).

## 5. Measurement (Step 8) — Table-1 protocol, isolated single-stream, threads=4, skip≈0, 3 reps
288 runs, **0 invalid_skip** (all skip 0.0%). Per-size sAP (mean over 24 logs × 3 reps):

| cell | all | small | medium | large |
|---|---|---|---|---|
| FP32-COCO (GPU) | 0.1957 | 0.0159 | 0.1839 | 0.4768 |
| INT8-COCO-local (NPU) | 0.1874 | 0.0090 | 0.1535 | 0.4796 |
| FP32-FT (GPU) | 0.2185 | 0.0485 | 0.2373 | 0.4559 |
| INT8-FT-local (NPU) | 0.1920 | 0.0316 | 0.1976 | 0.4362 |

## 6. CORE: INT8−FP32 per-size quantization gap (paired, per-log Wilcoxon n=24)
| pair | small | medium | large |
|---|---|---|---|
| COCO-local | −0.0068 (−43.1%, p=6.8e-5 \*\*\*) | −0.0304 (−16.5%, p=1.3e-5 \*\*\*) | +0.0028 (+0.6%, n.s. p=0.33) |
| FT-local | −0.0169 (−34.9%, p=1.2e-7 \*\*\*) | −0.0397 (−16.7%, p=6.0e-7 \*\*\*) | −0.0196 (−4.3%, **n.s. p=0.079**) |

## 7. Gate G + Outcome (§0)
- **Gate G (COCO-local small-biased)**: PASS — small −43.1% (sig) ≫ large +0.6% (n.s.). The recipe
  preserves the small-biased structure under train-calibration (consistent with compiler-check's
  val-calib −36.6%/−1.5%).
- **Outcome A**: FT-local keeps the direction — small relative loss −34.9% (highly sig) ≫ large
  −4.3% (single-digit, statistically **n.s.**), medium in between. Small-biased structure survives
  fine-tuning that raised small-object FP32 accuracy ~3× (0.016→0.049). So the structure is **not a
  low-accuracy-regime artifact** within this export configuration.
- **Honest nuance (not hidden)**: in ABSOLUTE terms FT-local's large gap grew (+0.0028 COCO →
  −0.0196 FT) and small/large absolute losses are now comparable (−0.0169 vs −0.0196); but per the
  pre-registered §0 criterion (relative loss; large single-digit % / negligible-or-n.s.), this is
  Outcome A. The large penalty remains single-digit % and statistically n.s. (p=0.079, borderline).
  Generality of the recipe is NOT claimed (single fixed recipe/version).

## 8. Relation to vendor / Table 1
These INT8 numbers come from the local qbcompiler recipe (train-calib), which compiler-check proved
is NOT equivalent to the vendor blob (per-size Δ up to 0.008 > 0.003, FAIL). They are therefore an
**internal COCO-vs-FT paired comparison only** and are not placed on the same axis as main Table 1.
This result pertains to a supplementary / Limitations note about whether the small-biased structure
is regime-dependent; paper integration is performed in a separate channel (paper .tex untouched).

## 9. Stretch (Step 10) — reversal direction reproduces on FT (internal only)
N=4 (logs [2,22,3,21]) + L3_VLM, 3 reps, worst-stream sAP. FT GPU (best.pt) vs FT-local NPU
(single-mode mxq `qbc_ft_traincalib_single.mxq` sha c0520594, recompiled for 4 concurrent NPU):

| placement | worst-stream | mean | skip% |
|---|---|---|---|
| All-GPU (FT) | 0.026 ± 0.004 | 0.082 | 97.9 |
| All-NPU (FT-local) | 0.108 ± 0.000 | 0.188 | 1.1 |

Under the GPU-saturating VLM, All-NPU worst-stream beats All-GPU by ~4.2× — the **same reversal
direction** as YOLOv11s in the main paper, now on the fine-tuned model. Direction-only, internal
comparison (single-mode FT-local recipe differs from the global8 cells in §5-6; not Table-comparable).

## 10. Forbiddens compliance
No fabrication; 0 invalid runs in aggregates. Recipe fixed once (no hp/calib/checkpoint/compile
re-tuning after seeing results). val never used for train/calib. Local INT8 not juxtaposed with
published Table 1 numbers on a shared axis. Published data unmodified. Gate G not loosened. Paper
.tex untouched.

## Artifacts
- `exp_ft_local_results.csv` (288 rows), `ft_gate_offline_map.json`, `ft_dual_compile.log`,
  `ft_measure_stdout.log`, `qbc_calib200_train.list`.
- Stretch: `ft_stretch_stdout.log`, `ft_single_compile.log`.
- Models: `ft_runs/ft_yolo11s/weights/best.pt` (FT FP32), `qbc_{coco,ft}_traincalib_global8.mxq`,
  `qbc_ft_traincalib_single.mxq` (Stretch N=4 NPU).
- Scripts: ft_prep_dataset.py, ft_train_driver.sh, ft_gate_offline_map.py, ft_dual_compile.py,
  ft_measure_cells.py, ft_stretch_vlm.py, dl_extract_argoverse_train.sh.
