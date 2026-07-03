# rev26 — detector-generality check: reversal REPRODUCES on YOLOv8n

_New measurement. Same protocol as YOLOv11s (rev19/22/25): both devices `torch.set_num_threads(4)`, same co-tenants, same N=4 Comp.A panel, same ResNet50 GPU-pressure sweep. Detector ONLY changed. 3 reps/point. Sanity pre/post: NPU infer ~9.1 ms, skip 0% (threads=4). core scripts / YOLOv11s artifacts / paper / results unchanged._

## STEP 0 — why YOLOv8n (not YOLO11n)
**YOLO11n has no INT8 export** — `mobilint/YOLO11n` returns HTTP 404 (`aries/global8/yolo11n.mxq` not found), matching the paper's existing statement. Hand-compiling with `qbcompiler` would NOT be the vendor pipeline that produced the YOLO11s mxq (vendor-published, recipe unknown), so it would be an invalid same-protocol comparison. Per user decision, we use the **vendor-published, cached YOLOv8n INT8 export** (`models--mobilint--YOLOv8n/.../aries/yolov8n.mxq`, loaded global8) — a genuine smaller-detector export from the same Mobilint pipeline as YOLO11s. This tests architecture/detector generality honestly.

## STEP 1 — Isolated single-stream per-size (Table 1 equiv)
| size | GPU | NPU | gap | rel | reproduces? |
|---|---|---|---|---|---|
| small | 0.0051 | 0.0032 | −0.0019 | **−36.3%** | ✓ GPU≫NPU (quantization small-biased) |
| medium | 0.0988 | 0.0760 | −0.0228 | −23.1% | ✓ |
| large | 0.3961 | 0.3640 | −0.0320 | **−8.1%** | ✓ mildest on large (GPU still slightly ahead) |

- **Quantization is steeply size-stratified, small-biased** (−36.3% small ≫ −8.1% large) — the YOLOv11s pattern reproduces.
- Isolated winner is the **GPU at every size** → isolated prescription = **All-GPU** (same as YOLOv11s).
- Honest difference from YOLOv11s: on large, YOLOv8n keeps a small GPU lead (−8.1%) rather than YOLOv11s's statistical tie (+0.1%). The *direction* (small worst, large mildest) holds; large is not a perfect tie. (NPU infer 8.8 ms, skip 0%.)

## STEP 2 — Contention worst/mean (Table 3 equiv), N=4
| co-tenant | All-GPU worst (skip) | All-NPU worst (skip) | winner |
|---|---|---|---|
| L1_light (CNN) | 0.0717 (14%) | 0.0604 (0%) | GPU |
| L2_lm (LM) | 0.0419 (70%) | 0.0323 (NPU skip **65%**) | GPU |
| **L3_vlm (VLM)** | **0.0092 (98%)** | **0.0604 (0%)** | **NPU (6.6×)** |

**The full conditional structure reproduces:**
- **VLM (GPU-saturating) → NPU wins, 6.6×** — the reversal. ✓✓
- **CNN (light) → GPU wins** (low contention). ✓
- **LM (CPU-touching) → GPU wins because the NPU also degrades** (NPU skip 65% — the LM contends for the host CPU running NPU post-processing). ✓ — the *same mechanism and the same LM-confound* as YOLOv11s.

## STEP 3 — Sweep crossover (Fig.2 equiv)
All-NPU flat **0.0604**; All-GPU declines monotonically with GPU skip:
`1%:0.0764 → 14%:0.0720 → 34%:0.0613 → 47%:0.0556 → 62%:0.0492`.
**Crossover ≈ 36% GPU skip** (vs YOLOv11s ~48%). All-GPU monotone ↓, All-NPU flat, they cross — at a **different threshold**, exactly as the paper predicts ("threshold is detector/platform-specific, the phenomenon transfers").

## STEP 4 — Summary table

| Detector | isolated winner (small) | isolated large | contention winner (VLM) | All-NPU worst / All-GPU worst | crossover GPU-skip |
|---|---|---|---|---|---|
| YOLOv11s | GPU (−48%) | tie (+0.1%, n.s.) | NPU | 0.083 / 0.016 (5.2×) | ~48% |
| **YOLOv8n** | **GPU (−36%)** | **GPU mild (−8.1%)** | **NPU** | **0.060 / 0.009 (6.6×)** | **~36%** |

## Verdict — **REVERSAL REPRODUCES** (isolated = GPU, VLM contention = NPU)
On YOLOv8n (a smaller, different-family vendor INT8 export), measured under the identical protocol:
1. isolated profiling prefers the **GPU** at every size (quantization small-biased, −36% small ≫ −8% large);
2. under the GPU-saturating VLM the device choice **reverses** — All-NPU beats All-GPU **6.6×** on worst-stream;
3. the reversal is **conditional on a GPU-saturating co-tenant** (VLM yes; light CNN no; CPU-touching LM no, via the same NPU-host confound);
4. the sweep crossover exists and sits at **~36% GPU skip — a different threshold from YOLOv11s's ~48%**, supporting "threshold is detector-specific, phenomenon transferable."

Honest caveats: YOLOv8n is smaller (absolute sAP lower; large gap −8.1% rather than a tie), but worst-stream values are not floored (0.06 range) so the directional comparison is sound. This is a positive, clean reproduction — not engineered.

## Files
- `results/rev26_yolov8n_single_stream.csv`, `rev26_yolov8n_contention.csv`, `rev26_yolov8n_sweep.csv`, `rev26_summary.md`
- `accv_experiments/scripts/phase_rev26_yolov8n.py` (detector-only change; core eval scripts unmodified)
- YOLOv8n vendor mxq: `models--mobilint--YOLOv8n/.../aries/yolov8n.mxq`, global8, same Mobilint pipeline as YOLOv11s.
