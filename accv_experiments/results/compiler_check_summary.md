# EXP-FT-COMPILER-CHECK — Summary

**Goal.** Verify whether the local Mobilint compiler reproduces the vendor INT8 blob
(`yolo11s.mxq`, sha b2441f9d) used in the paper's Table 1, on the original COCO-pretrained
YOLOv11s. PASS opens the EXP-FT self-compile path; FAIL means self-compile is confounded.

**Verdict: FAIL.** The local-compiled INT8 reproduces the small-object bias direction and matches
on AP_small, but per-size sAP exceeds the Δ ≤ 0.003 tolerance on medium (Δ 0.0042) and large
(Δ 0.0082). The self-compiled .mxq is a *different* quantization than the vendor blob, so it cannot
be used for EXP-FT without confounding the fine-tuning effect with a quantization-recipe change.

---

## 1. Compiler inventory + version
- `qbcompiler 1.1.2` (+aries2), wheel `qbcompiler-1.1.2+aries2-py3-none-any.whl`
  (~/PycharmProjects/MobilintTest). Same package observed in the prior session.
- The vendor compiler version used to build b2441f9d is **unknown** (HF snapshot
  `models--mobilint--YOLO11s` rev 8e62b1a9 contains only `aries/yolo11s.mxq`, no recipe/config).
  Version match is therefore unverifiable a priori; this gate tests equivalence empirically instead.

## 2. ABI result (was the prior-session blocker)
- In the inference venv (torch 2.12.0+cu130) `qbcompiler.mmc` fails:
  `undefined symbol _ZN3c104impl12PyObjectSlotD1Ev`. Root cause: mmc.so needs the **CXX11 ABI**
  (116 cxx11-tagged undefined symbols) and torch ≤2.4/≥... ABI specifics; torch 2.12 (cxx11 ABI on,
  but symbol removed) and torch 1.13/2.4 (cxx11 ABI off) both mismatch.
- **Resolved in a separate compile venv** `.venv_qbc`: torch **2.8.0+cu128** (cxx11 ABI=1, provides
  `PyObjectSlotD1Ev`, supports Blackwell sm_120) + onnxruntime-gpu 1.20.1 + qbcompiler 1.1.2.
  `qbcompiler.mmc` imports OK; `Compiler.compile/mblt_compile` available; CUDA matmul runs on the
  RTX 5090. So the compiler **can** run (separate from the inference venv) — the prior "toolchain
  unusable" blocker is lifted at the ABI level. Equivalence is the remaining gate, and it fails.

## 3. Reproduction-compile configuration (documented assumptions; vendor recipe unknown)
- Input: `yolo11s.pt` (sha 85a76f…, COCO-pretrained) → ONNX via Ultralytics 8.4.56,
  imgsz 640, **opset 13**, static, fp32. (vendor opset unknown → reasonable default.)
- Preset: `yolo_640` (uint8 input, letterbox 640×640 padValue 114, imageChannels 3, extends
  "detection"). inference_scheme = **global8** (the mode the vendor blob is used in for Table 1).
- Calibration: **200 Argoverse-HD val frames, uniform sample** (`qbc_calib200/`). The vendor's
  calibration set/size is unknown → reasonable default, recorded.
- device=gpu, backend=onnx. Output: `qbc_local_yolo11s_global8.mxq` (10,680,985 B, sha 27086a6a…).
  (Note: vendor legacy b2441f9d is 11.87 MB; the mblt-zoo global8 build is 10,674,203 B — the local
  global8 output matches the zoo global8 size class, not the older legacy blob.)

## 4. Equivalence (Table 1 protocol: NPU isolated, 24 logs, threads=4, skip≈0, 3 reps)
Operating-point note: `p1r6_baseline_yolo11s.csv` NPU values are a 24-thread misconfig (skip ~55%,
staleness-contaminated large) and are NOT the Table 1 source. Both models here were measured at
threads=4 (skip 0.0%), reproducing the clean Table 1 operating point.

| | all | small | medium | large | skip% | infer ms |
|---|---|---|---|---|---|---|
| vendor b2441f9d | 0.1859 | 0.0082 | 0.1477 | 0.4773 | 0.0 | 10.1 |
| local qbc | 0.1847 | 0.0101 | 0.1522 | 0.4698 | 0.0 | 10.0 |
| Table 1 (published NPU) | — | 0.009 | 0.148 | 0.478 | ~0 | 10.0 |
| GPU FP32 (reused, not re-measured) | 0.1957 | 0.0159 | 0.1839 | 0.4768 | ~0 | 8.5 |

Vendor re-measure ≈ published Table 1 (Δ ≤ 0.001 every size) → harness/operating point validated.

**Per-size equivalence (local vs published Table 1):**
| size | local | Table 1 | \|Δ\| | ≤ 0.003 |
|---|---|---|---|---|
| small | 0.0101 | 0.009 | 0.0011 | ✅ |
| medium | 0.1522 | 0.148 | 0.0042 | ❌ |
| large | 0.4698 | 0.478 | 0.0082 | ❌ |

(local vs vendor head-to-head: small 0.0018, medium 0.0045, large 0.0075 — same conclusion.)

**Quantization-gap direction (NPU − GPU FP32), both small-biased:**
- local: small −0.0058 (−36.6%), medium −0.0317, large −0.0070 (−1.5%) → small-biased ✓
- vendor: small −0.0077 (−48.1%), medium −0.0362, large +0.0005 (+0.1%) → small-biased ✓

## 5. Verdict (§0)
- PASS requires per-size Δ ≤ 0.003 at **every** size AND small-biased direction.
- Direction: PASS (both small-biased). Per-size: **FAIL** — medium (0.0042) and large (0.0082)
  exceed 0.003. The local compile is slightly less small-penalizing (−36.6% vs −48.1% on small) and
  slightly worse on large (−1.5% vs +0.1%), consistent with a different calibration/bit recipe.
- → **FAIL.** Self-compile is confounded relative to the vendor blob. **EXP-FT INT8 must use the
  vendor export path or be held.** The result is not reinterpreted as "approximately equivalent":
  the gap is small but real and exceeds the pre-registered tolerance.

## 6. What this changes vs the earlier EXP-FT prerequisite report
- Earlier Blocker 2 ("toolchain unusable") is partially overturned: the compiler **runs** in an
  ABI-matched venv. But it does **not** reproduce the vendor INT8 to spec, so the self-compile route
  for EXP-FT is still closed. (Blocker 1 — absent Argoverse-HD train images — is unchanged.)

## Artifacts
- `compiler_check_results.csv` — 144 rows (2 mxq × 24 logs × 3 reps), per-size sAP + skip.
- `qbc_local_yolo11s_global8.mxq` (local INT8), `yolo11s.onnx` (opset13), `qbc_calib200/` (calib set).
- `qbc_compile_stdout.log`, `qbc_equiv_stdout.log`.
- Compile venv: `.venv_qbc` (torch 2.8.0+cu128 cxx11, qbcompiler 1.1.2). Scripts:
  `accv_experiments/scripts/qbc_compile_check.py`, `qbc_equiv_eval.py`.
