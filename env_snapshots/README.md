# Environment states & reproducibility risk (A2 investigation, 2026-07-04)

Rebuttal/camera-ready re-experiment prep. The A2 co-tenancy work revealed that
the **venv drifted uncommitted** between the paper-era measurements and now,
which changed absolute deadline-miss numbers and silently broke ORT-CUDA. This
file documents the two known environment states and the reconstruction path.

## Timeline of paper-era measurements
| result | file | date |
|---|---|---|
| tab:main L2/L3 rows, heavy-bg | `rev20_5strat_heavybg.csv` | 2026-06-09 |
| cpuload (L2LM gpu_util 49.5%, NPU DM 79%) | `cpuload_raw.csv` | 2026-06-18 |
| tab:main L1CNN, Fig sweeps (ResNet k-sweep) | `rev30_clean_resnet/rev30_raw.csv` | 2026-06-28 |

## State A — paper-era (June 2026, partially reconstructed)
Evidence (logs `compiler_check_summary.md`, `exp_ft_local_summary.md`, dated
2026-06-12/13): **torch 2.8.0+cu128 (CUDA 12.8)**. onnxruntime-gpu 1.20.1
(CUDA-12 build) — worked because the CUDA-12.8 cuDNN matched. cuDNN cu12 ~9.10
(uv cache has `nvidia_cudnn_cu12-9.10.2.21`). Driver 580.159.03 (assumed same).
- NOT fully pinned: no committed lockfile from this period. `requirements-accv.txt`
  in git (last touched 2026-05-27) already lists CUDA-13 pins, so it does NOT
  reflect the measurement-time venv — **this is the reproducibility landmine**.

## State B — current (2026-07-04, this repo)
`env-current-2026-07-04.lock` (full pip freeze). Key:
```
torch==2.12.0 (+cu130, CUDA 13.0)   onnxruntime-gpu==1.20.1 (CUDA-12 build)
nvidia-cudnn-cu13==9.12.0.46         driver 580.159.03
system CUDA: /usr/local/cuda-12.0 only (CUDA 13 via pip nvidia-cuda-runtime==13.0.96)
```
Plus the ORT-CUDA shim `vendor_ort_cu12/` (nvidia-cudnn-cu12 9.12.0.46,
nvidia-cublas-cu12 12.6.4.1, nvidia-cuda-runtime-cu12 12.6.77) via
`analysis/ort_env.sh` — makes ORT co-tenants run on GPU again.

## What broke / drifted (State A -> B)
1. **ORT-CUDA silently disabled**: torch upgrade to cu130 replaced the CUDA-12.8
   cuDNN with cu13; ORT 1.20.1 (CUDA-12) then loaded skewed system cuDNN and fell
   back to CPU. Fixed in State B via the shim (see `vendor_ort_cu12/README.md`).
2. **Absolute anchor drift NOT explained by ORT alone** (measured 2026-07-04,
   State B + shim): ResNet50 k=1 All-GPU GPU DM = **59%** (paper 24%);
   L2LM All-NPU NPU DM = **25%** (paper 76%); L2LM co-tenant gpu_util = **73.5%**
   (paper 49.5%). synth CPU control stable (~32% vs ~27%). The GPU-path drift at
   matched gpu_util (~70%) suggests **torch 2.12/CUDA-13 GPU-scheduling** or
   cuDNN-version differences, not just ORT. Reconstructing State A likely needs
   the torch-2.8/cu128 FOREGROUND too, not only the ORT co-tenant.

## Reconstruction path (State A) and difficulty
Path: fresh venv (python 3.10) with
```
torch==2.8.0  torchvision (cu128 index: --index-url https://download.pytorch.org/whl/cu128)
onnxruntime-gpu==1.20.1
nvidia-cudnn-cu12==9.10.2.21  nvidia-cublas-cu12 (cu128-matched)  nvidia-cuda-runtime-cu12
mblt_model_zoo==1.5.1  (NPU; requires torch>=2.4.1 -> OK with 2.8)
ultralytics==8.4.56  numpy==2.2.6  opencv-python-headless
```
- **Difficulty: MEDIUM.** cu128 wheels are on the PyTorch index; NPU SDK is
  torch-2.8-compatible. Container (CUDA 12.8 base image) is cleaner than a venv
  for isolating the CUDA toolkit, but the NPU driver (`qbruntime`) must be mapped
  in — verify the Mobilint runtime works inside the container first.
- **Residual risk:** exact ORT/cuDNN patch versions at measurement time are not
  pinned (only torch 2.8+cu128 is confirmed from logs), so a reconstruction is a
  best-effort of State A, not a bit-exact restore. The GPU-scheduling drift may or
  may not fully revert.

## Recommendation for camera-ready
Freeze State B (this lockfile + shim) as the reference going forward, OR
reconstruct State A in a container and re-run the specific tables/figures whose
numbers matter, but **do not mix State A (paper) and State B (new) numbers in one
table**. Commit a lockfile from whichever state is used for the final numbers.
