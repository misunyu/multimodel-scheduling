# Source before running ORT-CUDA co-tenant experiments (see vendor_ort_cu12/README.md).
# Points onnxruntime-gpu 1.20.1 (CUDA-12 build) at an isolated, self-consistent
# CUDA-12 cuDNN/cuBLAS/cudart set, so ORT co-tenants run on GPU (not CPU fallback).
# torch (CUDA-13) and the NPU stack are unaffected (different sonames).
_ORT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/vendor_ort_cu12/nvidia"
export LD_LIBRARY_PATH="${_ORT_ROOT}/cudnn/lib:${_ORT_ROOT}/cublas/lib:${_ORT_ROOT}/cuda_runtime/lib:${LD_LIBRARY_PATH}"
