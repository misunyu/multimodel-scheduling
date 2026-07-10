# Multimodel Scheduling (Mobilint NPU + GPU)

Profiles multiple models across **CPU, GPU (CUDA) and the Mobilint Aries NPU**,
runs them concurrently under a schedule, and trains an XGBoost model that
recommends the best device placement (deployment) for a given workload.

This replaces the previous Neubla-NPU integration. Each model runs on all three
device types; the placement predictor is trained from real contention runs.

## Models

Every model runs on **both GPU and the Mobilint NPU** (CPU too):

| Logical name | Kind | NPU (Mobilint) | GPU / CPU |
|--------------|------|----------------|-----------|
| `yolo11n/s/m/l/x` | detection (vision) | `.mxq` (mblt_model_zoo) | ONNX Runtime (`models/onnx/*.onnx`) |
| `resnet50` | classification (vision) | `.mxq` | ONNX Runtime |
| `llama1b` (Llama-3.2-1B) | text-generation (LLM) | `mobilint/Llama-3.2-1B-Instruct` @W8 | `unsloth/Llama-3.2-1B-Instruct` (transformers) |
| `qwen2_vl` (Qwen2-VL-2B) | image-text-to-text (VLM) | `mobilint/Qwen2-VL-2B-Instruct` | `Qwen/Qwen2-VL-2B-Instruct` (transformers) |

The model registry (`model_registry.py`) is the single source of truth. Vision
models report **FPS**; LLM/VLM report **tokens/sec** (a separate XGBoost target).
LLM/VLM are profiling + placement only — they run headless (not rendered in the
GUI) but participate in schedule generation, contention runs and prediction.

## Runtime environment

The Mobilint SDK (`qbruntime`, `mblt_model_zoo`), `onnxruntime-gpu`,
CUDA-enabled `torch`, and `transformers` must be installed (see
`requirements.txt`). `runtime_env.sh` sets `PYTHON_BIN` and HF cache/offline
vars; source it before running anything:

```bash
source runtime_env.sh
```

## Workflow

```bash
# 1) Profile every model on CPU / GPU / NPU  ->  static profile JSON
$PYTHON_BIN profile_models.py \
    --out xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json

# 2) Generate device-placement combinations  ->  model_schedules.yaml
$PYTHON_BIN generate_schedules.py --out xgboost_model/schedules/model_schedules.yaml
cp xgboost_model/schedules/model_schedules.yaml model_schedules.yaml

# 3) Collect contention training data (headless). Writes results/performance_*.json
QT_QPA_PLATFORM=offscreen $PYTHON_BIN schedule_executor_main.py \
    --schedule model_schedules.yaml --duration 10 --auto_start_all

# 4) Train the 3-target XGBoost model (throughput FPS, drop rate, tokens/sec)
$PYTHON_BIN xgboost_model/deploy_selector_xgb_suite.py train \
    --perf_dir xgboost_model/performance_data \
    --schedule_dir xgboost_model/schedules \
    --static_json xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json \
    --model_out xgboost_model/artifacts/deploy_xgb

# 5) Predict the best combination for a schedule
$PYTHON_BIN xgboost_model/deploy_selector_xgb_suite.py predict \
    --schedule_yaml model_schedules.yaml \
    --static_json xgboost_model/performance_data/sample_profiling_data/sample_profiling_data.json \
    --model_in xgboost_model/artifacts/deploy_xgb --topk 5
```

`best_deploy_finder_executor.py` is the GUI front-end for steps 2/5 (predict best
placement, then launch the executor on the winning combination).

## Key modules

- `model_registry.py` — model set, kinds, devices, asset resolution.
- `runtime/mobilint_vision.py` — Mobilint NPU vision wrapper (detection/classification).
- `runtime/llm_engine.py` — LLM/VLM engine on GPU/CPU/NPU (tokens/sec, prefill).
- `model_processors.py` — per-view worker processes (detection/classification/LLM/VLM × cpu/gpu/npu).
- `unified_viewer.py` — orchestrates the 4-view run and aggregates per-view stats.
- `profile_models.py` — static per-model per-device profiler.
- `generate_schedules.py` — enumerates placement combinations.
- `xgboost_model/deploy_selector_xgb_suite.py` — 3-target trainer/predictor.

Device tokens everywhere are `cpu` / `gpu` / `npu`. Worker processes use the
`spawn` start method so CUDA and the NPU initialize cleanly per process.
