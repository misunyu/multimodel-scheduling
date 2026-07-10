# Changelog

## Deadline-miss metric + input-rate sweep (paper-aligned)

Adopted the input-rate methodology of the Predictive Placement paper and replaced
the drop-rate objective with a deadline-miss objective.

- **Queues:** per-view output/result queues raised 5 -> 10.
- **Metric:** `drop_rate` -> **`deadline_miss_rate`**. A request misses if its
  end-to-end latency exceeds a period-based deadline `1000/infps * DEADLINE_FACTOR`
  (factor=3 for pipeline/IPC headroom). Window value = mean of per-application
  miss rates (offered = infps*window, on-time = completions within deadline).
- **Input-rate sweep:** each placement is profiled at `baseline_rate * {1,2,3,4}`,
  where baseline = slowest-allowed-device throughput (`profile_models.py`). Train
  on 1x/2x/4x, hold out 3x (`generate_schedules.py` + `.meta.json` sidecar).
- **Frame IPC:** video frames are downscaled to 640px before crossing the
  multiprocessing queue so high input rates aren't IPC-bound (workers re-resize).
- **XGBoost:** throughput targets normalized to [0,1] per (workload, rate) group
  (T = F/Fmax); y2 = deadline miss rate; 3-fold CV grid selects hyperparameters
  (MAE); predictions clipped to [0,1]. Score `S = T + beta*T_tok - alpha*miss`,
  alpha=0.3, beta=0.5 (alpha/beta are runtime-only, not trained).
- **Evaluation:** `evaluate_model.py` adds a 3x rate hold-out and paper-style
  per-workload Top-1/Top-5 hit-rate + oracle-gap reporting.

## Neubla NPU -> Mobilint NPU + GPU migration

The scheduling app and XGBoost placement predictor were migrated from the Neubla
NPU to the **Mobilint Aries NPU + NVIDIA GPU + CPU** stack.

- **Devices:** `cpu` / `gpu` / `npu` (was cpu / npu0 / npu1). Placement features
  are `exec_cpu` / `exec_gpu` / `exec_npu`.
- **Models:** `yolo11n/s/m/l/x`, `resnet50`, `llama1b` (Llama-3.2-1B), `qwen2_vl`
  (Qwen2-VL-2B) — each runs on both GPU and NPU. (Replaces yolov3/resnet50 big/small.)
- **Runtime:** removed the Neubla `npu.py` driver; added `runtime/mobilint_vision.py`
  (NPU vision via `mblt_model_zoo`), `runtime/llm_engine.py` (LLM/VLM on GPU/CPU/NPU),
  and a GPU path via `onnxruntime-gpu`. `model_processors.py` rewritten around
  `device ∈ {cpu,gpu,npu}`; worker processes use the `spawn` start method.
- **XGBoost:** now three targets — vision throughput (FPS), drop rate, and
  generative throughput (tokens/sec). See `xgboost_model/deploy_selector_xgb_suite.py`.
- **Tooling:** `profile_models.py` (per-device profiler) and `generate_schedules.py`
  replace the Neubla GUI profiler. `model_registry.py` is the model source of truth.
- Removed Neubla eval scripts, `.o`/partition assets and stale profiling data.

See `README.md` for the current profile -> generate -> execute -> train -> predict workflow.

---

# Changes Made to Support Custom Scheduling File

## Overview
The application has been modified to accept a custom scheduling information file (e.g., model_schedules.yaml) as a command-line parameter. This allows users to specify different scheduling configurations without modifying the code.

## Changes Made

### 1. Modified `multimodel_gui.py`
- Added argument parsing to accept a scheduling file parameter
- Added `--schedule` (or `-s`) command-line option with default value of 'model_schedules.yaml'
- Modified the UnifiedViewer instantiation to pass the schedule file parameter

### 2. Modified `unified_viewer.py`
- Updated the `__init__` method to accept a `schedule_file` parameter
- Modified the `initialize_model_settings` method to use the provided schedule file
- Updated error messages to reference the actual file path being used

## Usage Examples
```bash
# Use default model_schedules.yaml
python multimodel_gui.py

# Use a custom scheduling file
python multimodel_gui.py --schedule custom_schedules.yaml

# Use short form parameter
python multimodel_gui.py -s another_schedule.yaml
```

## Testing
The changes have been tested with:
- Default behavior (no parameter provided)
- Custom schedule file using long-form parameter (--schedule)
- Custom schedule file using short-form parameter (-s)
- Error handling for non-existent files

All tests confirmed that the application correctly uses the specified scheduling file or falls back to defaults when the file cannot be loaded.