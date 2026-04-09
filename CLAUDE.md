# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-model inference scheduling platform with a PyQt5 GUI for running multiple DNN models (YOLO, ResNet, etc.) concurrently on heterogeneous devices (CPU, GPU, NPU). Uses XGBoost to predict optimal deployment combinations and ONNX Runtime for inference.

## Common Commands

```bash
# Install dependencies (GPU-enabled, requires CUDA 12)
pip install -r requirements.txt

# Run the schedule generator GUI (profiles ONNX models, generates schedule YAMLs)
python schedule_generator_main.py [--target-device target_device.yaml]

# Run the deploy finder/executor (end-to-end: generate schedules, predict best, execute)
python best_deploy_finder_executor.py [--models-root ./models]

# Run schedule executor on a single YAML schedule
python schedule_executor_main.py --schedule <schedule.yaml> --duration 30 --auto_start_all

# Run all test schedules sequentially (each YAML in tests/ directory)
./run_tests_schedules.sh [--timeout SECONDS] [SCHEDULE_DIR]
```

## Architecture

### Data Flow

1. **Schedule Generation**: User selects ONNX models -> `ScheduleGenerator` builds 2^N device-assignment combinations -> writes YAML schedule files
2. **Prediction**: Schedule YAML + device config -> `DeployPredictor` (XGBoost) -> ranks combinations by predicted throughput/QoS
3. **Execution**: `ScheduleExecutor` iterates combinations -> launches `UnifiedViewer` per combination -> model processors run inference -> real-time metrics displayed

### Key Modules

| Module | Role |
|--------|------|
| `best_deploy_finder_executor.py` | Main app: integrates generation, prediction, and execution in one GUI |
| `unified_viewer.py` | Execution viewer: coordinates model processors, view handlers, and live metrics |
| `schedule_executor_main.py` | Headless sequential executor: loads YAML, runs each combination via `UnifiedViewer` |
| `schedule_generator_app.py` | GUI for ONNX model profiling (CPU/NPU latency) and schedule YAML generation |
| `model_processors.py` | Per-device inference functions (`run_yolo_gpu_process`, `run_resnet_cpu_process`, etc.) |
| `view_handlers.py` | Display handlers (`YoloViewHandler`, `ResNetViewHandler`) with FPS/latency tracking |
| `image_processing.py` | Pre/post-processing (letterboxing, NMS, class label rendering) |
| `schedule_generator_logic.py` | `ScheduleGenerator`: builds combination schedules from model selections |
| `deploy_predictor_logic.py` | `DeployPredictor`: wraps XGBoost suite for best-combination prediction |
| `xgboost_model/deploy_selector_xgb_suite.py` | XGBoost training/prediction with multiple modes (rank, score, double, two_target) |

### Configuration Files

- `target_device.yaml` — device spec (CPU count, NPU count/IDs); read by schedule generator and executor
- `model_schedules.yaml` — auto-generated schedule with model-to-device assignments per combination
- `.ui` files — Qt Designer layouts for the various GUI windows

### Performance Logging

Model inference metrics are logged asynchronously (`utils.async_log`) to JSON Lines files under `results/`. Each record includes timestamps, FPS, latency, and run IDs. The QoS violation score is `V(t) = (1/T) · Σ_{τ=t-T+1..t} v(τ)` where `v(τ) = (1/N) · Σ_i max(0, ℓ_i(τ)/L_SLO,i − 1)` and `T = 5s`.

### Scripts Directory

`scripts/` contains standalone analysis and comparison utilities (latency-based selectors, random search, alpha sweeps, plotting). These are not part of the main application but are used for evaluation and paper figures.

## Technology Stack

- **Python 3.8+** (pyenv-managed in Docker)
- **PyQt5** for GUI (`.ui` files loaded at runtime)
- **ONNX Runtime GPU** (`onnxruntime-gpu`) for CPU/GPU inference; custom ops for NPU
- **TensorRT** for GPU optimization
- **XGBoost + scikit-learn** for deployment prediction
- **OpenCV** (headless) for image processing
- **psutil** for CPU/system metrics
- **Docker** support via `Dockerfile` + `entrypoint.sh` (Ubuntu 22.04 base)
