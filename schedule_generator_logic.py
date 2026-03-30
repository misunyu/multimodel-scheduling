import os
import yaml
from typing import Optional, Dict, List, Any

class ScheduleGenerator:
    def __init__(self, log_callback=None):
        self.log_callback = log_callback

    def log(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def build_schedule_from_selection(self, checked_paths, device_conf_path: str, out_path: Optional[str] = None, input_fps_by_model=None) -> dict:
        """Generate a schedule dict (and optionally YAML) from selected top-level model folders/.onnx files and device config.
        Always builds CPU/GPU combinations.
        """
        return self._build_schedule_cpu_gpu(checked_paths, device_conf_path, out_path, input_fps_by_model)

    def _build_schedule_cpu_gpu(self, checked_paths, device_conf_path: str, out_path: Optional[str] = None, input_fps_by_model=None) -> dict:
        """CPU/GPU mode: each model -> cpu or gpu (2^N combinations)."""
        models = []
        for p in checked_paths:
            if os.path.isdir(p):
                models.append(os.path.basename(p))
            elif p.lower().endswith('.onnx'):
                models.append(os.path.splitext(os.path.basename(p))[0])
        models = [m for m in models if m]

        if not models:
            raise ValueError("No models selected (folder or .onnx file).")

        # Removed 4-model limit to allow 5+ models as per user request (2^5=32).
        # We calculate the expected number of combinations for logging/warning.
        expected_count = 2 ** len(models)
        self.log(f"[Predict] Models selected: {len(models)}, Expected combinations: {expected_count}")

        # Load device info
        try:
            with open(device_conf_path, 'r') as f:
                device_config = yaml.safe_load(f) or {}
            cpu_cfg = device_config.get("devices", {}).get("cpu", {})
            cpu_count = cpu_cfg.get("count", 1)
            gpu_cfg = device_config.get("devices", {}).get("gpu", {})
            gpu_count = gpu_cfg.get("count", 1)  # Default 1
            gpu_ids = list(gpu_cfg.get("ids", [0] if gpu_count > 0 else []))
            self.log(f"[Predict] Mode: CPU/GPU, Config: {device_conf_path}")
            self.log(f"[Predict] Detected: CPU count={cpu_count}, GPU count={gpu_count}, GPU IDs={gpu_ids}")
        except Exception as e:
            raise RuntimeError(f"Failed to load device config '{device_conf_path}': {e}")

        if gpu_count <= 0:
            self.log("[Predict] GPU is disabled by device config (count <= 0). Generating CPU-only schedules.")

        # Generate 2^N combinations
        combinations = []
        def rec(idx, assign):
            if idx >= len(models):
                combinations.append(assign.copy())
                return
            model = models[idx]
            # Option 1: CPU
            assign[model] = "cpu"
            rec(idx + 1, assign)
            # Option 2: GPU (only if enabled)
            if gpu_count > 0:
                assign[model] = "gpu"
                rec(idx + 1, assign)

        rec(0, {})
        actual_count = len(combinations)
        self.log(f"[Predict] Generated {actual_count} combinations (CPU/GPU only)")

        if len(models) == 5 and actual_count != 32:
            self.log(f"[Warning] Expected 32 combinations for 5 models, but got {actual_count}!")
        elif actual_count != expected_count:
            self.log(f"[Warning] Expected {expected_count} combinations, but got {actual_count}!")

        return self._write_schedule_yaml(models, combinations, out_path, input_fps_by_model)

    def _write_schedule_yaml(self, models, combinations, out_path: Optional[str] = None, input_fps_by_model=None) -> dict:
        schedules = {}
        for i, combo in enumerate(combinations):
            combo_name = f"combination_{i+1}"
            schedules[combo_name] = {}
            for j, (model, device) in enumerate(combo.items()):
                model_id = f"{model}_{device}"
                infps = None
                if isinstance(input_fps_by_model, dict):
                    v = input_fps_by_model.get(model)
                    try:
                        if v is not None:
                            infps = int(v)
                    except Exception:
                        infps = None
                if infps is None:
                    lname = model.lower()
                    if "resnet50" in lname: infps = 2
                    elif "yolov3" in lname: infps = 30
                    else: infps = 10
                entry = {
                    "model": model,
                    "execution": device,
                    "display": f"view{j+1}",
                }
                if infps is not None:
                    entry["infps"] = int(infps)
                schedules[combo_name][model_id] = entry
        if out_path:
            try:
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write("# model_schedules.yaml\n# Auto-generated\n\n")
                    f.write(yaml.dump(schedules, default_flow_style=False))
                self.log(f"[Predict] Saved schedule to {out_path}")
            except Exception as e:
                self.log(f"[Warn] Failed to write schedule YAML '{out_path}': {e}")
        return schedules

    def build_cpu_only_schedule(self, checked_paths, out_path: Optional[str] = None, input_fps_by_model=None) -> dict:
        """Create a schedule where all selected models run on CPU."""
        models = []
        for p in checked_paths:
            if os.path.isdir(p):
                models.append(os.path.basename(p))
            elif p.lower().endswith('.onnx'):
                models.append(os.path.splitext(os.path.basename(p))[0])
        models = sorted(list(set([m for m in models if m])))

        if not models:
            raise ValueError("No models selected for CPU-only fallback.")

        schedules = {"combination_1": {}}
        for j, model in enumerate(models):
            model_id = f"{model}_cpu"
            infps = None
            if isinstance(input_fps_by_model, dict):
                v = input_fps_by_model.get(model)
                try:
                    if v is not None: infps = int(v)
                except Exception: pass
            if infps is None:
                lname = model.lower()
                if "resnet50" in lname: infps = 2
                elif "yolov3" in lname: infps = 30
                else: infps = 10

            schedules["combination_1"][model_id] = {
                "model": model,
                "execution": "cpu",
                "display": f"view{j+1}",
                "infps": infps
            }

        if out_path:
            try:
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write("# model_schedules.yaml\n# Fallback CPU-only\n\n")
                    f.write(yaml.dump(schedules, default_flow_style=False))
                self.log(f"[Predict] Saved CPU-only schedule to {out_path}")
            except Exception as e:
                self.log(f"[Warn] Failed to write CPU-only schedule: {e}")
        return schedules
