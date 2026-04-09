"""
Reactive deployment manager (Mode 2).

Builds on top of Mode 1 (adaptive hot-swap) and adds:
  1) Rollback: if the same model set is re-deployed with a different combination
     and V(t) rises by >= 5 after 5s stabilisation, revert to the previous combo.
  2) Fallback heuristic: if a different model set is deployed and V(t) > 40
     after 5s, generate a conservative GPU-placement using estimated inference
     cost and available GPU memory.

This module is ONLY invoked when adaptive_mode == 2.
"""
import copy
import os
import subprocess
import time
import yaml
from threading import Thread, Event

from adaptive_deploy import AdaptiveDeployManager


# ---------------------------------------------------------------------------
# GPU memory utility
# ---------------------------------------------------------------------------

def get_available_gpu_memory_mb(device_id: int = 0) -> float:
    """Return available GPU memory in MB via nvidia-smi.

    Falls back to a conservative 2048 MB if the query fails.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits", f"--id={device_id}"],
            timeout=5,
        )
        return float(out.decode().strip().split("\n")[0])
    except Exception as e:
        print(f"[ReactiveDeployManager] nvidia-smi query failed: {e}. Using fallback 2048 MB.")
        return 2048.0


# Rough per-model GPU memory estimates (MB).  These are conservative defaults
# used when no runtime profiling data is available.
_MODEL_GPU_MEM_MB = {
    "resnet50": 250,
    "resnext50": 350,
    "mnasnet": 150,
    "squeezenet1.0-12": 120,
    "vgg19": 600,
    "yolov4": 500,
    "yolov3_small": 300,
    "yolov3_big": 450,
    "shufflenet-v2-12": 100,
    "gpt2": 500,
}

_DEFAULT_GPU_MEM_MB = 200  # fallback for unknown models


def _estimate_gpu_mem(model_name: str) -> float:
    for key, val in _MODEL_GPU_MEM_MB.items():
        if key in model_name.lower():
            return val
    return _DEFAULT_GPU_MEM_MB


# ---------------------------------------------------------------------------
# Violation score helpers
# ---------------------------------------------------------------------------

def _collect_vscore(viewer, window_T: int = 5) -> float:
    """Compute V(t) = (1/T) * sum_{tau=t-T+1}^{t} v(tau) from the viewer.

    v(tau) = (1/N) * sum_i max(0, l_i(tau)/L_SLO,i - 1)
    """
    views_without = getattr(viewer, 'views_without_model', set())
    sched = [v for v in ("view1", "view2", "view3", "view4") if v not in views_without]

    v_sum = 0.0
    n_active = 0
    for vname in sched:
        handler = getattr(viewer, f"{vname}_handler", None)
        if handler is None:
            continue
        infer_ms = float(getattr(handler, 'avg_infer_time', 0.0) or 0.0)
        wait_ms = float(getattr(handler, 'avg_wait_ms', 0.0) or 0.0)
        li = infer_ms + wait_ms  # end-to-end response time
        if li <= 0:
            # Cold-starting view (no measurement yet) — skip so v(t) is not
            # diluted toward zero by views that have not produced data.
            continue
        ms = getattr(handler, 'model_settings', None) or {}
        infps = float((ms.get(vname) or {}).get('infps', 10.0) or 10.0)
        l_slo = 1000.0 / infps if infps > 0 else 100.0
        v_sum += max(0.0, (li / l_slo) - 1.0)
        n_active += 1

    v_t = (v_sum / n_active) if n_active > 0 else 0.0

    # Sliding window of v(tau) samples on the viewer (one entry per call)
    hist = getattr(viewer, '_v_history_mode2', [])
    hist.append(v_t)
    if len(hist) > window_T:
        hist[:] = hist[-window_T:]
    viewer._v_history_mode2 = hist

    return sum(hist) / len(hist) if hist else 0.0


def _collect_model_costs(viewer) -> dict:
    """Return {model_name: avg_infer_time_ms} for each scheduled view."""
    views_without = getattr(viewer, 'views_without_model', set())
    sched = [v for v in ("view1", "view2", "view3", "view4") if v not in views_without]
    costs = {}
    for vname in sched:
        handler = getattr(viewer, f"{vname}_handler", None)
        if handler is None:
            continue
        cfg = (viewer.model_settings or {}).get(vname, {})
        model_name = cfg.get("model", "")
        if not model_name:
            continue
        infer_ms = float(getattr(handler, 'avg_infer_time', 0.0) or 0.0)
        costs[model_name] = max(costs.get(model_name, 0.0), infer_ms)
    return costs


# ---------------------------------------------------------------------------
# Fallback heuristic schedule builder
# ---------------------------------------------------------------------------

def build_fallback_schedule(models_with_infps: dict, model_costs: dict,
                            gpu_mem_mb: float) -> dict:
    """Build a conservative CPU/GPU assignment using a greedy heuristic.

    Args:
        models_with_infps: {model_name: infps} for every model in the new set.
        model_costs: {model_name: avg_infer_time_ms} (higher = more expensive).
        gpu_mem_mb: Available GPU memory in MB.

    Returns:
        dict  {model_name: "cpu" or "gpu"}
    """
    # Sort by cost descending
    sorted_models = sorted(models_with_infps.keys(),
                           key=lambda m: model_costs.get(m, 0.0),
                           reverse=True)
    assignment = {}
    remaining_mem = gpu_mem_mb
    for model in sorted_models:
        est = _estimate_gpu_mem(model)
        if est <= remaining_mem:
            assignment[model] = "gpu"
            remaining_mem -= est
        else:
            assignment[model] = "cpu"
    return assignment


def _assignment_to_schedule(assignment: dict, infps_map: dict, combo_name: str) -> dict:
    """Convert a {model: device} assignment into a schedule YAML dict."""
    schedule = {combo_name: {}}
    for i, (model, device) in enumerate(assignment.items()):
        model_id = f"{model}_{device}"
        entry = {"model": model, "execution": device, "display": f"view{i+1}"}
        infps = infps_map.get(model)
        if infps is not None:
            entry["infps"] = int(infps)
        schedule[combo_name][model_id] = entry
    return schedule


# ---------------------------------------------------------------------------
# ReactiveDeployManager
# ---------------------------------------------------------------------------

class ReactiveDeployManager:
    """Mode 2 deploy manager: adaptive hot-swap + reactive rollback/fallback.

    Usage (from unified_viewer.py or schedule_executor_main.py):
        mgr = ReactiveDeployManager(viewer)
        mgr.execute(schedule_file, combination_name,
                     prev_combo_name=..., prev_model_set=...)
    """

    STABILISATION_SEC = 5       # seconds to wait before measuring V(t)
    WINDOW_T = 5                # length of the V(t) sliding window (seconds)
    ROLLBACK_DELTA = 5.0        # V(t) increase threshold for rollback
    FALLBACK_VSCORE = 40.0      # V(t) threshold for fallback heuristic

    def __init__(self, viewer):
        self.viewer = viewer

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def execute(self, schedule_file, combination_name,
                prev_combo_name=None, prev_model_set=None, prev_vscore=None):
        """Perform adaptive hot-swap, then monitor and react.

        Phase 1: Delegate to AdaptiveDeployManager for the actual hot-swap.
        Phase 2: After stabilisation, evaluate V(t):
           - Same model set, different combo → rollback if V(t) rose by >= ROLLBACK_DELTA
           - Different model set → fallback heuristic if V(t) > FALLBACK_VSCORE
        """
        v = self.viewer
        v._v_history_mode2 = []  # fresh sliding window for V(t)

        # --- Phase 1: hot-swap via AdaptiveDeployManager --------------------
        adaptive_mgr = AdaptiveDeployManager(v)
        adaptive_mgr.execute(schedule_file, combination_name)
        print(f"[ReactiveDeployManager] Phase 1 complete: hot-swap to {combination_name}")

        # Determine current model set from viewer settings
        new_model_set = set()
        for vname in ("view1", "view2", "view3", "view4"):
            m = (v.model_settings or {}).get(vname, {}).get("model", "")
            if m:
                new_model_set.add(m)

        same_models = (prev_model_set is not None and new_model_set == prev_model_set)

        # --- Phase 2: background monitor thread -----------------------------
        monitor = Thread(
            target=self._monitor_and_react,
            args=(schedule_file, combination_name,
                  prev_combo_name, same_models, prev_vscore),
            name="reactive_monitor",
            daemon=True,
        )
        monitor.start()

    # ------------------------------------------------------------------
    # Monitor thread
    # ------------------------------------------------------------------
    def _monitor_and_react(self, schedule_file, new_combo,
                           prev_combo, same_models, prev_vscore):
        """Wait for stabilisation then decide rollback or fallback."""
        v = self.viewer
        print(f"[ReactiveDeployManager] Monitoring: waiting {self.STABILISATION_SEC}s for stabilisation...")

        # Wait for stabilisation (5 seconds)
        for _ in range(self.STABILISATION_SEC):
            if not getattr(v, '_run_active', False):
                print("[ReactiveDeployManager] Run stopped during stabilisation. Aborting monitor.")
                return
            time.sleep(1.0)

        if not getattr(v, '_run_active', False):
            return

        # Measure stable V(t)
        new_vscore = _collect_vscore(v, window_T=self.WINDOW_T)
        print(f"[ReactiveDeployManager] Stable V(t) = {new_vscore:.4f} "
              f"(prev={prev_vscore}, same_models={same_models})")

        # --- Decision: same model set, different combo → rollback? ----------
        if same_models and prev_combo and prev_vscore is not None:
            delta = new_vscore - prev_vscore
            if delta >= self.ROLLBACK_DELTA:
                print(f"[ReactiveDeployManager] ROLLBACK: V(t) rose by {delta:.2f} >= {self.ROLLBACK_DELTA}. "
                      f"Reverting to {prev_combo}")
                self._do_rollback(schedule_file, prev_combo)
                return
            else:
                print(f"[ReactiveDeployManager] V(t) delta={delta:.2f} < {self.ROLLBACK_DELTA}. Keeping {new_combo}.")
                return

        # --- Decision: different model set → check threshold ----------------
        if not same_models:
            if new_vscore <= self.FALLBACK_VSCORE:
                print(f"[ReactiveDeployManager] V(t)={new_vscore:.2f} <= {self.FALLBACK_VSCORE}. "
                      f"New deployment acceptable.")
                return
            else:
                print(f"[ReactiveDeployManager] V(t)={new_vscore:.2f} > {self.FALLBACK_VSCORE}. "
                      f"Triggering fallback heuristic.")
                self._do_fallback(schedule_file, new_combo)
                return

        print(f"[ReactiveDeployManager] No action needed for {new_combo}.")

    # ------------------------------------------------------------------
    # Rollback to previous combination
    # ------------------------------------------------------------------
    def _do_rollback(self, schedule_file, prev_combo):
        # Call directly from the monitor thread — AdaptiveDeployManager only
        # touches thread-safe queues/events, no Qt widgets.
        self._apply_combo(schedule_file, prev_combo)

    # ------------------------------------------------------------------
    # Fallback heuristic
    # ------------------------------------------------------------------
    def _do_fallback(self, schedule_file, current_combo):
        v = self.viewer

        # Collect model costs from running handlers
        model_costs = _collect_model_costs(v)
        if not model_costs:
            print("[ReactiveDeployManager] No model costs available. Skipping fallback.")
            return

        # Collect infps from current settings
        infps_map = {}
        for vname in ("view1", "view2", "view3", "view4"):
            cfg = (v.model_settings or {}).get(vname, {})
            m = cfg.get("model", "")
            if m:
                infps_map[m] = cfg.get("infps", 10)

        # Query GPU memory
        gpu_mem = get_available_gpu_memory_mb()
        print(f"[ReactiveDeployManager] Available GPU memory: {gpu_mem:.0f} MB")
        print(f"[ReactiveDeployManager] Model costs: {model_costs}")

        # Build fallback assignment
        assignment = build_fallback_schedule(infps_map, model_costs, gpu_mem)
        print(f"[ReactiveDeployManager] Fallback assignment: {assignment}")

        # Generate a fallback combo name
        fallback_combo = "combination_fallback"

        # Write to a temp schedule and apply
        fallback_schedule = _assignment_to_schedule(assignment, infps_map, fallback_combo)

        # Merge into existing schedule file or write standalone
        fallback_path = os.path.join(os.path.dirname(schedule_file), "_reactive_fallback.yaml")
        try:
            with open(fallback_path, 'w', encoding='utf-8') as f:
                yaml.dump(fallback_schedule, f, default_flow_style=False)
        except Exception as e:
            print(f"[ReactiveDeployManager] Failed to write fallback schedule: {e}")
            return

        self._apply_combo(fallback_path, fallback_combo)

    # ------------------------------------------------------------------
    # Apply a combination via adaptive hot-swap (main-thread)
    # ------------------------------------------------------------------
    def _apply_combo(self, schedule_file, combo_name):
        v = self.viewer
        if not getattr(v, '_run_active', False):
            print(f"[ReactiveDeployManager] Run no longer active. Skipping apply of {combo_name}.")
            return
        print(f"[ReactiveDeployManager] Applying combination: {combo_name} from {schedule_file}")
        adaptive_mgr = AdaptiveDeployManager(v)
        adaptive_mgr.execute(schedule_file, combo_name)
