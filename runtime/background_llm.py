"""Isolated background LLM/VLM runner for the Q3/Q5 mixed-set experiments.

Runs generative models (llama1b, qwen2_vl) as *isolated child processes* that
loop generation continuously, creating accelerator contention against the
vision foreground. Placement (gpu/cpu/npu) is a **coarse switch** target: on a
placement change the old child is terminated and a new one is started on the new
device -- background is not a QoS/deadline target, so a brief stall during the
switch is fine (see mixed-set instruction sec.0, revised).

Deliberately NOT integrated with the vision hot-swap / QoS / ready_event path:
V(t) stays vision-only, and the vision dispatch is untouched. Default OFF (no
children spawned) so every background-off result (C3/B2/sec.4b) reproduces
exactly.

Isolation: each child is a separate ``python -m runtime.bg_entry`` process, so
torch/transformers are imported only there -- the vision onnxruntime-gpu workers
never share a process with torch's cuDNN (the isolation the vision-GPU
restoration depends on). This is subprocess-based rather than multiprocessing so
the child never re-imports the Qt/vision main module.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from typing import Callable, List, Tuple

import model_registry as reg

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class BackgroundManager:
    """Owns the isolated background generative processes.

    `sync(entries)` reconciles the running children to the desired placement for
    the current combo: it stops any model that is gone or whose device changed,
    and starts any model that is newly present or moved (the coarse switch).
    When disabled it is a total no-op, so background-off runs are unaffected.
    """

    def __init__(self, enabled: bool = False, max_new_tokens: int = 64,
                 ready_timeout: float = 240.0, log: Callable[[str], None] = print):
        self.enabled = bool(enabled)
        self.max_new_tokens = int(max_new_tokens)
        self.ready_timeout = float(ready_timeout)
        self._log = log
        # model_name -> (device, Popen, ready_path)
        self._procs = {}

    def sync(self, entries: List[Tuple[str, str]]):
        """entries: list of (model_name, execution_device) for the current combo.

        Only generative entries are acted on; vision entries are ignored (the
        vision viewer handles those). Waits for each newly started child to load
        before returning so the caller knows the background is in place.
        """
        if not self.enabled:
            return
        want = {}
        for model, dev in entries:
            try:
                if reg.is_llm_like(model):
                    want[model] = reg.norm_device(dev)
            except Exception:
                continue
        # Stop children that are no longer wanted, or moved to another device.
        for model in list(self._procs):
            dev, _proc, _ready = self._procs[model]
            if want.get(model) != dev:
                self._stop_one(model)
        # Start children that are newly wanted or were just moved.
        for model, dev in want.items():
            if model not in self._procs:
                self._start_one(model, dev)

    def _start_one(self, model: str, device: str):
        fd, ready_path = tempfile.mkstemp(prefix=f"bgllm_{model}_{device}_", suffix=".ready")
        os.close(fd)
        try:
            os.remove(ready_path)  # child (re)creates it on load; absence == not ready
        except OSError:
            pass
        cmd = [sys.executable, "-m", "runtime.bg_entry",
               "--model", model, "--device", device,
               "--ready", ready_path,
               "--max-new-tokens", str(self.max_new_tokens)]
        proc = subprocess.Popen(cmd, cwd=_REPO_ROOT, env=dict(os.environ))
        self._log(f"[bg-llm] start {model} on {device} (pid {proc.pid}); waiting for load")
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                self._log(f"[bg-llm] {model}/{device} exited early (rc={proc.returncode})")
                break
            if os.path.exists(ready_path):
                break
            time.sleep(0.5)
        else:
            self._log(f"[bg-llm] WARNING {model}/{device} not ready within "
                      f"{self.ready_timeout:.0f}s; continuing")
        self._procs[model] = (device, proc, ready_path)

    def _stop_one(self, model: str):
        dev, proc, ready_path = self._procs.pop(model)
        self._log(f"[bg-llm] stop {model}/{dev} (pid {proc.pid})")
        try:
            if proc.poll() is None:
                proc.terminate()  # SIGTERM -> child disposes engine and exits
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
        except Exception as e:
            self._log(f"[bg-llm] stop {model} error: {e}")
        finally:
            try:
                os.remove(ready_path)
            except OSError:
                pass

    def shutdown(self):
        """Terminate every child (idempotent). Safe to call when disabled."""
        for model in list(self._procs):
            self._stop_one(model)
