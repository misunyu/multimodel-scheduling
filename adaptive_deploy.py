"""
Adaptive deployment manager for hot-swapping model workers when devices change.

This module is ONLY invoked when the use_adaptive_deploy checkbox is checked.
It keeps models on the same device running and gracefully transitions models
whose device changes by starting the new worker in the background and
switching over once it is ready.
"""
import copy
import os
import queue
import yaml
import multiprocessing as mp
from threading import Event, Thread
from queue import Queue

from model_processors import (
    run_yolo_cpu_process,
    run_resnet_cpu_process,
    run_yolo_gpu_process,
    run_resnet_gpu_process,
    run_yolo_npu_process,
    run_resnet_npu_process,
)


def _output_queue_attr(view_name):
    """Return the attribute name used on UnifiedViewer for the output queue of a view."""
    if view_name in ("view1", "view2"):
        return f"{view_name}_output_queue"
    return f"{view_name}_result_queue"


def _same_device(old_cfg, new_cfg):
    """Return True when model AND execution device are identical."""
    if not old_cfg or not new_cfg:
        return False
    return (old_cfg.get("model", "") == new_cfg.get("model", "")
            and old_cfg.get("execution", "cpu") == new_cfg.get("execution", "cpu"))


def _create_worker_thread(view_name, cfg, frame_queue, output_queue, shutdown_event, ready_event):
    """Create (but do not start) the appropriate worker thread for *cfg*."""
    model = cfg.get("model", "")
    execution = cfg.get("execution", "cpu")

    if "yolo" in model:
        if execution == "gpu":
            return Thread(
                target=run_yolo_gpu_process,
                args=(frame_queue, output_queue, shutdown_event, view_name, model),
                kwargs={"ready_event": ready_event},
                daemon=True,
            )
        elif execution in ("npu0", "npu1"):
            npu_id = int(execution[-1])
            return mp.Process(
                target=run_yolo_npu_process,
                args=(frame_queue, output_queue, shutdown_event, npu_id, view_name, model),
            )
        else:
            return Thread(
                target=run_yolo_cpu_process,
                args=(frame_queue, output_queue, shutdown_event, view_name),
                kwargs={"ready_event": ready_event},
                daemon=True,
            )
    else:
        if execution == "gpu":
            return Thread(
                target=run_resnet_gpu_process,
                args=(frame_queue, output_queue, shutdown_event, view_name),
                kwargs={"ready_event": ready_event},
                daemon=True,
            )
        elif execution in ("npu0", "npu1"):
            npu_id = int(execution[-1])
            return mp.Process(
                target=run_resnet_npu_process,
                args=(frame_queue, output_queue, shutdown_event, npu_id, view_name, model),
            )
        else:
            return Thread(
                target=run_resnet_cpu_process,
                args=(frame_queue, output_queue, shutdown_event, view_name),
                kwargs={"ready_event": ready_event},
                daemon=True,
            )


class AdaptiveDeployManager:
    """Manages adaptive (hot-swap) deployment transitions on a UnifiedViewer.

    Usage (from unified_viewer.py):
        mgr = AdaptiveDeployManager(viewer)
        mgr.execute(schedule_file, combination_name)
    """

    # Named views whose queues/processes live as attributes on the viewer
    NAMED_VIEWS = ("view1", "view2", "view3", "view4")

    def __init__(self, viewer):
        self.viewer = viewer

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_combination(schedule_file, combination_name):
        """Parse a schedule YAML and return model_settings dict for one combination.

        Returns (settings_dict, views_without_model_set, headless_ids_list).
        This does NOT touch the viewer — it is a pure read operation.
        """
        with open(schedule_file, "r") as f:
            config = yaml.safe_load(f) or {}

        combo_key = combination_name
        if combo_key not in config and config:
            combo_key = next(iter(config))

        settings = {}
        views_without = set()
        headless_ids = []

        if combo_key in config:
            for model_config_name, model_config in (config[combo_key] or {}).items():
                if not isinstance(model_config, dict) or "display" not in model_config:
                    continue
                view_name = model_config.get("display")
                vnorm = str(view_name).strip().lower() if view_name is not None else ""

                if (not vnorm) or (vnorm in {"none", "off", "hidden", "no", "false", "0"}):
                    safe_name = str(model_config_name).replace(" ", "_")
                    hid = f"headless_{safe_name}"
                    headless_ids.append(hid)
                    settings[hid] = {
                        "model": model_config.get("model", ""),
                        "execution": model_config.get("execution", "cpu"),
                        "infps": model_config.get("infps", None),
                    }
                    continue

                if vnorm in {"view1", "view2", "view3", "view4"}:
                    settings[vnorm] = {
                        "model": model_config.get("model", ""),
                        "execution": model_config.get("execution", "cpu"),
                        "infps": model_config.get("infps", None),
                    }

        for vn in ("view1", "view2", "view3", "view4"):
            if vn not in settings:
                views_without.add(vn)
                settings[vn] = {"model": "", "execution": "cpu"}

        return settings, views_without, headless_ids, combo_key

    def execute(self, schedule_file, combination_name):
        """Perform an adaptive combination switch.

        Key difference from the non-adaptive path: we do NOT call
        viewer.initialize_model_settings() which would replace model_settings
        and views_without_model atomically, causing a brief state where
        metrics reads 0. Instead we parse the YAML independently and
        update the viewer's state incrementally, keeping kept-view
        handler references and stats intact throughout.
        """
        v = self.viewer

        # --- 1. Snapshot old state -------------------------------------------
        old_settings = copy.deepcopy(v.model_settings)
        old_yolo_views = set(getattr(v, 'yolo_views', set()))
        old_resnet_views = set(getattr(v, 'resnet_views', set()))

        # --- 2. Parse new settings WITHOUT touching the viewer ---------------
        new_settings, new_views_without, new_headless_ids, combo_key = \
            self._parse_combination(schedule_file, combination_name)

        # --- 3. Per-view comparison & hot-swap -------------------------------
        kept_views = set()
        swapped_views = set()

        for vname in self.NAMED_VIEWS:
            old_cfg = old_settings.get(vname, {})
            new_cfg = new_settings.get(vname, {})

            # View unused in both old and new
            old_unused = not old_cfg.get("model", "")
            new_unused = not new_cfg.get("model", "")
            if old_unused and new_unused:
                kept_views.add(vname)
                continue

            # View newly unused -> stop old worker
            if new_unused and not old_unused:
                self._stop_view_worker(vname)
                swapped_views.add(vname)
                continue

            # View newly used -> start fresh worker
            if old_unused and not new_unused:
                self._start_fresh_view(vname, new_cfg)
                swapped_views.add(vname)
                continue

            if _same_device(old_cfg, new_cfg):
                print(f"[AdaptiveDeploy] {vname}: same device ({new_cfg.get('execution','cpu')}), keeping worker")
                kept_views.add(vname)
            else:
                print(f"[AdaptiveDeploy] {vname}: device changed "
                      f"({old_cfg.get('execution','cpu')} -> {new_cfg.get('execution','cpu')}), hot-swapping")
                self._hot_swap_view(vname, old_cfg, new_cfg)
                swapped_views.add(vname)

        # --- 4. Incrementally update viewer state (kept views stay intact) ---
        # Update model_settings in-place: overwrite entries for swapped views,
        # keep existing entries for kept views so handler references stay
        # valid. For kept views we still refresh the per-view `infps` so that
        # an input-rate-only phase change (same placement, higher rate)
        # propagates to the V(t) SLO calculation and the feeder dispatch
        # interval. Without this the handler keeps the previous phase's
        # infps and V(t) reads ~0 even when the workers are visibly
        # overloaded.
        for vname in self.NAMED_VIEWS:
            if vname in swapped_views:
                v.model_settings[vname] = new_settings[vname]
            elif vname in kept_views and new_settings.get(vname):
                new_infps = new_settings[vname].get("infps")
                if new_infps is not None:
                    v.model_settings[vname]["infps"] = new_infps
        # Update views_without_model
        v.views_without_model = new_views_without
        # Update combination name and schedule label
        v.current_combination = combo_key
        v.schedule_file = schedule_file
        v.requested_combination = combination_name
        try:
            v.info_window.update_schedule_name(f"Current Schedule: {combo_key}")
        except Exception:
            pass
        # Update headless ids for new combination
        v.headless_ids = new_headless_ids
        # Update model_settings for headless entries
        for hid in new_headless_ids:
            if hid in new_settings:
                v.model_settings[hid] = new_settings[hid]

        # --- 5. Handle headless views (stop all old, start new) -----
        self._restart_headless(old_settings, new_settings)

        # --- 6. Update feeder yolo/resnet view-sets --------------------------
        self._update_feeders(old_yolo_views, old_resnet_views, swapped_views, new_settings)

        # --- 6b. Drain kept-view input queues so pre-transition backlog
        # frames (enqueued during the overloaded phase_b) don't leak their
        # stale enqueue timestamps into post-transition wait_ms measurements.
        # Stop-and-restart gets this drain for free by tearing down the
        # worker and its queue; we mirror that here for kept views only.
        for vname in kept_views:
            if self.viewer.model_settings.get(vname, {}).get("model"):
                self._drain(getattr(self.viewer, f"{vname}_frame_queue", None))
                self._drain(getattr(self.viewer, _output_queue_attr(vname), None))

        # --- 7. Reset handler running averages across the transition --------
        # ViewHandler.avg_infer_time / avg_wait_ms are cumulative means since
        # construction, so pre-swap overloaded samples persist indefinitely
        # for kept views (and for swapped views since the handler object is
        # reused with just its result_queue reassigned). That prevents V(t)
        # from converging to the new placement's steady state. Stop-and-
        # restart gets this reset for free because it tears down and
        # recreates every handler. Mirror that here so the two paths are
        # comparable.
        for vname in self.NAMED_VIEWS:
            handler = getattr(v, f"{vname}_handler", None)
            if handler is not None and hasattr(handler, "reset_stats"):
                try:
                    handler.reset_stats()
                except Exception as e:
                    print(f"[AdaptiveDeploy] {vname}: reset_stats failed: {e}")

        print(f"[AdaptiveDeploy] Transition complete. kept={kept_views}, swapped={swapped_views}")

    # ------------------------------------------------------------------
    # Internal: hot-swap a single named view
    # ------------------------------------------------------------------
    def _hot_swap_view(self, vname, old_cfg, new_cfg):
        """Start new worker in background; once ready, switch queues and stop old worker.

        For NPU workers: stop old worker FIRST (NPU requires exclusive
        process-level access and proper Close() before re-Init()), then
        start the new NPU process and wait for it to be ready via a
        multiprocessing.Event signalled from inside the worker after
        Init + LoadModel complete.
        """
        v = self.viewer

        # References to old resources
        old_shutdown = getattr(v, f"{vname}_shutdown_event", None)
        old_process = getattr(v, f"{vname}_process", None)
        old_frame_q = getattr(v, f"{vname}_frame_queue", None)
        old_output_q = getattr(v, _output_queue_attr(vname), None)

        # Create new resources — use multiprocessing types for NPU workers
        new_execution = new_cfg.get("execution", "cpu")
        _is_npu = new_execution in ("npu0", "npu1")
        new_frame_q = mp.Queue(maxsize=2) if _is_npu else Queue(maxsize=2)
        new_output_q = mp.Queue(maxsize=1) if _is_npu else Queue(maxsize=1)
        new_shutdown = mp.Event() if _is_npu else Event()

        if _is_npu:
            # NPU path: start new NPU process, wait for init (~8s),
            # then swap queues and stop old worker.
            # If same NPU core: must stop old worker first (Close before Init).
            old_execution = old_cfg.get("execution", "cpu")
            same_npu_core = (old_execution == new_execution)
            NPU_INIT_WAIT = 8  # seconds — NPU Init+LoadModel takes ~4-5s

            def _npu_swap():
                import time as _tw

                # If same NPU core: must Close old before Init new
                if same_npu_core:
                    print(f"[AdaptiveDeploy] {vname}: same NPU core ({new_execution}) "
                          f"— stopping old worker first (Close before Init)")
                    if old_shutdown is not None:
                        old_shutdown.set()
                    if old_process is not None and old_process.is_alive():
                        old_process.join(timeout=5.0)
                        if old_process.is_alive():
                            try:
                                old_process.terminate()
                                old_process.join(timeout=2.0)
                            except Exception:
                                pass
                    for q in (old_frame_q, old_output_q):
                        self._drain(q)
                    _tw.sleep(0.5)

                # Start new NPU worker
                print(f"[AdaptiveDeploy] {vname}: starting NPU worker ({new_execution})")
                new_process = _create_worker_thread(
                    vname, new_cfg, new_frame_q, new_output_q,
                    new_shutdown, None)
                new_process.start()

                # Wait for NPU Init + LoadModel to complete
                _tw.sleep(NPU_INIT_WAIT)
                if new_process.is_alive():
                    print(f"[AdaptiveDeploy] {vname}: NPU worker running after {NPU_INIT_WAIT}s")
                else:
                    print(f"[AdaptiveDeploy] {vname}: NPU worker died during init!")

                # Swap feeder/handler queues
                self._swap_feeder_queue(vname, new_cfg, new_frame_q)
                handler = getattr(v, f"{vname}_handler", None)
                if handler is not None:
                    handler.result_queue = new_output_q

                # Stop old worker (if not already stopped for same-core case)
                if not same_npu_core:
                    if old_shutdown is not None:
                        old_shutdown.set()
                    if old_process is not None and old_process.is_alive():
                        old_process.join(timeout=3.0)
                    for q in (old_frame_q, old_output_q):
                        self._drain(q)

                # Install new resources on the viewer
                setattr(v, f"{vname}_frame_queue", new_frame_q)
                setattr(v, _output_queue_attr(vname), new_output_q)
                setattr(v, f"{vname}_shutdown_event", new_shutdown)
                setattr(v, f"{vname}_process", new_process)

                print(f"[AdaptiveDeploy] {vname}: NPU hot-swap complete")

            watcher = Thread(target=_npu_swap,
                             name=f"adaptive_npu_swap_{vname}", daemon=True)
            watcher.start()
        else:
            # CPU/GPU path: start new first, then swap (original hot-swap)
            ready_event = Event()
            new_process = _create_worker_thread(
                vname, new_cfg, new_frame_q, new_output_q,
                new_shutdown, ready_event)
            new_process.start()

            def _watcher():
                ready_event.wait()  # blocks until model loaded
                print(f"[AdaptiveDeploy] {vname}: new worker ready, switching queues")

                # 1. Swap feeder input queue
                self._swap_feeder_queue(vname, new_cfg, new_frame_q)

                # 2. Swap handler output queue
                handler = getattr(v, f"{vname}_handler", None)
                if handler is not None:
                    handler.result_queue = new_output_q

                # 3. Stop old worker
                if old_shutdown is not None:
                    old_shutdown.set()
                if old_process is not None and old_process.is_alive():
                    old_process.join(timeout=3.0)

                # 4. Drain old queues
                for q in (old_frame_q, old_output_q):
                    self._drain(q)

                # 5. Install new resources on the viewer
                setattr(v, f"{vname}_frame_queue", new_frame_q)
                setattr(v, _output_queue_attr(vname), new_output_q)
                setattr(v, f"{vname}_shutdown_event", new_shutdown)
                setattr(v, f"{vname}_process", new_process)

                print(f"[AdaptiveDeploy] {vname}: hot-swap complete")

            watcher = Thread(target=_watcher,
                             name=f"adaptive_watcher_{vname}", daemon=True)
            watcher.start()

    # ------------------------------------------------------------------
    # Internal: stop a view worker (view becomes unused)
    # ------------------------------------------------------------------
    def _stop_view_worker(self, vname):
        v = self.viewer
        ev = getattr(v, f"{vname}_shutdown_event", None)
        proc = getattr(v, f"{vname}_process", None)
        if ev is not None:
            ev.set()
        if proc is not None and proc.is_alive():
            proc.join(timeout=3.0)
        # Drain queues
        self._drain(getattr(v, f"{vname}_frame_queue", None))
        self._drain(getattr(v, _output_queue_attr(vname), None))
        print(f"[AdaptiveDeploy] {vname}: stopped (no longer used)")

    # ------------------------------------------------------------------
    # Internal: start a fresh view worker (view newly used)
    # ------------------------------------------------------------------
    def _start_fresh_view(self, vname, cfg):
        v = self.viewer
        _is_npu = cfg.get("execution", "cpu") in ("npu0", "npu1")
        frame_q = mp.Queue(maxsize=2) if _is_npu else Queue(maxsize=2)
        output_q = mp.Queue(maxsize=1) if _is_npu else Queue(maxsize=1)
        shutdown_ev = mp.Event() if _is_npu else Event()

        setattr(v, f"{vname}_frame_queue", frame_q)
        setattr(v, _output_queue_attr(vname), output_q)
        setattr(v, f"{vname}_shutdown_event", shutdown_ev)

        proc = _create_worker_thread(vname, cfg, frame_q, output_q, shutdown_ev, ready_event=None)
        proc.start()
        setattr(v, f"{vname}_process", proc)

        # Update yolo/resnet set
        model = cfg.get("model", "")
        if "yolo" in model:
            v.yolo_views.add(vname)
            v.resnet_views.discard(vname)
        else:
            v.resnet_views.add(vname)
            v.yolo_views.discard(vname)

        print(f"[AdaptiveDeploy] {vname}: started fresh ({cfg.get('execution','cpu')})")

    # ------------------------------------------------------------------
    # Internal: restart all headless workers
    # ------------------------------------------------------------------
    def _restart_headless(self, old_settings, new_settings):
        v = self.viewer
        # Stop old headless
        for ev in (getattr(v, 'headless_shutdown_events', {}) or {}).values():
            try:
                ev.set()
            except Exception:
                pass
        for proc in (getattr(v, 'headless_processes', []) or []):
            if proc and proc.is_alive():
                proc.join(timeout=3.0)
        for q in (getattr(v, 'headless_frame_queues', {}) or {}).values():
            self._drain(q)
        for q in (getattr(v, 'headless_output_queues', {}) or {}).values():
            self._drain(q)

        # Reset headless containers
        v.headless_frame_queues = {}
        v.headless_output_queues = {}
        v.headless_shutdown_events = {}
        v.headless_processes = []

        # Start new headless workers (reuse viewer's existing logic for headless in initialize_processes)
        # We only call the headless portion; named views are already handled above.
        for hid in list(getattr(v, 'headless_ids', []) or []):
            if hid not in v.headless_frame_queues:
                v.headless_frame_queues[hid] = Queue(maxsize=2)
            if hid not in v.headless_output_queues:
                v.headless_output_queues[hid] = Queue(maxsize=1)
            if hid not in v.headless_shutdown_events:
                v.headless_shutdown_events[hid] = Event()

            cfg = v.model_settings.get(hid, {})
            proc = _create_worker_thread(
                hid, cfg,
                v.headless_frame_queues[hid],
                v.headless_output_queues[hid],
                v.headless_shutdown_events[hid],
                ready_event=None,
            )
            proc.start()
            v.headless_processes.append(proc)

            model = cfg.get("model", "")
            if "yolo" in model:
                v.yolo_views.add(hid)
            else:
                v.resnet_views.add(hid)

    # ------------------------------------------------------------------
    # Internal: update feeders after swap
    # ------------------------------------------------------------------
    def _update_feeders(self, old_yolo, old_resnet, swapped_views, new_settings):
        """Synchronise feeder view-sets and queue maps for swapped views."""
        v = self.viewer

        for vname in swapped_views:
            cfg = new_settings.get(vname, {})
            model = cfg.get("model", "")

            # Update yolo/resnet membership
            if "yolo" in model:
                v.yolo_views.add(vname)
                v.resnet_views.discard(vname)
            elif model:
                v.resnet_views.add(vname)
                v.yolo_views.discard(vname)
            else:
                v.yolo_views.discard(vname)
                v.resnet_views.discard(vname)

        # Push updated view_frame_queues into feeders
        if getattr(v, 'video_feeder', None):
            v.video_feeder.yolo_views = set(v.yolo_views)
            v.video_feeder.update_intervals(v.model_settings)
        if getattr(v, 'resnet_feeder', None):
            v.resnet_feeder.resnet_views = set(v.resnet_views)
            v.resnet_feeder.update_intervals(v.model_settings)

    # ------------------------------------------------------------------
    # Internal: swap feeder queue for a view
    # ------------------------------------------------------------------
    def _swap_feeder_queue(self, vname, cfg, new_frame_q):
        v = self.viewer
        model = cfg.get("model", "")
        if "yolo" in model:
            feeder = getattr(v, 'video_feeder', None)
        else:
            feeder = getattr(v, 'resnet_feeder', None)
        if feeder is not None:
            feeder.swap_view_queue(vname, new_frame_q)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _drain(q):
        if q is None:
            return
        try:
            while True:
                q.get_nowait()
        except Exception:
            pass
